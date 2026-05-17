from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, replace

import torch
from tqdm.auto import tqdm

from sketchawq.config import SketchAWQConfig
from sketchawq.models.layers import iter_named_linears
from sketchawq.quant.adaptive import LayerAdaptiveInput, build_layer_adaptive_plans
from sketchawq.quant.awq_scale_search import search_awq_scale_and_clip
from sketchawq.quant.hessian_sketch import build_hessian_sketch
from sketchawq.quant.observer import LayerStats, collect_layer_stats_with_summary
from sketchawq.quant.outliers import (
    apply_outlier_policy_to_candidate,
    select_outlier_channels,
)
from sketchawq.quant.pack import PackedQuantizedTensor, pack_quantized_tensor
from sketchawq.quant.quantizer import (
    dequantize_weight_with_input_scale,
    quantize_weight_with_input_scale,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LayerQuantizationSummary:
    name: str
    bits: int
    group_size: int
    adaptive_sensitivity_score: float
    adaptive_group_size_applied: bool
    sketch_mode: str
    uses_diagonal: bool
    uses_low_rank: bool
    scale_clip_search: str
    awq_alpha: float
    clip_ratio: float
    awq_scale_normalization: str
    awq_min_scale: float
    awq_max_scale: float
    search_score: float
    scale_score: float | None
    clip_score: float | None
    sketch_rank: int
    input_channels: int
    output_channels: int
    weight_numel: int
    layer_elapsed_seconds: float
    fp16_weight_mb: float
    estimated_quantized_mb: float
    estimated_compression_ratio: float
    outlier_strategy: str
    outlier_metric: str
    outlier_fraction: float
    outlier_scale_boost: float
    outlier_channels: int
    protected_channels: int


@dataclass(frozen=True, slots=True)
class PackedLayerArtifact:
    name: str
    packed_weight: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor | None
    input_scale: torch.Tensor
    protected_indices: torch.Tensor
    protected_weight: torch.Tensor | None
    bits: int
    group_size: int
    original_shape: tuple[int, ...]
    symmetric: bool

    def dequantize(self) -> torch.Tensor:
        packed = PackedQuantizedTensor(
            packed_weight=self.packed_weight,
            scale=self.scale,
            zero_point=self.zero_point,
            bits=self.bits,
            group_size=self.group_size,
            original_shape=self.original_shape,
            symmetric=self.symmetric,
        )
        weight = packed.dequantize()
        safe_scale = self.input_scale.to(device=weight.device, dtype=weight.dtype).clamp_min(1e-6)
        weight = weight / safe_scale.view(1, -1)
        if self.protected_weight is not None and self.protected_indices.numel() > 0:
            indices = self.protected_indices.to(device=weight.device)
            protected = self.protected_weight.to(device=weight.device, dtype=weight.dtype)
            weight = weight.clone()
            weight.index_copy_(1, indices, protected)
        return weight


@dataclass(frozen=True, slots=True)
class QuantizationSummary:
    method: str
    num_layers: int
    elapsed_seconds: float
    cuda_peak_memory_mb: float | None
    stats_accumulator_device: str | None
    stats_accumulator_dtype: str | None
    timings: dict[str, float | int | bool]
    layers: list[LayerQuantizationSummary]
    packed_layers: list[PackedLayerArtifact]
    artifact_format: str = "sketchawq-packed-v1"

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "artifact_format": self.artifact_format,
            "num_layers": self.num_layers,
            "elapsed_seconds": self.elapsed_seconds,
            "cuda_peak_memory_mb": self.cuda_peak_memory_mb,
            "stats_accumulator_device": self.stats_accumulator_device,
            "stats_accumulator_dtype": self.stats_accumulator_dtype,
            "timings": self.timings,
            "layers": [asdict(layer) for layer in self.layers],
        }


def quantize_model(
    model: torch.nn.Module,
    calibration_batches,
    config: SketchAWQConfig,
) -> QuantizationSummary:
    start = time.perf_counter()
    timings: dict[str, float | int | bool] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sketch_rank = config.sketch.effective_rank()
    LOGGER.info("Collecting activation statistics and Hessian sketch inputs")
    phase_start = time.perf_counter()
    stats_result = collect_layer_stats_with_summary(
        model,
        calibration_batches,
        sketch_rank=sketch_rank,
        max_rows_per_batch=config.sketch.max_rows_per_batch,
        seed=config.runtime.seed,
        row_sampling=config.sketch.row_sampling,
        mask_mismatch_action=config.sketch.mask_mismatch_action,
        activation_sample_rows=config.quantization.output_error_rows,
    )
    stats_by_name = stats_result.stats
    stats_accumulator_device, stats_accumulator_dtype = _summarize_stats_accumulators(
        stats_by_name
    )
    timings["collect_stats_seconds"] = time.perf_counter() - phase_start
    timings.update(stats_result.timings)

    summaries: list[LayerQuantizationSummary] = []
    packed_layers: list[PackedLayerArtifact] = []
    build_packed_artifacts = config.output.save_model
    phase_start = time.perf_counter()
    named_linears = list(iter_named_linears(model))
    sketches_by_name = {}
    adaptive_inputs: list[LayerAdaptiveInput] = []
    for name, module in named_linears:
        stats = stats_by_name.get(name)
        if stats is None or stats.count == 0:
            continue
        sketch = build_hessian_sketch(stats, config.sketch)
        sketches_by_name[name] = sketch
        adaptive_inputs.append(
            LayerAdaptiveInput(
                name=name,
                weight=module.weight.detach(),
                mean_abs_activation=stats.mean_abs,
                sketch=sketch,
            )
        )
    adaptive_plans = build_layer_adaptive_plans(adaptive_inputs, config)
    timings["prepare_sketches_seconds"] = time.perf_counter() - phase_start
    LOGGER.info("Quantizing %d Linear layers", len(named_linears))
    method = _method_label(config)

    phase_start = time.perf_counter()
    for name, module in tqdm(named_linears, desc="SketchAWQ layers"):
        layer_start = time.perf_counter()
        stats = stats_by_name.get(name)
        if stats is None or stats.count == 0:
            LOGGER.warning("Skipping %s because no calibration activations were observed", name)
            continue

        weight = module.weight.detach()
        sketch = sketches_by_name.get(name)
        if sketch is None:
            sketch = build_hessian_sketch(stats, config.sketch)
        adaptive_plan = adaptive_plans.get(name)
        group_size = (
            adaptive_plan.group_size
            if adaptive_plan is not None
            else config.quantization.group_size
        )
        outlier_fraction = (
            adaptive_plan.outlier_fraction
            if adaptive_plan is not None
            else config.outliers.fraction
        )
        effective_quant_config = replace(config.quantization, group_size=group_size)
        effective_outlier_config = replace(config.outliers, fraction=outlier_fraction)
        outlier_selection = select_outlier_channels(
            weight,
            stats.mean_abs,
            sketch,
            effective_outlier_config,
        )
        search_result = search_awq_scale_and_clip(
            weight,
            stats.mean_abs,
            sketch,
            effective_quant_config,
            config.sketch,
            outlier_config=effective_outlier_config,
            outlier_selection=outlier_selection,
            activation_rows=stats.activation_sample,
        )
        quantized = quantize_weight_with_input_scale(
            weight,
            bits=effective_quant_config.bits,
            group_size=effective_quant_config.group_size,
            symmetric=effective_quant_config.symmetric,
            clip_ratio=search_result.clip_ratio,
            input_scale=search_result.scales,
        )
        quantized_weight = dequantize_weight_with_input_scale(
            quantized,
            input_scale=search_result.scales,
            device=weight.device,
            dtype=weight.dtype,
        )
        quantized_weight = apply_outlier_policy_to_candidate(
            weight,
            quantized_weight,
            outlier_selection,
            effective_outlier_config,
        )
        if build_packed_artifacts:
            packed = pack_quantized_tensor(quantized)
            protected_indices = torch.empty(0, dtype=torch.long)
            protected_weight = None
            if effective_outlier_config.protects_channels() and outlier_selection.count > 0:
                protected_indices = outlier_selection.indices.detach().cpu().to(torch.long)
                protected_weight = (
                    weight.detach()
                    .index_select(1, protected_indices.to(device=weight.device))
                    .to(device="cpu")
                )
            packed_layers.append(
                PackedLayerArtifact(
                    name=name,
                    packed_weight=packed.packed_weight,
                    scale=packed.scale,
                    zero_point=packed.zero_point,
                    input_scale=search_result.scales.detach().cpu().to(torch.float32),
                    protected_indices=protected_indices,
                    protected_weight=protected_weight,
                    bits=effective_quant_config.bits,
                    group_size=effective_quant_config.group_size,
                    original_shape=tuple(weight.shape),
                    symmetric=effective_quant_config.symmetric,
                )
            )
        module.weight.data.copy_(
            quantized_weight.to(device=module.weight.device, dtype=module.weight.dtype)
        )
        layer_elapsed = time.perf_counter() - layer_start
        protected_channels = (
            outlier_selection.count if effective_outlier_config.protects_channels() else 0
        )
        quantized_mb, compression_ratio = _estimate_quantized_storage(
            module,
            bits=effective_quant_config.bits,
            group_size=effective_quant_config.group_size,
            symmetric=effective_quant_config.symmetric,
            protected_channels=protected_channels,
            protected_bits=effective_outlier_config.protected_bits,
        )

        summaries.append(
            LayerQuantizationSummary(
                name=name,
                bits=effective_quant_config.bits,
                group_size=effective_quant_config.group_size,
                adaptive_sensitivity_score=(
                    0.0
                    if adaptive_plan is None
                    else adaptive_plan.adaptive_sensitivity_score
                ),
                adaptive_group_size_applied=(
                    False
                    if adaptive_plan is None
                    else adaptive_plan.uses_sensitive_group_size
                ),
                sketch_mode=sketch.mode,
                uses_diagonal=sketch.diagonal is not None,
                uses_low_rank=sketch.basis is not None,
                scale_clip_search=effective_quant_config.scale_clip_search,
                awq_alpha=search_result.alpha,
                clip_ratio=search_result.clip_ratio,
                awq_scale_normalization=effective_quant_config.awq_scale_normalization,
                awq_min_scale=effective_quant_config.awq_min_scale,
                awq_max_scale=effective_quant_config.awq_max_scale,
                search_score=search_result.score,
                scale_score=search_result.scale_score,
                clip_score=search_result.clip_score,
                sketch_rank=sketch.rank,
                input_channels=module.in_features,
                output_channels=module.out_features,
                weight_numel=module.weight.numel(),
                layer_elapsed_seconds=layer_elapsed,
                fp16_weight_mb=module.weight.numel() * 2 / 1_000_000,
                estimated_quantized_mb=quantized_mb,
                estimated_compression_ratio=compression_ratio,
                outlier_strategy=(
                    effective_outlier_config.strategy
                    if effective_outlier_config.active()
                    else "none"
                ),
                outlier_metric=effective_outlier_config.metric,
                outlier_fraction=effective_outlier_config.fraction,
                outlier_scale_boost=effective_outlier_config.scale_boost,
                outlier_channels=outlier_selection.count,
                protected_channels=protected_channels,
            )
        )

    timings["layer_quantization_seconds"] = time.perf_counter() - phase_start
    timings["layer_elapsed_sum_seconds"] = sum(layer.layer_elapsed_seconds for layer in summaries)
    elapsed = time.perf_counter() - start
    timed_total = sum(
        timings[key]
        for key in (
            "collect_stats_seconds",
            "prepare_sketches_seconds",
            "layer_quantization_seconds",
        )
    )
    timings["other_quantization_seconds"] = max(0.0, elapsed - timed_total)
    timings["total_seconds"] = elapsed
    cuda_peak_memory_mb = None
    if torch.cuda.is_available():
        cuda_peak_memory_mb = torch.cuda.max_memory_allocated() / 1_000_000
    return QuantizationSummary(
        method=method,
        num_layers=len(summaries),
        elapsed_seconds=elapsed,
        cuda_peak_memory_mb=cuda_peak_memory_mb,
        stats_accumulator_device=stats_accumulator_device,
        stats_accumulator_dtype=stats_accumulator_dtype,
        timings=timings,
        layers=summaries,
        packed_layers=packed_layers,
    )


def _estimate_quantized_storage(
    module: torch.nn.Linear,
    *,
    bits: int,
    group_size: int,
    symmetric: bool = True,
    protected_channels: int = 0,
    protected_bits: int = 16,
) -> tuple[float, float]:
    weight_values = module.weight.numel()
    groups_per_row = math.ceil(module.in_features / group_size)
    group_count = module.out_features * groups_per_row
    quantized_weight_bytes = weight_values * bits / 8
    scale_bytes = group_count * 4
    zero_point_bytes = 0 if symmetric else group_count * 4
    protected_bytes = module.out_features * protected_channels * protected_bits / 8
    total_bytes = quantized_weight_bytes + scale_bytes + zero_point_bytes + protected_bytes
    total_mb = total_bytes / 1_000_000
    fp16_mb = weight_values * 2 / 1_000_000
    ratio = fp16_mb / max(total_mb, 1e-12)
    return total_mb, ratio


def _summarize_stats_accumulators(
    stats_by_name: dict[str, LayerStats],
) -> tuple[str | None, str | None]:
    devices = sorted(
        {
            stats.accumulator_device
            for stats in stats_by_name.values()
            if getattr(stats, "accumulator_device", None)
        }
    )
    dtypes = sorted(
        {
            stats.accumulator_dtype
            for stats in stats_by_name.values()
            if getattr(stats, "accumulator_dtype", None)
        }
    )
    return _join_unique(devices), _join_unique(dtypes)


def _join_unique(values: list[str]) -> str | None:
    if not values:
        return None
    return ";".join(values)


def _method_label(config: SketchAWQConfig) -> str:
    if not config.sketch.uses_diagonal():
        label = "AWQ-only-v0"
    elif not config.sketch.uses_low_rank():
        label = "SketchAWQ-diagonal-v0"
    else:
        label = "SketchAWQ-diagonal-low-rank-v0"
    if config.outliers.active():
        label = label.removesuffix("-v0")
        label = f"{label}-outlier-{config.outliers.strategy}-v0"
    if config.adaptive.enabled:
        label = label.removesuffix("-v0")
        return f"{label}-adaptive-v0"
    return label
