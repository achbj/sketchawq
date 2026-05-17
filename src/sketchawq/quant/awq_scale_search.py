from __future__ import annotations

from dataclasses import dataclass

import torch

from sketchawq.config import OutlierConfig, QuantizationConfig, SketchConfig
from sketchawq.quant.hessian_sketch import HessianSketch
from sketchawq.quant.loss import output_error_score, weighted_error_score
from sketchawq.quant.outliers import (
    OutlierSelection,
    _column_mean_abs,
    apply_outlier_policy_to_candidate,
    apply_outlier_scaling,
    select_outlier_channels,
)
from sketchawq.quant.quantizer import fake_quantize_weight


@dataclass(frozen=True, slots=True)
class ScaleSearchResult:
    scales: torch.Tensor
    alpha: float
    score: float


@dataclass(frozen=True, slots=True)
class ScaleClipSearchResult:
    scales: torch.Tensor
    alpha: float
    clip_ratio: float
    score: float
    scale_score: float | None
    clip_score: float | None


def search_awq_scale_and_clip(
    weight: torch.Tensor,
    mean_abs_activation: torch.Tensor,
    sketch: HessianSketch,
    quant_config: QuantizationConfig,
    sketch_config: SketchConfig,
    outlier_config: OutlierConfig | None = None,
    outlier_selection: OutlierSelection | None = None,
    activation_rows: torch.Tensor | None = None,
) -> ScaleClipSearchResult:
    """Choose AWQ input scaling and clipping jointly by default.

    Greedy search is kept as a configurable ablation, but joint search avoids the
    alpha-then-clip coupling where the best clip is only best for a previously
    fixed alpha.
    """
    outlier_config = outlier_config or OutlierConfig()
    if outlier_selection is None:
        outlier_selection = select_outlier_channels(
            weight,
            mean_abs_activation,
            sketch,
            outlier_config,
        )

    if quant_config.scale_clip_search == "greedy":
        from sketchawq.quant.clip_search import search_clip_ratio

        scale_result = search_awq_scales(
            weight,
            mean_abs_activation,
            sketch,
            quant_config,
            sketch_config,
            outlier_config=outlier_config,
            outlier_selection=outlier_selection,
            activation_rows=activation_rows,
        )
        clip_result = search_clip_ratio(
            weight,
            scale_result.scales,
            sketch,
            quant_config,
            sketch_config,
            outlier_config=outlier_config,
            outlier_selection=outlier_selection,
            activation_rows=activation_rows,
        )
        return ScaleClipSearchResult(
            scales=scale_result.scales,
            alpha=scale_result.alpha,
            clip_ratio=clip_result.clip_ratio,
            score=clip_result.score,
            scale_score=scale_result.score,
            clip_score=clip_result.score,
        )

    best: ScaleClipSearchResult | None = None
    for alpha in quant_config.awq_alpha_candidates:
        scales = make_awq_scales(
            weight,
            mean_abs_activation,
            alpha=alpha,
            normalization=quant_config.awq_scale_normalization,
            min_scale=quant_config.awq_min_scale,
            max_scale=quant_config.awq_max_scale,
        )
        scales = apply_outlier_scaling(scales, outlier_selection, outlier_config)
        for clip_ratio in quant_config.clip_candidates:
            candidate = fake_quantize_weight(
                weight,
                bits=quant_config.bits,
                group_size=quant_config.group_size,
                symmetric=quant_config.symmetric,
                clip_ratio=clip_ratio,
                input_scale=scales,
            )
            candidate = apply_outlier_policy_to_candidate(
                weight,
                candidate,
                outlier_selection,
                outlier_config,
            )
            score = _candidate_search_score(
                weight,
                candidate,
                sketch=sketch,
                quant_config=quant_config,
                sketch_config=sketch_config,
                activation_rows=activation_rows,
            )
            result = ScaleClipSearchResult(
                scales=scales.cpu(),
                alpha=float(alpha),
                clip_ratio=float(clip_ratio),
                score=score,
                scale_score=None,
                clip_score=None,
            )
            if best is None or result.score < best.score:
                best = result
    if best is None:
        raise RuntimeError("No AWQ scale/clip candidates were evaluated.")
    return best


def search_awq_scales(
    weight: torch.Tensor,
    mean_abs_activation: torch.Tensor,
    sketch: HessianSketch,
    quant_config: QuantizationConfig,
    sketch_config: SketchConfig,
    outlier_config: OutlierConfig | None = None,
    outlier_selection: OutlierSelection | None = None,
    activation_rows: torch.Tensor | None = None,
) -> ScaleSearchResult:
    """Choose an AWQ-style input-channel scale using a tiny candidate search."""
    outlier_config = outlier_config or OutlierConfig()
    if outlier_selection is None:
        outlier_selection = select_outlier_channels(
            weight,
            mean_abs_activation,
            sketch,
            outlier_config,
        )
    best: ScaleSearchResult | None = None
    for alpha in quant_config.awq_alpha_candidates:
        scales = make_awq_scales(
            weight,
            mean_abs_activation,
            alpha=alpha,
            normalization=quant_config.awq_scale_normalization,
            min_scale=quant_config.awq_min_scale,
            max_scale=quant_config.awq_max_scale,
        )
        scales = apply_outlier_scaling(scales, outlier_selection, outlier_config)
        candidate = fake_quantize_weight(
            weight,
            bits=quant_config.bits,
            group_size=quant_config.group_size,
            symmetric=quant_config.symmetric,
            clip_ratio=1.0,
            input_scale=scales,
        )
        candidate = apply_outlier_policy_to_candidate(
            weight,
            candidate,
            outlier_selection,
            outlier_config,
        )
        score = _candidate_search_score(
            weight,
            candidate,
            sketch=sketch,
            quant_config=quant_config,
            sketch_config=sketch_config,
            activation_rows=activation_rows,
        )
        result = ScaleSearchResult(scales=scales.cpu(), alpha=float(alpha), score=score)
        if best is None or result.score < best.score:
            best = result
    if best is None:
        raise RuntimeError("No AWQ scale candidates were evaluated.")
    return best


def make_awq_scales(
    weight: torch.Tensor,
    mean_abs_activation: torch.Tensor,
    *,
    alpha: float,
    normalization: str = "geometric_mean",
    min_scale: float = 0.25,
    max_scale: float = 4.0,
) -> torch.Tensor:
    """Build equivalent input-channel scales from activation and weight saliency.

    The numerator protects channels with large activations. The denominator avoids
    over-scaling columns whose weights are already large.
    """
    device = weight.device
    dtype = torch.float32
    activation = mean_abs_activation.to(device=device, dtype=dtype).clamp_min(1e-8)
    activation = activation / activation.mean().clamp_min(1e-8)

    weight_strength = _column_mean_abs(weight).clamp_min(1e-8)
    weight_strength = weight_strength / weight_strength.mean().clamp_min(1e-8)

    alpha = float(alpha)
    scales = activation.pow(alpha) / weight_strength.pow(max(1.0 - alpha, 0.0))
    scales = scales.clamp(min_scale, max_scale)
    scales = _normalize_awq_scales(scales, normalization)
    return scales.to(device=device, dtype=weight.dtype)


def _normalize_awq_scales(scales: torch.Tensor, normalization: str) -> torch.Tensor:
    mode = normalization.strip().lower().replace("-", "_")
    aliases = {
        "geometric": "geometric_mean",
        "geomean": "geometric_mean",
        "geo_mean": "geometric_mean",
        "mean": "arithmetic_mean",
        "arithmetic": "arithmetic_mean",
        "arith_mean": "arithmetic_mean",
        "raw": "none",
        "off": "none",
        "false": "none",
    }
    mode = aliases.get(mode, mode)
    if mode == "geometric_mean":
        return scales / scales.log().mean().exp().clamp_min(1e-8)
    if mode == "arithmetic_mean":
        return scales / scales.mean().clamp_min(1e-8)
    if mode == "none":
        return scales
    raise ValueError("normalization must be one of arithmetic_mean, geometric_mean, none")


def _candidate_search_score(
    weight: torch.Tensor,
    candidate: torch.Tensor,
    *,
    sketch: HessianSketch,
    quant_config: QuantizationConfig,
    sketch_config: SketchConfig,
    activation_rows: torch.Tensor | None = None,
) -> float:
    weighted_score = None
    if quant_config.search_objective in {"weighted_error", "hybrid"}:
        weighted_score = weighted_error_score(
            weight,
            candidate,
            diagonal=sketch.diagonal,
            basis=sketch.basis,
            low_rank_weight=sketch_config.low_rank_weight,
            normalize_terms=sketch_config.normalize_error_terms,
        )
    if quant_config.search_objective == "weighted_error":
        assert weighted_score is not None
        return float(weighted_score)

    if activation_rows is None or activation_rows.numel() == 0:
        if weighted_score is not None:
            return float(weighted_score)
        return weighted_error_score(
            weight,
            candidate,
            diagonal=sketch.diagonal,
            basis=sketch.basis,
            low_rank_weight=sketch_config.low_rank_weight,
            normalize_terms=sketch_config.normalize_error_terms,
        )

    output_score = output_error_score(
        weight,
        candidate,
        activation_rows,
        normalize_terms=sketch_config.normalize_error_terms,
    )
    if quant_config.search_objective == "output_error":
        return float(output_score)

    assert weighted_score is not None
    return float(
        quant_config.hessian_error_weight * weighted_score
        + quant_config.output_error_weight * output_score
    )
