from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from sketchawq.models.layers import iter_named_linears

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LayerStats:
    name: str
    in_features: int
    abs_sum: torch.Tensor
    sq_sum: torch.Tensor
    count: int = 0
    sketch_omega: torch.Tensor | None = None
    sketch_sum: torch.Tensor | None = None
    sketch_count: int = 0
    activation_sample: torch.Tensor | None = None
    accumulator_device: str = "cpu"
    accumulator_dtype: str = "float32"

    @classmethod
    def create(
        cls,
        name: str,
        in_features: int,
        sketch_rank: int,
        seed: int,
        *,
        device: torch.device | str | None = None,
    ) -> "LayerStats":
        accumulator_device = torch.device(device) if device is not None else torch.device("cpu")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        omega = None
        sketch_sum = None
        if sketch_rank > 0:
            omega = torch.randn(in_features, sketch_rank, generator=generator, dtype=torch.float32)
            omega = omega / max(float(sketch_rank) ** 0.5, 1.0)
            omega = omega.to(device=accumulator_device)
            sketch_sum = torch.zeros(
                in_features,
                sketch_rank,
                dtype=torch.float32,
                device=accumulator_device,
            )
        return cls(
            name=name,
            in_features=in_features,
            abs_sum=torch.zeros(in_features, dtype=torch.float32, device=accumulator_device),
            sq_sum=torch.zeros(in_features, dtype=torch.float32, device=accumulator_device),
            sketch_omega=omega,
            sketch_sum=sketch_sum,
            accumulator_device=accumulator_device.type,
            accumulator_dtype=_dtype_label(torch.float32),
        )

    @property
    def mean_abs(self) -> torch.Tensor:
        return self.abs_sum.float() / max(self.count, 1)

    @property
    def mean_square(self) -> torch.Tensor:
        return self.sq_sum.float() / max(self.count, 1)

    @property
    def hessian_times_omega(self) -> torch.Tensor | None:
        if self.sketch_sum is None or self.sketch_count == 0:
            return None
        return (self.sketch_sum / self.sketch_count).float()

    def move_to_cpu(self) -> None:
        self.abs_sum = self.abs_sum.detach().cpu()
        self.sq_sum = self.sq_sum.detach().cpu()
        if self.sketch_omega is not None:
            self.sketch_omega = self.sketch_omega.detach().cpu()
        if self.sketch_sum is not None:
            self.sketch_sum = self.sketch_sum.detach().cpu()
        if self.activation_sample is not None:
            self.activation_sample = self.activation_sample.detach().cpu()


@dataclass(frozen=True, slots=True)
class StatsCollectionResult:
    stats: dict[str, LayerStats]
    timings: dict[str, float | int | bool]


class ActivationObserver:
    """Collect activation saliency and a streaming Hessian sketch for Linear layers."""

    def __init__(
        self,
        *,
        sketch_rank: int,
        max_rows_per_batch: int,
        seed: int,
        row_sampling: str = "random",
        mask_mismatch_action: str = "warn",
        activation_sample_rows: int = 0,
    ) -> None:
        self.sketch_rank = sketch_rank
        self.max_rows_per_batch = max_rows_per_batch
        self.seed = seed
        self.row_sampling = _normalize_row_sampling(row_sampling)
        self.mask_mismatch_action = _normalize_mask_mismatch_action(mask_mismatch_action)
        self.activation_sample_rows = max(int(activation_sample_rows), 0)
        self.stats: dict[str, LayerStats] = {}
        self._row_generators: dict[str, torch.Generator] = {}
        self._sample_generators: dict[str, torch.Generator] = {}
        self._handles: list[Any] = []
        self._attention_mask: torch.Tensor | None = None

    def attach(self, model: torch.nn.Module) -> None:
        for layer_index, (name, module) in enumerate(iter_named_linears(model)):
            self.stats[name] = LayerStats.create(
                name=name,
                in_features=module.in_features,
                sketch_rank=self.sketch_rank,
                seed=self.seed + layer_index,
                device=module.weight.device,
            )
            generator_device = _generator_device(module.weight.device)
            row_generator = torch.Generator(device=generator_device)
            row_generator.manual_seed(self.seed + 10_000 + layer_index)
            self._row_generators[name] = row_generator
            sample_generator = torch.Generator(device=generator_device)
            sample_generator.manual_seed(self.seed + 20_000 + layer_index)
            self._sample_generators[name] = sample_generator
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._attention_mask = None

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        self._attention_mask = attention_mask

    def _make_hook(self, name: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            del module, output
            if not inputs:
                return
            activation = inputs[0].detach()
            if activation.shape[-1] != self.stats[name].in_features:
                return
            flat = activation.reshape(-1, activation.shape[-1]).float()
            flat = _select_unmasked_rows(
                flat,
                activation,
                self._attention_mask,
                mask_mismatch_action=self.mask_mismatch_action,
            )
            if flat.numel() == 0:
                return

            stats = self.stats[name]
            stats.abs_sum.add_(flat.abs().sum(dim=0).to(device=stats.abs_sum.device))
            stats.sq_sum.add_(flat.pow(2).sum(dim=0).to(device=stats.sq_sum.device))
            stats.count += flat.shape[0]

            if self.activation_sample_rows > 0:
                rows = _sample_rows(
                    flat,
                    self.activation_sample_rows,
                    method=self.row_sampling,
                    generator=self._sample_generators.get(name),
                )
                if stats.activation_sample is None:
                    stats.activation_sample = rows
                else:
                    combined = torch.cat([stats.activation_sample, rows], dim=0)
                    stats.activation_sample = _sample_rows(
                        combined,
                        self.activation_sample_rows,
                        method=self.row_sampling,
                        generator=self._sample_generators.get(name),
                    )

            if stats.sketch_omega is not None and stats.sketch_sum is not None:
                rows = _sample_rows(
                    flat,
                    self.max_rows_per_batch,
                    method=self.row_sampling,
                    generator=self._row_generators.get(name),
                )
                omega = stats.sketch_omega.to(device=rows.device, dtype=rows.dtype)
                projected = rows @ omega
                update = rows.transpose(0, 1) @ projected
                stats.sketch_sum.add_(update.to(device=stats.sketch_sum.device))
                stats.sketch_count += rows.shape[0]

        return hook


def _select_unmasked_rows(
    flat: torch.Tensor,
    activation: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    mask_mismatch_action: str = "warn",
) -> torch.Tensor:
    if attention_mask is None:
        return flat

    action = _normalize_mask_mismatch_action(mask_mismatch_action)
    mask = attention_mask.to(device=activation.device)
    if tuple(mask.shape) == tuple(activation.shape[:-1]):
        flat_mask = mask.reshape(-1).bool()
    elif mask.numel() == flat.shape[0]:
        flat_mask = mask.reshape(-1).bool()
    else:
        message = (
            f"attention_mask shape {tuple(mask.shape)} does not match activation prefix "
            f"{tuple(activation.shape[:-1])} or flattened row count {flat.shape[0]}; "
            "using all activation rows"
        )
        if action == "error":
            raise ValueError(message)
        if action == "warn":
            LOGGER.warning("%s", message)
        return flat

    if not flat_mask.any():
        return flat[:0]
    return flat.index_select(0, flat_mask.nonzero(as_tuple=False).squeeze(-1))


def _sample_rows(
    flat: torch.Tensor,
    max_rows: int,
    *,
    method: str = "random",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if flat.shape[0] <= max_rows:
        return flat
    method = _normalize_row_sampling(method)
    if method == "linspace":
        indices = torch.linspace(0, flat.shape[0] - 1, steps=max_rows, device=flat.device).long()
        return flat.index_select(0, indices)

    if generator is None:
        indices = torch.randperm(flat.shape[0], device=flat.device)[:max_rows]
    else:
        try:
            indices = torch.randperm(
                flat.shape[0],
                generator=generator,
                device=flat.device,
            )[:max_rows]
        except RuntimeError:
            indices = torch.randperm(flat.shape[0], generator=generator)[:max_rows].to(flat.device)
    indices = indices.sort().values
    return flat.index_select(0, indices)


def _generator_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{device.index}" if device.index is not None else "cuda"
    return "cpu"


def _dtype_label(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _normalize_row_sampling(method: str) -> str:
    normalized = method.strip().lower().replace("-", "_")
    aliases = {
        "seeded_random": "random",
        "randperm": "random",
        "stride": "linspace",
        "uniform_stride": "linspace",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"random", "linspace"}:
        raise ValueError("row_sampling must be one of linspace, random")
    return normalized


def _normalize_mask_mismatch_action(action: str) -> str:
    normalized = action.strip().lower().replace("-", "_")
    aliases = {
        "raise": "error",
        "strict": "error",
        "silent": "ignore",
        "off": "ignore",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"warn", "error", "ignore"}:
        raise ValueError("mask_mismatch_action must be one of error, ignore, warn")
    return normalized


def collect_layer_stats(
    model: torch.nn.Module,
    calibration_batches,
    *,
    sketch_rank: int,
    max_rows_per_batch: int,
    seed: int,
    row_sampling: str = "random",
    mask_mismatch_action: str = "warn",
    activation_sample_rows: int = 0,
) -> dict[str, LayerStats]:
    return collect_layer_stats_with_summary(
        model,
        calibration_batches,
        sketch_rank=sketch_rank,
        max_rows_per_batch=max_rows_per_batch,
        seed=seed,
        row_sampling=row_sampling,
        mask_mismatch_action=mask_mismatch_action,
        activation_sample_rows=activation_sample_rows,
    ).stats


def collect_layer_stats_with_summary(
    model: torch.nn.Module,
    calibration_batches,
    *,
    sketch_rank: int,
    max_rows_per_batch: int,
    seed: int,
    row_sampling: str = "random",
    mask_mismatch_action: str = "warn",
    activation_sample_rows: int = 0,
) -> StatsCollectionResult:
    observer = ActivationObserver(
        sketch_rank=sketch_rank,
        max_rows_per_batch=max_rows_per_batch,
        seed=seed,
        row_sampling=row_sampling,
        mask_mismatch_action=mask_mismatch_action,
        activation_sample_rows=activation_sample_rows,
    )
    batches = 0
    tokens = 0
    all_ones_mask_batches = 0
    masked_batches = 0
    use_cache_disabled = _model_accepts_use_cache(model)
    observer.attach(model)
    try:
        model.eval()
        _synchronize_model_device(model)
        forward_start = time.perf_counter()
        with torch.inference_mode():
            for batch in calibration_batches:
                kwargs = _as_model_kwargs(batch)
                attention_mask = kwargs.get("attention_mask")
                all_ones_mask, mask_tokens = _attention_mask_status(attention_mask)
                tokens += mask_tokens if mask_tokens is not None else _input_token_count(kwargs)
                if all_ones_mask:
                    all_ones_mask_batches += 1
                    observer.set_attention_mask(None)
                else:
                    if attention_mask is not None:
                        masked_batches += 1
                    observer.set_attention_mask(attention_mask)
                if use_cache_disabled:
                    kwargs["use_cache"] = False
                try:
                    model(**kwargs)
                finally:
                    observer.set_attention_mask(None)
                batches += 1
        _synchronize_model_device(model)
        forward_seconds = time.perf_counter() - forward_start
    finally:
        observer.remove()
    for stats in observer.stats.values():
        stats.move_to_cpu()
    return StatsCollectionResult(
        stats=observer.stats,
        timings={
            "stats_batches": batches,
            "stats_tokens": tokens,
            "stats_forward_seconds": forward_seconds,
            "stats_tokens_per_second": _safe_rate(tokens, forward_seconds),
            "stats_use_cache_disabled": use_cache_disabled,
            "stats_attention_mask_fast_path_batches": all_ones_mask_batches,
            "stats_masked_batches": masked_batches,
        },
    )


def _as_model_kwargs(batch: Any) -> dict[str, Any]:
    if hasattr(batch, "as_model_kwargs"):
        return dict(batch.as_model_kwargs())
    if isinstance(batch, Mapping):
        return dict(batch)
    return dict(batch)


def _model_accepts_use_cache(model: torch.nn.Module) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.name == "use_cache" or parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _attention_mask_status(mask: Any) -> tuple[bool, int | None]:
    if mask is None or not isinstance(mask, torch.Tensor):
        return False, None
    if mask.numel() == 0:
        return False, 0
    token_count = int(mask.detach().sum().item())
    return token_count == mask.numel(), token_count


def _input_token_count(kwargs: dict[str, Any]) -> int:
    input_ids = kwargs.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.numel())
    return 0


def _safe_rate(count: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return float(count) / float(seconds)


def _synchronize_model_device(model: torch.nn.Module) -> None:
    device = _first_model_device(model)
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _first_model_device(model: torch.nn.Module) -> torch.device | None:
    for parameter in model.parameters(recurse=True):
        return parameter.device
    for buffer in model.buffers(recurse=True):
        return buffer.device
    return None
