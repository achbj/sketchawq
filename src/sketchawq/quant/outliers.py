from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sketchawq.config import OutlierConfig
from sketchawq.quant.hessian_sketch import HessianSketch


@dataclass(frozen=True, slots=True)
class OutlierSelection:
    indices: torch.Tensor
    mask: torch.Tensor
    scores: torch.Tensor
    threshold: float | None

    @property
    def count(self) -> int:
        return int(self.indices.numel())


def select_outlier_channels(
    weight: torch.Tensor,
    mean_abs_activation: torch.Tensor,
    sketch: HessianSketch,
    config: OutlierConfig,
) -> OutlierSelection:
    """Select input channels that deserve extra quantization care.

    Selection is per linear layer and independent of output rows. That keeps the
    policy compatible with AWQ-style equivalent scaling and, for the protected
    variant, gives a simple side-channel storage estimate.
    """
    input_channels = weight.shape[1]
    scores = score_outlier_channels(weight, mean_abs_activation, sketch, config.metric)
    k = _channel_budget(input_channels, config)
    if not config.active() or k == 0:
        return _empty_selection(input_channels, scores)

    k = min(k, input_channels)
    top = torch.topk(scores, k=k, largest=True, sorted=False)
    indices = top.indices.sort().values.detach().cpu()
    mask = torch.zeros(input_channels, dtype=torch.bool)
    mask[indices] = True
    selected_scores = scores.detach().cpu().index_select(0, indices)
    threshold = float(selected_scores.min().item()) if selected_scores.numel() else None
    return OutlierSelection(
        indices=indices,
        mask=mask,
        scores=scores.detach().cpu(),
        threshold=threshold,
    )


def score_outlier_channels(
    weight: torch.Tensor,
    mean_abs_activation: torch.Tensor,
    sketch: HessianSketch,
    metric: str,
) -> torch.Tensor:
    return _outlier_scores(weight, mean_abs_activation, sketch, metric)


def apply_outlier_scaling(
    scales: torch.Tensor,
    selection: OutlierSelection,
    config: OutlierConfig,
) -> torch.Tensor:
    """Boost selected AWQ scales so outlier columns receive a smaller effective step."""
    if not config.active() or config.strategy != "scale" or selection.count == 0:
        return scales

    adjusted = scales.clone()
    indices = selection.indices.to(device=adjusted.device)
    adjusted.index_copy_(0, indices, adjusted.index_select(0, indices) * float(config.scale_boost))
    if config.renormalize_scales:
        adjusted = adjusted / adjusted.float().log().mean().exp().to(
            device=adjusted.device,
            dtype=adjusted.dtype,
        ).clamp_min(1e-8)
    return adjusted


def restore_outlier_channels(
    original: torch.Tensor,
    candidate: torch.Tensor,
    selection: OutlierSelection,
) -> torch.Tensor:
    """Restore selected columns exactly for protected-channel ablations."""
    if selection.count == 0:
        return candidate
    restored = candidate.clone()
    indices = selection.indices.to(device=restored.device)
    restored.index_copy_(1, indices, original.to(device=restored.device).index_select(1, indices))
    return restored


def apply_outlier_policy_to_candidate(
    original: torch.Tensor,
    candidate: torch.Tensor,
    selection: OutlierSelection,
    config: OutlierConfig,
) -> torch.Tensor:
    if config.protects_channels():
        return restore_outlier_channels(original, candidate, selection)
    return candidate


def _outlier_scores(
    weight: torch.Tensor,
    mean_abs_activation: torch.Tensor,
    sketch: HessianSketch,
    metric: str,
) -> torch.Tensor:
    device = weight.device
    activation = mean_abs_activation.to(device=device, dtype=torch.float32).clamp_min(0)
    hessian = sketch.diagonal
    if hessian is None:
        hessian_score = activation.pow(2)
    else:
        hessian_score = hessian.to(device=device, dtype=torch.float32).clamp_min(0)
    weight_score = _column_mean_abs(weight).clamp_min(0)

    if metric == "activation":
        return _normalize(activation)
    if metric == "hessian":
        return _normalize(hessian_score)
    if metric == "weight":
        return _normalize(weight_score)
    if metric == "activation_weight":
        return _normalize(activation) * _normalize(weight_score)
    if metric == "hessian_weight":
        return _normalize(hessian_score) * _normalize(weight_score)
    raise ValueError(f"Unsupported outlier metric: {metric}")


def _normalize(values: torch.Tensor) -> torch.Tensor:
    values = values.float().clamp_min(0)
    return values / values.mean().clamp_min(1e-8)


def _column_mean_abs(weight: torch.Tensor, row_chunk_size: int = 4096) -> torch.Tensor:
    """Compute per-input-column mean abs without a full FP32 copy of large weights."""
    output_channels, input_channels = weight.shape
    if output_channels == 0:
        return torch.zeros(input_channels, device=weight.device, dtype=torch.float32)

    totals = torch.zeros(input_channels, device=weight.device, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, output_channels, row_chunk_size):
            chunk = weight.detach()[start : start + row_chunk_size]
            totals.add_(chunk.to(dtype=torch.float32).abs().sum(dim=0))
    return totals / float(output_channels)


def _channel_budget(input_channels: int, config: OutlierConfig) -> int:
    budget = int(math.ceil(input_channels * config.fraction))
    budget = max(budget, config.min_channels)
    if config.max_channels is not None:
        budget = min(budget, config.max_channels)
    return max(0, min(input_channels, budget))


def _empty_selection(input_channels: int, scores: torch.Tensor) -> OutlierSelection:
    return OutlierSelection(
        indices=torch.empty(0, dtype=torch.long),
        mask=torch.zeros(input_channels, dtype=torch.bool),
        scores=scores.detach().cpu(),
        threshold=None,
    )
