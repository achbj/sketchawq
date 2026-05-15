from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from sketchawq.config import SketchAWQConfig
from sketchawq.quant.hessian_sketch import HessianSketch
from sketchawq.quant.outliers import score_outlier_channels


@dataclass(frozen=True, slots=True)
class LayerAdaptiveInput:
    name: str
    weight: torch.Tensor
    mean_abs_activation: torch.Tensor
    sketch: HessianSketch


@dataclass(frozen=True, slots=True)
class LayerAdaptivePlan:
    name: str
    group_size: int
    outlier_fraction: float
    adaptive_sensitivity_score: float
    uses_sensitive_group_size: bool


def build_layer_adaptive_plans(
    layers: Sequence[LayerAdaptiveInput],
    config: SketchAWQConfig,
) -> dict[str, LayerAdaptivePlan]:
    layer_list = list(layers)
    if not layer_list:
        return {}

    sensitivity_scores = {
        item.name: _layer_sensitivity_score(item, config)
        for item in layer_list
    }
    sensitive_group_layers = _sensitive_group_layers(sensitivity_scores, config)
    baseline_sensitivity = _mean_positive(sensitivity_scores.values())

    plans: dict[str, LayerAdaptivePlan] = {}
    for item in layer_list:
        uses_sensitive_group_size = item.name in sensitive_group_layers
        group_size = (
            config.adaptive.sensitive_group_size
            if uses_sensitive_group_size and config.adaptive.sensitive_group_size is not None
            else config.quantization.group_size
        )
        sensitivity_score = sensitivity_scores[item.name]
        plans[item.name] = LayerAdaptivePlan(
            name=item.name,
            group_size=int(group_size),
            outlier_fraction=_adaptive_outlier_fraction(
                sensitivity_score,
                baseline_sensitivity,
                config,
            ),
            adaptive_sensitivity_score=sensitivity_score,
            uses_sensitive_group_size=uses_sensitive_group_size,
        )
    return plans


def _layer_sensitivity_score(item: LayerAdaptiveInput, config: SketchAWQConfig) -> float:
    scores = score_outlier_channels(
        item.weight,
        item.mean_abs_activation,
        item.sketch,
        config.adaptive.sensitivity_metric,
    ).detach().float()
    if scores.numel() == 0:
        return 0.0
    scores = scores[torch.isfinite(scores)]
    if scores.numel() == 0:
        return 0.0
    top_k = max(1, math.ceil(scores.numel() * config.adaptive.sensitivity_top_fraction))
    top_values = torch.topk(
        scores,
        k=min(top_k, scores.numel()),
        largest=True,
    ).values
    return float(top_values.mean().item())


def _sensitive_group_layers(
    sensitivity_scores: dict[str, float],
    config: SketchAWQConfig,
) -> set[str]:
    if (
        not config.adaptive.enabled
        or config.adaptive.sensitive_group_size is None
        or config.adaptive.group_size_top_fraction <= 0.0
    ):
        return set()

    count = math.ceil(len(sensitivity_scores) * config.adaptive.group_size_top_fraction)
    if count <= 0:
        return set()
    ranked = sorted(
        sensitivity_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return {name for name, _ in ranked[: min(count, len(ranked))]}


def _adaptive_outlier_fraction(
    sensitivity_score: float,
    baseline_sensitivity: float,
    config: SketchAWQConfig,
) -> float:
    base_fraction = float(config.outliers.fraction)
    if not config.adaptive.enabled or not config.outliers.active():
        return base_fraction

    max_fraction = config.adaptive.protected_fraction_max
    if max_fraction is None:
        max_fraction = base_fraction
    if baseline_sensitivity <= 0.0:
        raw_fraction = base_fraction
    else:
        raw_fraction = base_fraction * max(sensitivity_score, 0.0) / baseline_sensitivity
    return float(
        min(
            max(raw_fraction, config.adaptive.protected_fraction_min),
            max_fraction,
        )
    )


def _mean_positive(values) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return 0.0
    return float(sum(values) / len(values))
