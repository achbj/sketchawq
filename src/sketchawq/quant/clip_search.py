from __future__ import annotations

from dataclasses import dataclass

import torch

from sketchawq.config import OutlierConfig, QuantizationConfig, SketchConfig
from sketchawq.quant.awq_scale_search import (
    _candidate_search_score,
    _fake_quantize_candidate,
    _tied_embedding_score,
)
from sketchawq.quant.hessian_sketch import HessianSketch
from sketchawq.quant.outliers import (
    OutlierSelection,
    apply_outlier_policy_to_candidate,
)


@dataclass(frozen=True, slots=True)
class ClipSearchResult:
    clip_ratio: float
    score: float
    tied_embedding_score: float | None


def search_clip_ratio(
    weight: torch.Tensor,
    input_scales: torch.Tensor,
    sketch: HessianSketch,
    quant_config: QuantizationConfig,
    sketch_config: SketchConfig,
    outlier_config: OutlierConfig | None = None,
    outlier_selection: OutlierSelection | None = None,
    activation_rows: torch.Tensor | None = None,
    tied_embedding_row_weights: torch.Tensor | None = None,
) -> ClipSearchResult:
    outlier_config = outlier_config or OutlierConfig()
    best: ClipSearchResult | None = None
    for clip_ratio in quant_config.clip_candidates:
        candidate = _fake_quantize_candidate(
            weight,
            scales=input_scales,
            clip_ratio=clip_ratio,
            sketch=sketch,
            quant_config=quant_config,
            sketch_config=sketch_config,
        )
        if outlier_selection is not None:
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
            tied_embedding_row_weights=tied_embedding_row_weights,
            tied_embedding_error_weight=quant_config.tied_embedding_error_weight,
        )
        result = ClipSearchResult(
            clip_ratio=float(clip_ratio),
            score=score,
            tied_embedding_score=_tied_embedding_score(
                weight,
                candidate,
                row_weights=tied_embedding_row_weights,
            ),
        )
        if best is None or result.score < best.score:
            best = result
    if best is None:
        raise RuntimeError("No clip candidates were evaluated.")
    return best
