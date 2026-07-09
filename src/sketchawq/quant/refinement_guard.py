from __future__ import annotations

from dataclasses import dataclass

import torch

from sketchawq.quant.loss import output_error_score, row_weighted_error_score


@dataclass(frozen=True, slots=True)
class HoldoutCandidateScore:
    total: float
    output: float
    tied_embedding: float | None


@dataclass(frozen=True, slots=True)
class HoldoutRefinementSelection:
    baseline: HoldoutCandidateScore
    refined: HoldoutCandidateScore
    accepts_refined: bool


def score_holdout_output_candidate(
    weight: torch.Tensor,
    candidate: torch.Tensor,
    activation_rows: torch.Tensor,
    *,
    tied_embedding_row_weights: torch.Tensor | None = None,
    tied_embedding_error_weight: float = 1.0,
) -> HoldoutCandidateScore:
    """Score a candidate on held-out layer inputs and tied embedding rows."""
    if activation_rows.numel() == 0:
        raise ValueError("holdout activation_rows must not be empty")
    if tied_embedding_error_weight < 0:
        raise ValueError("tied_embedding_error_weight must be non-negative")

    output = output_error_score(
        weight,
        candidate,
        activation_rows,
        normalize_terms=True,
    )
    tied_embedding = None
    if tied_embedding_row_weights is not None and tied_embedding_row_weights.numel() > 0:
        tied_embedding = row_weighted_error_score(
            weight,
            candidate,
            tied_embedding_row_weights,
            normalize_terms=True,
        )
    total = output
    if tied_embedding is not None:
        total += tied_embedding_error_weight * tied_embedding
    return HoldoutCandidateScore(
        total=float(total),
        output=float(output),
        tied_embedding=None if tied_embedding is None else float(tied_embedding),
    )


def select_holdout_output_refinement(
    weight: torch.Tensor,
    baseline_candidate: torch.Tensor,
    refined_candidate: torch.Tensor,
    activation_rows: torch.Tensor,
    *,
    tied_embedding_row_weights: torch.Tensor | None = None,
    tied_embedding_error_weight: float = 1.0,
) -> HoldoutRefinementSelection:
    """Accept a refinement only when it strictly improves held-out output error."""
    baseline = score_holdout_output_candidate(
        weight,
        baseline_candidate,
        activation_rows,
        tied_embedding_row_weights=tied_embedding_row_weights,
        tied_embedding_error_weight=tied_embedding_error_weight,
    )
    refined = score_holdout_output_candidate(
        weight,
        refined_candidate,
        activation_rows,
        tied_embedding_row_weights=tied_embedding_row_weights,
        tied_embedding_error_weight=tied_embedding_error_weight,
    )
    return compare_holdout_output_scores(baseline, refined)


def compare_holdout_output_scores(
    baseline: HoldoutCandidateScore,
    refined: HoldoutCandidateScore,
) -> HoldoutRefinementSelection:
    """Choose from already-scored candidates without retaining both tensors."""
    return HoldoutRefinementSelection(
        baseline=baseline,
        refined=refined,
        accepts_refined=refined.total < baseline.total,
    )
