from __future__ import annotations

import torch

from sketchawq.quant.refinement_guard import select_holdout_output_refinement


def test_holdout_guard_accepts_a_strict_output_improvement() -> None:
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    activation_rows = torch.eye(2)
    baseline = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    refined = weight.clone()

    selection = select_holdout_output_refinement(
        weight,
        baseline,
        refined,
        activation_rows,
    )

    assert selection.accepts_refined is True
    assert selection.refined.total < selection.baseline.total
    assert selection.refined.tied_embedding is None


def test_holdout_guard_uses_tied_embedding_frequency_as_a_tie_breaker() -> None:
    weight = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    activation_rows = torch.tensor([[1.0, 0.0]])
    baseline = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    refined = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

    selection = select_holdout_output_refinement(
        weight,
        baseline,
        refined,
        activation_rows,
        tied_embedding_row_weights=torch.tensor([10.0, 0.0]),
        tied_embedding_error_weight=1.0,
    )

    assert selection.baseline.output == selection.refined.output
    assert selection.baseline.tied_embedding is not None
    assert selection.refined.tied_embedding is not None
    assert selection.accepts_refined is True
