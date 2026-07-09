from __future__ import annotations

import pytest
import torch

from sketchawq.config import SketchAWQConfig
from sketchawq.quant.error_compensation import (
    _future_projection,
    sequentially_compensate_quantized_weight,
)
from sketchawq.quant.loss import weighted_error_score
from sketchawq.quant.quantizer import quantize_weight_with_input_scale


def test_sequential_compensation_reduces_correlated_curvature_error() -> None:
    weight = torch.tensor([[0.85, 0.30, 0.85, 0.30]])
    diagonal = torch.full((4,), 0.01)
    factor = torch.ones(4, 1)
    baseline = quantize_weight_with_input_scale(
        weight,
        bits=2,
        group_size=2,
        symmetric=True,
        clip_ratio=1.0,
    )

    result = sequentially_compensate_quantized_weight(
        weight,
        baseline,
        input_scale=None,
        diagonal=diagonal,
        factor=factor,
        basis=None,
        low_rank_weight=0.0,
        refinement_steps=1,
        block_size=2,
    )

    baseline_score = weighted_error_score(
        weight,
        baseline.dequantize(),
        diagonal=diagonal,
        factor=factor,
        normalize_terms=False,
    )
    compensated_score = weighted_error_score(
        weight,
        result.quantized.dequantize(),
        diagonal=diagonal,
        factor=factor,
        normalize_terms=False,
    )

    assert compensated_score < baseline_score
    assert result.block_count == 2
    assert result.updated_block_count == 1
    assert result.quantized.bits == baseline.bits
    assert result.quantized.group_size == baseline.group_size
    assert result.quantized.qweight.shape == baseline.qweight.shape


def test_low_rank_update_matches_dense_conditional_curvature() -> None:
    diagonal = torch.tensor([0.7, 1.2, 0.9, 1.4])
    factor = torch.tensor(
        [
            [1.0, 0.2],
            [0.3, 0.8],
            [0.6, 0.1],
            [0.2, 0.9],
        ]
    )
    inverse_diagonal_factor = factor / diagonal.unsqueeze(1)
    remaining_gram = torch.eye(2) + factor.transpose(0, 1) @ inverse_diagonal_factor
    remaining_gram = remaining_gram - (
        factor[:2].transpose(0, 1) @ inverse_diagonal_factor[:2]
    )

    low_rank_update = factor[:2] @ _future_projection(
        remaining_gram,
        inverse_diagonal_factor[2:],
    )
    hessian = torch.diag(diagonal) + factor @ factor.transpose(0, 1)
    dense_update = hessian[:2, 2:] @ torch.linalg.inv(hessian[2:, 2:])

    torch.testing.assert_close(low_rank_update, dense_update, atol=1e-6, rtol=1e-6)


def test_sequential_compensation_requires_low_rank_sketch_and_local_refinement() -> None:
    config = SketchAWQConfig()
    config.quantization.error_compensation = "sequential_sketch"

    with pytest.raises(ValueError, match="scale_refinement"):
        config.validate()

    config.quantization.scale_refinement = "sketch"
    config.quantization.scale_refinement_steps = 1
    config.sketch.rank = 0

    with pytest.raises(ValueError, match="low-rank sketch"):
        config.validate()
