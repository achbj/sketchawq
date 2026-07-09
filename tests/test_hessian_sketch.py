from __future__ import annotations

import torch

from sketchawq.config import SketchConfig
from sketchawq.quant.hessian_sketch import build_hessian_sketch
from sketchawq.quant.observer import LayerStats


def test_hessian_sketch_builds_diagonal_and_basis() -> None:
    stats = LayerStats.create("linear", in_features=8, sketch_rank=2, seed=0)
    stats.abs_sum += 1
    stats.sq_sum += 2
    stats.count = 4
    assert stats.sketch_sum is not None
    stats.sketch_sum += 1
    stats.sketch_count = 4

    sketch = build_hessian_sketch(stats, SketchConfig(enabled=True, rank=2))
    assert sketch.diagonal.shape[0] == 8
    assert sketch.basis is not None
    assert sketch.basis.shape == (8, 2)
    assert sketch.mode == "diagonal_low_rank"


def test_hessian_sketch_can_disable_curvature_weighting() -> None:
    stats = LayerStats.create("linear", in_features=8, sketch_rank=0, seed=0)
    stats.sq_sum += 2
    stats.count = 4

    sketch = build_hessian_sketch(stats, SketchConfig(enabled=False, mode="none", rank=0))

    assert sketch.diagonal is None
    assert sketch.basis is None
    assert sketch.rank == 0
    assert sketch.mode == "none"


def test_hessian_sketch_diagonal_mode_does_not_build_low_rank_basis() -> None:
    stats = LayerStats.create("linear", in_features=8, sketch_rank=0, seed=0)
    stats.sq_sum += 2
    stats.count = 4

    sketch = build_hessian_sketch(stats, SketchConfig(enabled=True, mode="diagonal", rank=4))

    assert sketch.diagonal is not None
    assert sketch.diagonal.shape[0] == 8
    assert sketch.basis is None
    assert sketch.rank == 0
    assert sketch.mode == "diagonal"


def test_nystrom_sketch_preserves_low_rank_spectrum_and_input_diagonal() -> None:
    stats = LayerStats.create("linear", in_features=2, sketch_rank=2, seed=0)
    hessian = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
    assert stats.sketch_omega is not None
    assert stats.sketch_sum is not None
    stats.sq_sum.copy_(hessian.diag() * 10)
    stats.count = 10
    stats.sketch_sum.copy_(hessian @ stats.sketch_omega * 10)
    stats.sketch_count = 10

    sketch = build_hessian_sketch(
        stats,
        SketchConfig(
            enabled=True,
            rank=2,
            low_rank_representation="nystrom",
            nystrom_eigenvalue_floor=1e-8,
            diagonal_floor=1e-8,
        ),
    )

    assert sketch.basis is None
    assert sketch.factor is not None
    assert sketch.rank == 2
    assert sketch.low_rank_representation == "nystrom"
    approximation = sketch.factor @ sketch.factor.transpose(0, 1)
    approximation = approximation + torch.diag(
        sketch.diagonal - sketch.factor.square().sum(dim=1)
    )
    torch.testing.assert_close(approximation, hessian / hessian.diag().mean(), atol=1e-5, rtol=1e-5)
