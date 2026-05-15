from __future__ import annotations

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
