from __future__ import annotations

import torch

from sketchawq.config import OutlierConfig, SketchAWQConfig
from sketchawq.quant.hessian_sketch import HessianSketch
from sketchawq.quant.outliers import (
    apply_outlier_scaling,
    score_outlier_channels,
    restore_outlier_channels,
    select_outlier_channels,
)


def test_select_outlier_channels_uses_hessian_weight_score() -> None:
    weight = torch.tensor(
        [
            [1.0, 10.0, 1.0, 2.0],
            [1.0, 10.0, 1.0, 2.0],
        ]
    )
    activation = torch.tensor([4.0, 1.0, 2.0, 1.0])
    sketch = HessianSketch(
        diagonal=torch.tensor([1.0, 10.0, 1.0, 1.0]),
        basis=None,
        rank=0,
        mode="diagonal",
    )
    config = OutlierConfig(enabled=True, fraction=0.25, metric="hessian_weight")

    selection = select_outlier_channels(weight, activation, sketch, config)

    assert selection.count == 1
    assert selection.indices.tolist() == [1]
    assert selection.mask.tolist() == [False, True, False, False]


def test_select_outlier_channels_respects_min_and_max_channels() -> None:
    weight = torch.ones(2, 8)
    activation = torch.arange(1, 9, dtype=torch.float32)
    sketch = HessianSketch(diagonal=None, basis=None, rank=0, mode="none")
    config = OutlierConfig(enabled=True, fraction=1.0, min_channels=2, max_channels=3)

    selection = select_outlier_channels(weight, activation, sketch, config)

    assert selection.count == 3
    assert selection.indices.tolist() == [5, 6, 7]


def test_outlier_weight_score_matches_direct_mean_abs_for_chunked_layers() -> None:
    weight = torch.randn(17, 5, dtype=torch.float16)
    activation = torch.ones(5)
    sketch = HessianSketch(diagonal=torch.ones(5), basis=None, rank=0, mode="diagonal")

    scores = score_outlier_channels(weight, activation, sketch, "weight")

    expected = weight.float().abs().mean(dim=0)
    expected = expected / expected.mean().clamp_min(1e-8)
    torch.testing.assert_close(scores.cpu(), expected.cpu())


def test_apply_outlier_scaling_can_preserve_global_scale() -> None:
    scales = torch.ones(4)
    config = OutlierConfig(enabled=True, fraction=0.5, scale_boost=2.0, renormalize_scales=True)
    selection = select_outlier_channels(
        torch.ones(2, 4),
        torch.tensor([1.0, 10.0, 2.0, 20.0]),
        HessianSketch(diagonal=None, basis=None, rank=0, mode="none"),
        config,
    )

    adjusted = apply_outlier_scaling(scales, selection, config)

    expected = torch.tensor([1.0, 2.0, 1.0, 2.0]) / torch.sqrt(torch.tensor(2.0))
    torch.testing.assert_close(adjusted, expected)


def test_restore_outlier_channels_keeps_selected_columns_exact() -> None:
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    candidate = torch.zeros_like(original)
    selection = select_outlier_channels(
        original,
        torch.tensor([1.0, 100.0, 2.0, 50.0]),
        HessianSketch(diagonal=None, basis=None, rank=0, mode="none"),
        OutlierConfig(enabled=True, fraction=0.5),
    )

    restored = restore_outlier_channels(original, candidate, selection)

    torch.testing.assert_close(restored[:, [1, 3]], original[:, [1, 3]])
    torch.testing.assert_close(restored[:, [0, 2]], torch.zeros(3, 2))


def test_outlier_config_normalizes_user_facing_names() -> None:
    config = SketchAWQConfig()
    config.outliers.enabled = True
    config.outliers.strategy = "protected-channels"
    config.outliers.metric = "hessian-weight"
    config.outliers.fraction = 0.01

    config.validate()

    assert config.outliers.strategy == "protect"
    assert config.outliers.metric == "hessian_weight"
    assert config.outliers.active()
