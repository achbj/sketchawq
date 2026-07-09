from __future__ import annotations

import pytest
import torch

from sketchawq.config import QuantizationConfig, SketchConfig
from sketchawq.quant import awq_scale_search
from sketchawq.quant.awq_scale_search import make_awq_scales, search_awq_scale_and_clip
from sketchawq.quant.hessian_sketch import HessianSketch
from sketchawq.quant.loss import output_error_score, weighted_error_score


def test_joint_scale_clip_search_checks_all_pairs(monkeypatch) -> None:
    calls: list[tuple[float, float]] = []

    def fake_make_scales(
        weight: torch.Tensor,
        mean_abs_activation: torch.Tensor,
        *,
        alpha: float,
        normalization: str = "geometric_mean",
        min_scale: float = 0.25,
        max_scale: float = 4.0,
    ) -> torch.Tensor:
        del weight, mean_abs_activation, normalization, min_scale, max_scale
        return torch.full((2,), float(alpha))

    def fake_quantize_weight(
        weight: torch.Tensor,
        *,
        bits: int,
        group_size: int,
        symmetric: bool,
        clip_ratio: float,
        input_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del bits, group_size, symmetric
        assert input_scale is not None
        alpha = float(input_scale[0])
        calls.append((alpha, float(clip_ratio)))
        return torch.tensor([[alpha, float(clip_ratio)]], dtype=weight.dtype)

    def fake_score(
        original: torch.Tensor,
        candidate: torch.Tensor,
        *,
        diagonal: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        low_rank_weight: float = 0.15,
        normalize_terms: bool = True,
    ) -> float:
        del original, diagonal, basis, low_rank_weight, normalize_terms
        alpha = float(candidate[0, 0])
        clip = float(candidate[0, 1])
        return abs(alpha - 0.25) + abs(clip - 0.9)

    monkeypatch.setattr(awq_scale_search, "make_awq_scales", fake_make_scales)
    monkeypatch.setattr(awq_scale_search, "fake_quantize_weight", fake_quantize_weight)
    monkeypatch.setattr(awq_scale_search, "weighted_error_score", fake_score)

    result = search_awq_scale_and_clip(
        torch.ones(1, 2),
        torch.ones(2),
        HessianSketch(diagonal=None, basis=None, rank=0, mode="none"),
        QuantizationConfig(awq_alpha_candidates=[0.0, 0.25], clip_candidates=[1.0, 0.9]),
        SketchConfig(enabled=False, mode="none", rank=0),
    )

    assert result.alpha == 0.25
    assert result.clip_ratio == 0.9
    assert result.score < 1e-6
    assert result.scale_score is None
    assert result.clip_score is None
    assert calls == [(0.0, 1.0), (0.0, 0.9), (0.25, 1.0), (0.25, 0.9)]


def test_weighted_error_score_normalizes_curvature_terms() -> None:
    original = torch.tensor([[1.0, 2.0, 3.0], [1.5, -2.0, 0.5]])
    candidate = original * 0.9
    diagonal = torch.tensor([0.5, 1.0, 2.0])
    basis = torch.linalg.qr(torch.tensor([[1.0], [1.0], [0.5]])).Q

    score = weighted_error_score(original, candidate, diagonal=diagonal, basis=basis, low_rank_weight=0.25)
    scaled_score = weighted_error_score(
        original * 10.0,
        candidate * 10.0,
        diagonal=diagonal,
        basis=basis,
        low_rank_weight=0.25,
    )

    assert score > 0.0
    torch.testing.assert_close(torch.tensor(scaled_score), torch.tensor(score))


def test_spectrum_factor_uses_its_observed_curvature_without_manual_weight() -> None:
    original = torch.zeros(1, 2)
    factor = torch.tensor([[3.0], [0.0]])

    first_channel = weighted_error_score(
        original,
        torch.tensor([[1.0, 0.0]]),
        diagonal=torch.zeros(2),
        factor=factor,
        low_rank_weight=0.0,
        normalize_terms=False,
    )
    second_channel = weighted_error_score(
        original,
        torch.tensor([[0.0, 1.0]]),
        diagonal=torch.zeros(2),
        factor=factor,
        low_rank_weight=100.0,
        normalize_terms=False,
    )

    assert first_channel == 9.0
    assert second_channel == 0.0


def test_output_error_score_measures_calibration_output_reconstruction() -> None:
    original = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    candidate = torch.tensor([[0.5, 0.0], [0.0, 2.0]])
    activation_rows = torch.tensor([[2.0, 0.0], [0.0, 1.0]])

    score = output_error_score(original, candidate, activation_rows)
    scaled_score = output_error_score(original * 10.0, candidate * 10.0, activation_rows)

    assert score > 0.0
    torch.testing.assert_close(torch.tensor(scaled_score), torch.tensor(score))


def test_output_aware_search_uses_activation_rows(monkeypatch) -> None:
    def fake_make_scales(
        weight: torch.Tensor,
        mean_abs_activation: torch.Tensor,
        *,
        alpha: float,
        normalization: str = "geometric_mean",
        min_scale: float = 0.25,
        max_scale: float = 4.0,
    ) -> torch.Tensor:
        del weight, mean_abs_activation, normalization, min_scale, max_scale
        return torch.full((2,), float(alpha))

    def fake_quantize_weight(
        weight: torch.Tensor,
        *,
        bits: int,
        group_size: int,
        symmetric: bool,
        clip_ratio: float,
        input_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del weight, bits, group_size, symmetric, clip_ratio
        assert input_scale is not None
        alpha = float(input_scale[0])
        return torch.tensor([[alpha, 0.0]], dtype=torch.float32)

    def misleading_weight_score(*args, **kwargs) -> float:
        del args, kwargs
        return 0.0

    monkeypatch.setattr(awq_scale_search, "make_awq_scales", fake_make_scales)
    monkeypatch.setattr(awq_scale_search, "fake_quantize_weight", fake_quantize_weight)
    monkeypatch.setattr(awq_scale_search, "weighted_error_score", misleading_weight_score)

    config = QuantizationConfig(
        awq_alpha_candidates=[0.0, 1.0],
        clip_candidates=[1.0],
        scale_clip_search="joint",
        search_objective="output_error",
        output_error_rows=2,
    )
    result = search_awq_scale_and_clip(
        torch.tensor([[1.0, 0.0]]),
        torch.ones(2),
        HessianSketch(diagonal=None, basis=None, rank=0, mode="none"),
        config,
        SketchConfig(enabled=False, mode="none", rank=0),
        activation_rows=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
    )

    assert result.alpha == 1.0
    assert result.score == 0.0


def test_awq_scale_normalization_modes_are_explicit() -> None:
    weight = torch.tensor([[0.5, 2.0, 4.0], [1.0, 3.0, 0.25]])
    mean_abs_activation = torch.tensor([1.0, 4.0, 2.0])

    geometric = make_awq_scales(
        weight,
        mean_abs_activation,
        alpha=0.5,
        normalization="geometric_mean",
    )
    arithmetic = make_awq_scales(
        weight,
        mean_abs_activation,
        alpha=0.5,
        normalization="arithmetic_mean",
    )
    unnormalized = make_awq_scales(
        weight,
        mean_abs_activation,
        alpha=0.5,
        normalization="none",
    )

    torch.testing.assert_close(geometric.float().log().mean().exp(), torch.tensor(1.0))
    torch.testing.assert_close(arithmetic.float().mean(), torch.tensor(1.0))
    assert not torch.equal(geometric, arithmetic)
    assert not torch.equal(geometric, unnormalized)


def test_awq_scale_search_passes_configured_scale_bounds(monkeypatch) -> None:
    calls: list[tuple[float, float]] = []

    def fake_make_scales(
        weight: torch.Tensor,
        mean_abs_activation: torch.Tensor,
        *,
        alpha: float,
        normalization: str = "geometric_mean",
        min_scale: float = 0.25,
        max_scale: float = 4.0,
    ) -> torch.Tensor:
        del weight, mean_abs_activation, alpha, normalization
        calls.append((min_scale, max_scale))
        return torch.ones(2)

    def fake_quantize_weight(
        weight: torch.Tensor,
        *,
        bits: int,
        group_size: int,
        symmetric: bool,
        clip_ratio: float,
        input_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del bits, group_size, symmetric, clip_ratio, input_scale
        return weight.clone()

    monkeypatch.setattr(awq_scale_search, "make_awq_scales", fake_make_scales)
    monkeypatch.setattr(awq_scale_search, "fake_quantize_weight", fake_quantize_weight)

    config = QuantizationConfig(
        awq_alpha_candidates=[0.0, 0.5],
        clip_candidates=[1.0],
        awq_min_scale=0.125,
        awq_max_scale=8.0,
    )
    search_awq_scale_and_clip(
        torch.ones(1, 2),
        torch.ones(2),
        HessianSketch(diagonal=None, basis=None, rank=0, mode="none"),
        config,
        SketchConfig(enabled=False, mode="none", rank=0),
    )

    assert calls == [(0.125, 8.0), (0.125, 8.0)]


def test_search_uses_sketch_conditioned_scale_refinement(monkeypatch) -> None:
    calls: list[int] = []

    def fake_refined_quantize(
        weight: torch.Tensor,
        *,
        bits: int,
        group_size: int,
        symmetric: bool,
        clip_ratio: float,
        input_scale: torch.Tensor | None,
        diagonal: torch.Tensor | None,
        basis: torch.Tensor | None,
        low_rank_weight: float,
        refinement_steps: int,
    ) -> torch.Tensor:
        del (
            bits,
            group_size,
            symmetric,
            clip_ratio,
            input_scale,
            diagonal,
            basis,
            low_rank_weight,
        )
        calls.append(refinement_steps)
        return weight.clone()

    monkeypatch.setattr(
        awq_scale_search,
        "fake_quantize_weight_with_sketch_refinement",
        fake_refined_quantize,
    )

    config = QuantizationConfig(
        awq_alpha_candidates=[0.0],
        clip_candidates=[1.0],
        scale_refinement="sketch",
        scale_refinement_steps=3,
    )
    search_awq_scale_and_clip(
        torch.ones(1, 2),
        torch.ones(2),
        HessianSketch(diagonal=torch.ones(2), basis=None, rank=0, mode="diagonal"),
        config,
        SketchConfig(enabled=True, mode="diagonal", rank=0),
    )

    assert calls == [3]


def test_sketch_scale_refinement_requires_positive_steps() -> None:
    config = QuantizationConfig(
        scale_refinement="sketch",
        scale_refinement_steps=0,
    )

    with pytest.raises(ValueError, match="scale_refinement_steps"):
        config.validate()


def test_global_sketch_scale_refinement_requires_positive_steps() -> None:
    config = QuantizationConfig(
        scale_refinement="global_sketch",
        scale_refinement_steps=0,
    )

    with pytest.raises(ValueError, match="scale_refinement_steps"):
        config.validate()


def test_scale_refinement_guard_requires_a_refinement_and_output_rows() -> None:
    config = QuantizationConfig(
        scale_refinement_guard="holdout_output",
        output_error_rows=1,
    )

    with pytest.raises(ValueError, match="requires scale_refinement"):
        config.validate()

    config = QuantizationConfig(
        scale_refinement="sketch",
        scale_refinement_guard="holdout_output",
        output_error_rows=0,
    )

    with pytest.raises(ValueError, match="output_error_rows"):
        config.validate()
