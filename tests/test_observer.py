from __future__ import annotations

import logging

import pytest
import torch

from sketchawq.data.calibration import CalibrationBatch
from sketchawq.quant.observer import (
    LayerStats,
    _sample_rows,
    _select_unmasked_rows,
    collect_layer_stats,
)


class MaskedToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        return self.proj(input_ids.float())


def test_observer_ignores_padding_rows() -> None:
    model = MaskedToyModel()
    batch = CalibrationBatch(
        input_ids=torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0], [200.0, 200.0]]]),
        attention_mask=torch.tensor([[1, 1, 0, 0]]),
    )

    stats = collect_layer_stats(
        model,
        [batch],
        sketch_rank=0,
        max_rows_per_batch=8,
        seed=0,
    )["proj"]

    assert stats.count == 2
    torch.testing.assert_close(stats.abs_sum.float(), torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(stats.sq_sum.float(), torch.tensor([10.0, 20.0]))


def test_layer_stats_accumulators_use_requested_device_and_float32() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = LayerStats.create("linear", in_features=8, sketch_rank=2, seed=0, device=device)

    assert stats.abs_sum.device.type == device.type
    assert stats.sq_sum.device.type == device.type
    assert stats.abs_sum.dtype == torch.float32
    assert stats.sq_sum.dtype == torch.float32
    assert stats.sketch_omega is not None
    assert stats.sketch_sum is not None
    assert stats.sketch_omega.device.type == device.type
    assert stats.sketch_sum.device.type == device.type
    assert stats.sketch_omega.dtype == torch.float32
    assert stats.sketch_sum.dtype == torch.float32


def test_collect_layer_stats_returns_cpu_stats_after_gpu_accumulation() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedToyModel().to(device)
    batch = CalibrationBatch(
        input_ids=torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
            device=device,
        ),
        attention_mask=torch.tensor([[1, 1, 1, 1]], device=device),
    )

    stats = collect_layer_stats(
        model,
        [batch],
        sketch_rank=2,
        max_rows_per_batch=4,
        seed=0,
        activation_sample_rows=2,
    )["proj"]

    assert stats.abs_sum.device.type == "cpu"
    assert stats.sq_sum.device.type == "cpu"
    assert stats.sketch_omega is not None
    assert stats.sketch_sum is not None
    assert stats.sketch_omega.device.type == "cpu"
    assert stats.sketch_sum.device.type == "cpu"
    assert stats.activation_sample is not None
    assert stats.activation_sample.device.type == "cpu"


def test_observer_retains_activation_rows_for_output_error() -> None:
    model = MaskedToyModel()
    batch = CalibrationBatch(
        input_ids=torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0], [200.0, 200.0]]]),
        attention_mask=torch.tensor([[1, 1, 0, 0]]),
    )

    stats = collect_layer_stats(
        model,
        [batch],
        sketch_rank=0,
        max_rows_per_batch=8,
        seed=0,
        activation_sample_rows=2,
    )["proj"]

    assert stats.activation_sample is not None
    assert stats.activation_sample.shape == (2, 2)
    assert not torch.any(stats.activation_sample == 100.0)
    assert not torch.any(stats.activation_sample == 200.0)


def test_attention_mask_shape_mismatch_warns_and_uses_all_rows(caplog) -> None:
    activation = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    flat = activation.reshape(-1, 2)
    mismatched_mask = torch.ones(3)

    with caplog.at_level(logging.WARNING, logger="sketchawq.quant.observer"):
        selected = _select_unmasked_rows(
            flat,
            activation,
            mismatched_mask,
            mask_mismatch_action="warn",
        )

    torch.testing.assert_close(selected, flat)
    assert "attention_mask shape" in caplog.text
    assert "using all activation rows" in caplog.text


def test_attention_mask_shape_mismatch_can_raise() -> None:
    activation = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    flat = activation.reshape(-1, 2)
    mismatched_mask = torch.ones(3)

    with pytest.raises(ValueError, match="attention_mask shape"):
        _select_unmasked_rows(
            flat,
            activation,
            mismatched_mask,
            mask_mismatch_action="error",
        )


def test_sample_rows_seeded_random_is_reproducible() -> None:
    flat = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    first_generator = torch.Generator(device="cpu").manual_seed(123)
    second_generator = torch.Generator(device="cpu").manual_seed(123)

    first = _sample_rows(flat, 4, method="random", generator=first_generator)
    second = _sample_rows(flat, 4, method="random", generator=second_generator)
    linspace = _sample_rows(flat, 4, method="linspace")

    torch.testing.assert_close(first, second)
    assert not torch.equal(first, linspace)
