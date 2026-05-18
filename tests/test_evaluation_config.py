from __future__ import annotations

import torch

from sketchawq.config import SketchAWQConfig
from sketchawq.data.calibration import TextBatchConfig
from sketchawq.experiments import run_quantization


def test_sketchawq_evaluation_batches_can_use_packed_validation(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_make_text_batches(tokenizer, batch_config, *, device):
        captured["tokenizer"] = tokenizer
        captured["batch_config"] = batch_config
        captured["device"] = device
        return []

    monkeypatch.setattr(
        run_quantization,
        "make_text_batches",
        fake_make_text_batches,
    )
    config = SketchAWQConfig()
    config.evaluation.use_calibration_data = False
    config.evaluation.source = "dataset"
    config.evaluation.split = "validation"
    config.evaluation.num_samples = 128
    config.evaluation.sequence_length = 1024
    config.evaluation.packing = True
    config.evaluation.batch_size = 4

    run_quantization._make_evaluation_batches(
        object(),
        config,
        calibration_batches=[],
        device=torch.device("cpu"),
    )

    batch_config = captured["batch_config"]
    assert isinstance(batch_config, TextBatchConfig)
    assert batch_config.source == "dataset"
    assert batch_config.split == "validation"
    assert batch_config.num_samples == 128
    assert batch_config.sequence_length == 1024
    assert batch_config.packing is True
    assert batch_config.batch_size == 4


def test_sketchawq_evaluation_batches_can_request_full_split(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_make_text_batches(tokenizer, batch_config, *, device):
        captured["batch_config"] = batch_config
        return []

    monkeypatch.setattr(
        run_quantization,
        "make_text_batches",
        fake_make_text_batches,
    )
    config = SketchAWQConfig()
    config.evaluation.use_calibration_data = False
    config.evaluation.num_samples = None
    config.evaluation.max_batches = None
    config.evaluation.sequence_length = 2048
    config.evaluation.packing = True

    run_quantization._make_evaluation_batches(
        object(),
        config,
        calibration_batches=[],
        device=torch.device("cpu"),
    )

    batch_config = captured["batch_config"]
    assert isinstance(batch_config, TextBatchConfig)
    assert batch_config.num_samples is None
    assert batch_config.sequence_length == 2048
    assert batch_config.packing is True
