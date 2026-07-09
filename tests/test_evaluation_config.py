from __future__ import annotations

from types import SimpleNamespace

import pytest
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


def test_model_selection_requires_a_disjoint_calibration_holdout() -> None:
    config = SketchAWQConfig()
    config.model_selection.enabled = True

    with pytest.raises(ValueError, match="model_selection"):
        config.validate()


def test_model_selection_uses_holdout_and_skips_final_evaluation_loading(monkeypatch, tmp_path) -> None:
    config = SketchAWQConfig()
    config.calibration.holdout_samples = 1
    config.calibration.holdout_split = "validation"
    config.model_selection.enabled = True
    config.evaluation.enabled = False
    config.output.run_name = "selection_test"
    calibration_batches = [object()]
    holdout_batches = [object()]
    calls: list[tuple[object, str]] = []
    saved: dict[str, object] = {}

    monkeypatch.setattr(run_quantization, "load_config", lambda _: config)
    monkeypatch.setattr(
        run_quantization,
        "load_causal_lm",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        run_quantization,
        "make_calibration_and_holdout_batches",
        lambda *args, **kwargs: (calibration_batches, holdout_batches),
    )
    monkeypatch.setattr(
        run_quantization,
        "quantize_model",
        lambda *args, **kwargs: SimpleNamespace(
            method="SketchAWQ-test",
            packed_layers=(),
            to_dict=lambda: {"method": "SketchAWQ-test"},
        ),
    )
    monkeypatch.setattr(
        run_quantization,
        "_make_evaluation_batches",
        lambda *args, **kwargs: pytest.fail("final evaluation batches must not be loaded"),
    )

    def fake_evaluate(model, batches, selection_config, *, variant, method):
        calls.append((batches, variant))
        return {
            "variant": variant,
            "method": method,
            "benchmark": selection_config.benchmark_name,
            "perplexity": {"loss": 1.0, "perplexity": 2.0, "batches": 1.0},
        }

    monkeypatch.setattr(run_quantization, "evaluate_model_variant", fake_evaluate)
    monkeypatch.setattr(
        run_quantization,
        "save_quantized_artifacts",
        lambda **kwargs: saved.update(kwargs),
    )

    report = run_quantization.run_from_config("selection.yaml", output_root=tmp_path)

    assert [variant for _, variant in calls] == ["original_fp16", "quantized_sketchawq"]
    assert all(batches is holdout_batches for batches, _ in calls)
    assert "evaluation" not in report
    assert set(report["model_selection"]) == {"original", "quantized"}
    assert saved["report"] is report
