from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from sketchawq.config import SketchAWQConfig
from sketchawq.data.calibration import CalibrationBatch
from sketchawq.quant.pipeline import quantize_model


def test_pipeline_runs_on_tiny_random_gpt2() -> None:
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_inner=32,
        )
    )
    input_ids = torch.randint(0, 32, (1, 16))
    batch = CalibrationBatch(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    config = SketchAWQConfig()
    config.quantization.bits = 4
    config.quantization.group_size = 8
    config.sketch.rank = 2
    config.sketch.max_rows_per_batch = 8
    config.evaluation.enabled = False

    report = quantize_model(model, [batch], config)
    assert report.num_layers > 0
    assert report.method == "SketchAWQ-diagonal-low-rank-v0"
    assert report.timings["collect_stats_seconds"] >= 0
    assert report.timings["prepare_sketches_seconds"] >= 0
    assert report.timings["layer_quantization_seconds"] >= 0
    assert report.timings["total_seconds"] == pytest.approx(report.elapsed_seconds)
    assert report.packed_layers
    assert all(layer.packed_weight.dtype == torch.uint8 for layer in report.packed_layers)
    assert {layer.name for layer in report.packed_layers} == {layer.name for layer in report.layers}


def test_pipeline_skips_packed_layers_when_model_export_is_disabled() -> None:
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_inner=32,
        )
    )
    input_ids = torch.randint(0, 32, (1, 16))
    batch = CalibrationBatch(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    config = SketchAWQConfig()
    config.quantization.bits = 4
    config.quantization.group_size = 8
    config.sketch.rank = 2
    config.sketch.max_rows_per_batch = 8
    config.evaluation.enabled = False
    config.output.save_model = False

    report = quantize_model(model, [batch], config)

    assert report.layers
    assert report.packed_layers == []


def test_pipeline_labels_awq_only_ablation() -> None:
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_inner=32,
        )
    )
    input_ids = torch.randint(0, 32, (1, 16))
    batch = CalibrationBatch(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    config = SketchAWQConfig()
    config.quantization.bits = 4
    config.quantization.group_size = 8
    config.sketch.enabled = False
    config.sketch.mode = "none"
    config.sketch.rank = 0

    report = quantize_model(model, [batch], config)

    assert report.method == "AWQ-only-v0"
    assert report.layers
    assert {layer.sketch_mode for layer in report.layers} == {"none"}
    assert not any(layer.uses_diagonal for layer in report.layers)
    assert not any(layer.uses_low_rank for layer in report.layers)


def test_pipeline_reports_protected_outlier_channels() -> None:
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_inner=32,
        )
    )
    input_ids = torch.randint(0, 32, (1, 16))
    batch = CalibrationBatch(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    config = SketchAWQConfig()
    config.quantization.bits = 4
    config.quantization.group_size = 8
    config.sketch.rank = 2
    config.sketch.max_rows_per_batch = 8
    config.outliers.enabled = True
    config.outliers.strategy = "protect"
    config.outliers.fraction = 0.25

    report = quantize_model(model, [batch], config)

    assert "outlier-protect" in report.method
    assert report.layers
    assert all(layer.outlier_strategy == "protect" for layer in report.layers)
    assert any(layer.protected_channels > 0 for layer in report.layers)
