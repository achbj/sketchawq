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
    assert report.method == "SketchAWQ-diagonal-low-rank-tied-embedding-v1"
    assert report.timings["collect_stats_seconds"] >= 0
    assert report.timings["stats_batches"] == 1
    assert report.timings["stats_tokens"] == 16
    assert report.timings["stats_forward_seconds"] >= 0
    assert report.timings["stats_tokens_per_second"] > 0
    assert report.timings["stats_use_cache_disabled"] is True
    assert report.timings["stats_attention_mask_fast_path_batches"] == 1
    assert report.timings["prepare_sketches_seconds"] >= 0
    assert report.timings["layer_quantization_seconds"] >= 0
    assert report.timings["total_seconds"] == pytest.approx(report.elapsed_seconds)
    assert report.stats_accumulator_device == "cpu"
    assert report.stats_accumulator_dtype == "float32"
    assert report.to_dict()["stats_accumulator_device"] == "cpu"
    assert report.to_dict()["stats_accumulator_dtype"] == "float32"
    assert report.packed_layers
    assert all(layer.packed_weight.dtype == torch.uint8 for layer in report.packed_layers)
    assert {layer.name for layer in report.packed_layers} == {layer.name for layer in report.layers}
    assert report.layers[0].uses_tied_embedding_objective is True
    assert set(report.layers[0].tied_weight_names) == {"lm_head.weight", "transformer.wte.weight"}


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


def test_pipeline_reports_nystrom_curvature_representation() -> None:
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
    config.sketch.low_rank_representation = "nystrom"
    config.evaluation.enabled = False
    config.output.save_model = False

    report = quantize_model(model, [batch], config)

    assert report.method == "SketchAWQ-diagonal-nystrom-tied-embedding-v1"
    assert report.layers
    assert {layer.low_rank_representation for layer in report.layers} == {"nystrom"}
    assert all(layer.uses_spectrum_factor for layer in report.layers)


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

    assert report.method == "AWQ-only-tied-embedding-v1"
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


def test_pipeline_labels_and_reports_sketch_scale_refinement() -> None:
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
    config.quantization.scale_refinement = "sketch"
    config.quantization.scale_refinement_steps = 2
    config.sketch.rank = 2
    config.sketch.max_rows_per_batch = 8
    config.evaluation.enabled = False
    config.output.save_model = False

    report = quantize_model(model, [batch], config)

    assert "scale-refine" in report.method
    assert report.layers
    assert {layer.scale_refinement for layer in report.layers} == {"sketch"}
    assert {layer.scale_refinement_steps for layer in report.layers} == {2}


def test_pipeline_labels_and_reports_global_sketch_scale_refinement() -> None:
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
    config.quantization.scale_refinement = "global_sketch"
    config.quantization.scale_refinement_steps = 2
    config.sketch.rank = 2
    config.sketch.max_rows_per_batch = 8
    config.sketch.low_rank_representation = "nystrom"
    config.evaluation.enabled = False
    config.output.save_model = False

    report = quantize_model(model, [batch], config)

    assert "global-scale-refine" in report.method
    assert report.layers
    assert {layer.scale_refinement for layer in report.layers} == {"global_sketch"}
    assert {layer.scale_refinement_steps for layer in report.layers} == {2}


def test_pipeline_labels_and_reports_sequential_error_compensation() -> None:
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
    config.quantization.scale_refinement = "sketch"
    config.quantization.scale_refinement_steps = 2
    config.quantization.error_compensation = "sequential_sketch"
    config.sketch.rank = 2
    config.sketch.max_rows_per_batch = 8
    config.evaluation.enabled = False
    config.output.save_model = False

    report = quantize_model(model, [batch], config)

    assert "sequential-compensate" in report.method
    assert report.layers
    assert {layer.error_compensation for layer in report.layers} == {"sequential_sketch"}
    assert all(layer.error_compensation_blocks >= 1 for layer in report.layers)


def test_pipeline_applies_holdout_output_refinement_guard() -> None:
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
    selection_ids = torch.randint(0, 32, (1, 16))
    batch = CalibrationBatch(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    selection_batch = CalibrationBatch(
        input_ids=selection_ids,
        attention_mask=torch.ones_like(selection_ids),
    )
    config = SketchAWQConfig()
    config.quantization.bits = 4
    config.quantization.group_size = 8
    config.quantization.scale_refinement = "sketch"
    config.quantization.scale_refinement_steps = 2
    config.quantization.scale_refinement_guard = "holdout_output"
    config.quantization.output_error_rows = 4
    config.sketch.rank = 2
    config.sketch.max_rows_per_batch = 8
    config.evaluation.enabled = False
    config.output.save_model = False

    report = quantize_model(model, [batch], config, selection_batches=[selection_batch])

    assert "holdout-output-guard" in report.method
    assert report.timings["collect_holdout_stats_seconds"] >= 0
    assert report.layers
    assert {layer.scale_refinement_guard for layer in report.layers} == {"holdout_output"}
    assert all(layer.holdout_baseline_score is not None for layer in report.layers)
    assert all(layer.holdout_refined_score is not None for layer in report.layers)
    assert all(layer.holdout_refinement_accepted is not None for layer in report.layers)
