from __future__ import annotations

import pytest

from sketchawq.config import SketchAWQConfig
from sketchawq.reporting.metrics import build_layer_metric_rows, build_run_summary_row


def test_run_summary_includes_global_storage_estimates() -> None:
    config = SketchAWQConfig()
    report = {
        "metadata": {
            "run_id": "abc123",
            "created_at": "2026-05-02T00:00:00+00:00",
            "run_name": "storage_test",
            "environment": {"cuda_available": True, "gpu_name": "A10"},
        },
        "output_dir": "artifacts/storage_test",
        "quantization": {
            "method": "SketchAWQ-v0",
            "num_layers": 2,
            "elapsed_seconds": 1.0,
            "stats_accumulator_device": "cuda",
            "stats_accumulator_dtype": "float32",
            "timings": {
                "collect_stats_seconds": 0.4,
                "prepare_sketches_seconds": 0.1,
                "layer_quantization_seconds": 0.3,
                "layer_elapsed_sum_seconds": 0.25,
                "other_quantization_seconds": 0.2,
            },
            "cuda_peak_memory_mb": 100.0,
            "layers": [
                {"fp16_weight_mb": 10.0, "estimated_quantized_mb": 2.0},
                {"fp16_weight_mb": 30.0, "estimated_quantized_mb": 8.0},
            ],
        },
        "evaluation": {},
    }

    row = build_run_summary_row(report, config)

    assert row["fp16_weight_mb"] == 40.0
    assert row["estimated_quantized_mb"] == 10.0
    assert row["estimated_compression_ratio"] == pytest.approx(4.0)
    assert row["calibration_dataset"] == config.calibration.dataset_name
    assert row["calibration_samples"] == config.calibration.num_samples
    assert row["calibration_sequence_length"] == config.calibration.sequence_length
    assert row["evaluation_dataset"] == config.evaluation.dataset_name
    assert row["evaluation_samples"] == config.evaluation.num_samples
    assert row["evaluation_sequence_length"] == config.evaluation.sequence_length
    assert row["collect_stats_seconds"] == 0.4
    assert row["layer_quantization_seconds"] == 0.3
    assert row["stats_accumulator_device"] == "cuda"
    assert row["stats_accumulator_dtype"] == "float32"
    assert row["awq_min_scale"] == config.quantization.awq_min_scale
    assert row["awq_max_scale"] == config.quantization.awq_max_scale


def test_layer_metrics_keep_joint_search_score_separate_from_stage_scores() -> None:
    config = SketchAWQConfig()
    report = {
        "metadata": {"run_id": "abc123", "run_name": "joint_score_test"},
        "quantization": {
            "method": "SketchAWQ-v0",
            "layers": [
                {
                    "name": "proj",
                    "scale_clip_search": "joint",
                    "search_score": 0.125,
                    "scale_score": None,
                    "clip_score": None,
                }
            ],
        },
    }

    row = build_layer_metric_rows(report, config)[0]

    assert row["scale_clip_search"] == "joint"
    assert row["search_score"] == 0.125
    assert row["scale_score"] is None
    assert row["clip_score"] is None
    assert row["awq_min_scale"] == config.quantization.awq_min_scale
    assert row["awq_max_scale"] == config.quantization.awq_max_scale
