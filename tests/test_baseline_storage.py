from __future__ import annotations

import torch

from sketchawq.baselines.config import ExternalBaselineConfig
from sketchawq.baselines.runner import (
    _directory_size_mb,
    _model_parameter_storage_mb,
    _storage_metrics,
    build_baseline_run_summary_row,
)


def test_baseline_summary_carries_storage_metrics() -> None:
    config = ExternalBaselineConfig()
    report = {
        "metadata": {"run_id": "abc", "created_at": "now"},
        "quantization": {
            "method": "GPTQModel-GPTQ",
            "fp16_weight_mb": 100.0,
            "estimated_quantized_mb": 25.0,
            "estimated_compression_ratio": 4.0,
            "estimated_effective_bits_per_weight": 4.0,
        },
        "evaluation": {},
    }

    row = build_baseline_run_summary_row(report, config)

    assert row["fp16_weight_mb"] == 100.0
    assert row["estimated_quantized_mb"] == 25.0
    assert row["estimated_compression_ratio"] == 4.0
    assert row["estimated_effective_bits_per_weight"] == 4.0


def test_storage_metrics_compute_ratio_and_effective_bits() -> None:
    metrics = _storage_metrics(fp16_weight_mb=80.0, estimated_quantized_mb=20.0)

    assert metrics["fp16_weight_mb"] == 80.0
    assert metrics["estimated_quantized_mb"] == 20.0
    assert metrics["estimated_compression_ratio"] == 4.0
    assert metrics["estimated_effective_bits_per_weight"] == 4.0


def test_model_parameter_storage_counts_tied_parameters_once() -> None:
    class TiedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = torch.nn.Linear(4, 4, bias=False)
            self.right = torch.nn.Linear(4, 4, bias=False)
            self.right.weight = self.left.weight

    model = TiedModel()

    assert _model_parameter_storage_mb(model) == 16 * 4 / 1_000_000


def test_directory_size_mb_sums_files(tmp_path) -> None:
    (tmp_path / "a.bin").write_bytes(b"abc")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"de")

    assert _directory_size_mb(tmp_path) == 5 / 1_000_000
