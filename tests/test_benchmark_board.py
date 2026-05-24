from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sketchawq.experiments.plot_benchmark_board import (
    build_board_rows,
    _render_html,
    _write_csv,
    _write_matplotlib_plots,
)


def _sample_rows() -> list[dict[str, str]]:
    return [
        {
            "method": "GPTQModel-GPTQ",
            "run_name": "qwen3_4b_gptq_4bit",
            "model_repo_id": "Qwen/Qwen3-4B",
            "benchmark": "wikitext2_validation_packed_eval128",
            "bits": "4",
            "group_size": "128",
            "perplexity": "8.15",
            "original_perplexity": "7.95",
            "tokens_per_second": "920.5",
            "gpu_name": "NVIDIA A100-SXM4-40GB",
            "calibration_source": "dataset",
            "calibration_dataset": "allenai/c4",
            "calibration_samples": "128",
            "calibration_sequence_length": "2048",
            "calibration_packing": "true",
            "calibration_batch_size": "8",
            "sequence_length": "2048",
            "evaluation_dataset": "wikitext",
            "evaluation_samples": "64",
            "evaluation_sequence_length": "2048",
            "evaluation_packing": "true",
            "evaluation_batch_size": "1",
            "fp16_weight_mb": "8192",
            "estimated_quantized_mb": "2300",
            "estimated_compression_ratio": "3.56",
            "estimated_effective_bits_per_weight": "4.49",
        },
        {
            "method": "SketchAWQ-diagonal-low-rank-outlier-protect-v0",
            "run_name": "qwen3_4b_sketchawq_4bit",
            "model_repo_id": "Qwen/Qwen3-4B",
            "benchmark": "wikitext2_validation_packed_eval128",
            "bits": "4",
            "group_size": "128",
            "perplexity": "8.17",
            "original_perplexity": "7.95",
            "tokens_per_second": "880.1",
            "gpu_name": "NVIDIA A100-SXM4-40GB",
            "calibration_source": "dataset",
            "calibration_dataset": "allenai/c4",
            "calibration_samples": "128",
            "calibration_sequence_length": "2048",
            "calibration_packing": "true",
            "calibration_batch_size": "8",
            "sequence_length": "2048",
            "evaluation_dataset": "wikitext",
            "evaluation_samples": "64",
            "evaluation_sequence_length": "2048",
            "evaluation_packing": "true",
            "evaluation_batch_size": "1",
            "fp16_weight_mb": "8192",
            "estimated_quantized_mb": "2500",
            "estimated_compression_ratio": "3.28",
            "estimated_effective_bits_per_weight": "4.88",
        },
        {
            "method": "SketchAWQ-diagonal-low-rank-outlier-protect-v0",
            "run_name": "qwen3_4b_sketchawq_3bit",
            "model_repo_id": "Qwen/Qwen3-4B",
            "benchmark": "wikitext2_validation_packed_eval128",
            "bits": "3",
            "group_size": "64",
            "perplexity": "8.81",
            "original_perplexity": "7.95",
            "tokens_per_second": "845.2",
            "gpu_name": "NVIDIA A100-SXM4-40GB",
            "calibration_source": "dataset",
            "calibration_dataset": "allenai/c4",
            "calibration_samples": "128",
            "calibration_sequence_length": "2048",
            "calibration_packing": "true",
            "calibration_batch_size": "8",
            "sequence_length": "2048",
            "evaluation_dataset": "wikitext",
            "evaluation_samples": "64",
            "evaluation_sequence_length": "2048",
            "evaluation_packing": "true",
            "evaluation_batch_size": "1",
            "fp16_weight_mb": "8192",
            "estimated_quantized_mb": "1900",
            "estimated_compression_ratio": "4.31",
            "estimated_effective_bits_per_weight": "3.71",
        },
        {
            "method": "GPTQModel-GPTQ",
            "run_name": "qwen3_4b_gptq_3bit",
            "model_repo_id": "Qwen/Qwen3-4B",
            "benchmark": "wikitext2_validation_packed_eval128",
            "bits": "3",
            "group_size": "128",
            "perplexity": "9.12",
            "original_perplexity": "7.95",
            "tokens_per_second": "900.0",
            "gpu_name": "NVIDIA A100-SXM4-40GB",
            "calibration_source": "dataset",
            "calibration_dataset": "allenai/c4",
            "calibration_samples": "128",
            "calibration_sequence_length": "2048",
            "calibration_packing": "true",
            "calibration_batch_size": "8",
            "sequence_length": "2048",
            "evaluation_dataset": "wikitext",
            "evaluation_samples": "64",
            "evaluation_sequence_length": "2048",
            "evaluation_packing": "true",
            "evaluation_batch_size": "1",
            "fp16_weight_mb": "8192",
            "estimated_quantized_mb": "2100",
            "estimated_compression_ratio": "3.90",
            "estimated_effective_bits_per_weight": "4.10",
        },
    ]


def test_board_rows_add_method_labels_and_delta_from_best() -> None:
    rows = build_board_rows(
        _sample_rows(),
        model="Qwen/Qwen3-4B",
        benchmark="wikitext2_validation_packed_eval128",
    )

    four_bit_sketch = next(row for row in rows if row["bits"] == "4" and row["method_family"] == "SketchAWQ")
    three_bit_gptq = next(row for row in rows if row["bits"] == "3" and row["method_family"] == "GPTQ")

    assert four_bit_sketch["display_method"] == "SketchAWQ"
    assert four_bit_sketch["ppl_delta_from_best"] == pytest.approx(0.02)
    assert four_bit_sketch["comparison_note"].startswith("+0.020 PPL")
    assert three_bit_gptq["ppl_percent_from_best"] == pytest.approx((9.12 - 8.81) / 8.81 * 100)
    assert three_bit_gptq["best_for_bits"] == ""


def test_html_summary_explains_context_and_delta_columns() -> None:
    rows = build_board_rows(
        _sample_rows(),
        model="Qwen/Qwen3-4B",
        benchmark="wikitext2_validation_packed_eval128",
    )

    html = _render_html(rows)

    assert "Qwen/Qwen3-4B" in html
    assert "wikitext2_validation_packed_eval128" in html
    assert "3-bit winner" in html
    assert "Delta vs Best" in html
    assert "Model Size" in html
    assert "FP16 Size" in html
    assert "Eff. bpw" in html
    assert "GPU" in html
    assert "NVIDIA A100-SXM4-40GB" in html
    assert "Quantized Size" in html
    assert "Calibration Data" in html
    assert "Evaluation Data" in html
    assert "allenai/c4" in html
    assert "128 samples" in html
    assert "seq 2048" in html
    assert "batch 8" in html
    assert "dense dequantized simulation" in html


def test_board_csv_includes_derived_comparison_fields(tmp_path: Path) -> None:
    rows = build_board_rows(_sample_rows(), model=None, benchmark=None)
    output = tmp_path / "board.csv"

    _write_csv(rows, output)

    with output.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))

    assert "display_method" in header
    assert "ppl_delta_from_best" in header
    assert "comparison_note" in header
    assert "fp16_weight_mb" in header
    assert "estimated_quantized_mb" in header
    assert "estimated_compression_ratio" in header
    assert "estimated_effective_bits_per_weight" in header
    assert "gpu_name" in header
    assert "calibration_batch_size" in header


def test_benchmark_plot_writes_png_when_matplotlib_is_available(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    rows = build_board_rows(_sample_rows(), model=None, benchmark=None)

    paths = _write_matplotlib_plots(rows, tmp_path, "board")

    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.stat().st_size > 0 for path in paths)
