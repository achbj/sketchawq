from __future__ import annotations

import csv
from pathlib import Path

from sketchawq.eval.inference import INFERENCE_METRIC_FIELDS
from sketchawq.experiments.summarize_inference import (
    build_inference_summary_rows,
    load_inference_rows,
    write_inference_summary,
)


def test_inference_summary_uses_run_dirs_and_compares_to_matching_fp16(tmp_path: Path) -> None:
    _write_run_metrics(
        tmp_path / "qwen3_4b_4bit_fp16_repeat3",
        [
            _metric_row(
                run_name="qwen3_4b_4bit_fp16_repeat3",
                variant="fp16",
                bits="",
                prefill=1000.0,
                decode=100.0,
                generate=200.0,
            ),
            _metric_row(
                run_name="qwen3_4b_4bit_fp16_repeat3",
                variant="fp16",
                bits="",
                repeat_index=2,
                prefill=1400.0,
                decode=140.0,
                generate=280.0,
            ),
        ],
    )
    _write_run_metrics(
        tmp_path / "qwen3_4b_4bit_sketch_dense_repeat3",
        [
            _metric_row(
                run_name="qwen3_4b_4bit_sketch_dense_repeat3",
                variant="sketchawq_dense",
                bits="4",
                prefill=500.0,
                decode=50.0,
                generate=100.0,
            ),
            _metric_row(
                run_name="qwen3_4b_4bit_sketch_dense_repeat3",
                variant="sketchawq_dense",
                bits="4",
                repeat_index=2,
                prefill=700.0,
                decode=70.0,
                generate=140.0,
            ),
            {
                **_metric_row(
                    run_name="qwen3_4b_4bit_sketch_dense_repeat3",
                    variant="sketchawq_dense",
                    bits="4",
                    prefill=600.0,
                    decode=60.0,
                    generate=120.0,
                ),
                "measurement_kind": "summary",
                "repeat_index": "",
            },
        ],
    )
    # Root aggregate CSVs can be incomplete after Modal sync; recursive loading should ignore them.
    _write_metrics_csv(tmp_path / "inference_metrics.csv", [])

    rows = load_inference_rows([str(tmp_path)])
    summary_rows = build_inference_summary_rows(rows)
    sketch = next(row for row in summary_rows if row["model_variant"] == "sketchawq_dense")

    assert len(summary_rows) == 2
    assert sketch["sample_count"] == 2
    assert sketch["bits_context"] == "4"
    assert sketch["generate_tokens_per_second_median"] == 120.0
    assert sketch["generate_vs_fp16_ratio"] == 0.5
    assert sketch["fp16_run_names"] == "qwen3_4b_4bit_fp16_repeat3"


def test_write_inference_summary_outputs_csv_json_and_html(tmp_path: Path) -> None:
    rows = [
        {
            "model_repo_id": "Qwen/Qwen3-4B",
            "bits_context": "4",
            "model_variant": "fp16",
            "prompt_length": "128",
            "batch_size": "1",
            "sample_count": 3,
            "run_count": 1,
            "run_names": "tiny",
            "generate_tokens_per_second_median": 100.0,
            "generate_vs_fp16_ratio": 1.0,
        }
    ]

    outputs = write_inference_summary(rows, tmp_path, name="tiny_board")

    assert Path(outputs["csv"]).exists()
    assert Path(outputs["json"]).exists()
    assert Path(outputs["html"]).exists()
    assert "Inference Summary Board" in Path(outputs["html"]).read_text(encoding="utf-8")


def _write_run_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True)
    (path / "inference_report.json").write_text("{}", encoding="utf-8")
    _write_metrics_csv(path / "inference_metrics.csv", rows)


def _write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INFERENCE_METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metric_row(
    *,
    run_name: str,
    variant: str,
    bits: str,
    repeat_index: int = 1,
    prefill: float,
    decode: float,
    generate: float,
) -> dict[str, object]:
    return {
        "run_id": "abc",
        "created_at": "2026-05-18T00:00:00+00:00",
        "run_name": run_name,
        "model_repo_id": "Qwen/Qwen3-4B",
        "method": "FP16" if variant == "fp16" else "SketchAWQ",
        "model_variant": variant,
        "bits": bits,
        "group_size": "128" if bits else "",
        "inference_backend": "transformers_generate_loop",
        "weight_runtime_format": "fp16" if variant == "fp16" else "sketchawq_dense_dequantized",
        "measurement_kind": "sample",
        "repeat_index": repeat_index,
        "repeats": 2,
        "prompt_length": 128,
        "prompt_tokens": 128,
        "generated_tokens": 16,
        "decode_tokens": 16,
        "total_tokens": 144,
        "batch_size": 1,
        "warmup_runs": 1,
        "prefill_tokens_per_second": prefill,
        "decode_tokens_per_second": decode,
        "generate_tokens_per_second": generate,
        "time_to_first_token_ms": 50.0,
        "peak_memory_mb": 8000.0,
        "device": "cuda",
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "output_dir": run_name,
    }
