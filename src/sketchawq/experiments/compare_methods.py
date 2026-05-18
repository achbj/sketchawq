from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from sketchawq.reporting.metrics import RUN_SUMMARY_FIELDS


def _report_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            candidate = path / "quantization_report.json"
            if candidate.exists():
                paths.append(candidate)
            else:
                paths.extend(sorted(path.glob("*/quantization_report.json")))
        else:
            paths.append(path)
    return paths


def _summary_from_report(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary_path = path.with_name("run_summary.json")
    if summary_path.exists():
        row = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        quant = data.get("quantization", {})
        timings = quant.get("timings", {})
        ppl = data.get("evaluation", {}).get("perplexity", {})
        throughput = data.get("evaluation", {}).get("throughput", {})
        metadata = data.get("metadata", {})
        config = metadata.get("config", {})
        row = {
            "run_id": metadata.get("run_id"),
            "created_at": metadata.get("created_at"),
            "run_name": metadata.get("run_name") or config.get("output", {}).get("run_name"),
            "method": quant.get("method"),
            "model_repo_id": config.get("model", {}).get("repo_id"),
            "model_revision": config.get("model", {}).get("revision"),
            "bits": config.get("quantization", {}).get("bits"),
            "group_size": config.get("quantization", {}).get("group_size"),
            "symmetric": config.get("quantization", {}).get("symmetric"),
            "sketch_enabled": config.get("sketch", {}).get("enabled"),
            "sketch_mode": config.get("sketch", {}).get("mode"),
            "uses_diagonal": None,
            "uses_low_rank": None,
            "sketch_rank": config.get("sketch", {}).get("rank"),
            "low_rank_weight": config.get("sketch", {}).get("low_rank_weight"),
            "benchmark": config.get("evaluation", {}).get("benchmark_name"),
            "calibration_source": config.get("calibration", {}).get("source"),
            "calibration_dataset": config.get("calibration", {}).get("dataset_name"),
            "calibration_split": config.get("calibration", {}).get("split"),
            "calibration_samples": config.get("calibration", {}).get("num_samples"),
            "calibration_sequence_length": config.get("calibration", {}).get("sequence_length"),
            "calibration_packing": config.get("calibration", {}).get("packing"),
            "calibration_batch_size": config.get("method", {}).get(
                "calibration_batch_size",
                config.get("calibration", {}).get("batch_size"),
            ),
            "evaluation_source": config.get("evaluation", {}).get("source"),
            "evaluation_dataset": config.get("evaluation", {}).get("dataset_name"),
            "evaluation_split": config.get("evaluation", {}).get("split"),
            "evaluation_samples": config.get("evaluation", {}).get("num_samples"),
            "evaluation_sequence_length": config.get("evaluation", {}).get("sequence_length"),
            "evaluation_packing": config.get("evaluation", {}).get("packing"),
            "evaluation_batch_size": (
                config.get("calibration", {}).get("batch_size")
                if config.get("evaluation", {}).get("use_calibration_data")
                else config.get("evaluation", {}).get("batch_size")
            ),
            "sequence_length": config.get("calibration", {}).get("sequence_length"),
            "num_layers": quant.get("num_layers"),
            "quantization_seconds": quant.get("elapsed_seconds"),
            "collect_stats_seconds": timings.get("collect_stats_seconds"),
            "stats_batches": timings.get("stats_batches"),
            "stats_tokens": timings.get("stats_tokens"),
            "stats_forward_seconds": timings.get("stats_forward_seconds"),
            "stats_tokens_per_second": timings.get("stats_tokens_per_second"),
            "stats_use_cache_disabled": timings.get("stats_use_cache_disabled"),
            "stats_attention_mask_fast_path_batches": timings.get(
                "stats_attention_mask_fast_path_batches"
            ),
            "stats_masked_batches": timings.get("stats_masked_batches"),
            "prepare_sketches_seconds": timings.get("prepare_sketches_seconds"),
            "layer_quantization_seconds": timings.get("layer_quantization_seconds"),
            "layer_elapsed_sum_seconds": timings.get("layer_elapsed_sum_seconds"),
            "other_quantization_seconds": timings.get("other_quantization_seconds"),
            "stats_accumulator_device": quant.get("stats_accumulator_device"),
            "stats_accumulator_dtype": quant.get("stats_accumulator_dtype"),
            "cuda_peak_memory_mb": quant.get("cuda_peak_memory_mb"),
            "loss": ppl.get("loss"),
            "perplexity": ppl.get("perplexity"),
            "tokens_per_second": throughput.get("tokens_per_second"),
            "output_dir": data.get("output_dir"),
            "device": config.get("runtime", {}).get("device"),
            "cuda_available": metadata.get("environment", {}).get("cuda_available"),
            "gpu_name": metadata.get("environment", {}).get("gpu_name"),
        }
    if not row.get("run_name"):
        row["run_name"] = Path(row.get("output_dir") or path.parent).name
    row["report"] = str(path)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact comparison from run reports.")
    parser.add_argument("reports", nargs="+", help="Report files, run directories, or an artifacts root.")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument(
        "--table",
        choices=["run", "evaluation"],
        default="run",
        help="Which comparison table to collect.",
    )
    parser.add_argument("--output", default=None, help="Optional output path. Defaults to stdout.")
    args = parser.parse_args()

    if args.table == "evaluation":
        rows = _evaluation_rows(args.reports)
        fields = _union_fields(rows)
        _write_rows(rows, fields, args)
        return

    rows = [_summary_from_report(path) for path in _report_paths(args.reports)]
    _write_rows(rows, RUN_SUMMARY_FIELDS + ["report"], args)


def _evaluation_rows(inputs: list[str]) -> list[dict]:
    rows: list[dict] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            candidates = []
            direct = path / "evaluation_metrics.csv"
            if direct.exists():
                candidates.append(direct)
            candidates.extend(sorted(path.glob("*/evaluation_metrics.csv")))
        else:
            candidates = [path]
        for candidate in candidates:
            with candidate.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    row["metrics_file"] = str(candidate)
                    rows.append(row)
    return rows


def _union_fields(rows: list[dict]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _write_rows(rows: list[dict], fields: list[str], args: argparse.Namespace) -> None:
    if args.format == "json":
        payload = json.dumps(rows, indent=2)
        if args.output:
            Path(args.output).write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return

    handle = Path(args.output).open("w", encoding="utf-8", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            handle.close()


if __name__ == "__main__":
    main()
