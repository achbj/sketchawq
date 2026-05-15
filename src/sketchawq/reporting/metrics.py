from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from sketchawq.config import SketchAWQConfig, config_to_dict


RUN_SUMMARY_FIELDS = [
    "run_id",
    "created_at",
    "run_name",
    "method",
    "model_repo_id",
    "model_revision",
    "bits",
    "group_size",
    "symmetric",
    "scale_clip_search",
    "search_objective",
    "awq_min_scale",
    "awq_max_scale",
    "output_error_rows",
    "output_error_weight",
    "hessian_error_weight",
    "outlier_enabled",
    "outlier_strategy",
    "outlier_metric",
    "outlier_fraction",
    "outlier_scale_boost",
    "outlier_min_channels",
    "outlier_max_channels",
    "outlier_protected_bits",
    "adaptive_enabled",
    "adaptive_sensitivity_metric",
    "adaptive_sensitivity_top_fraction",
    "adaptive_protected_fraction_min",
    "adaptive_protected_fraction_max",
    "adaptive_sensitive_group_size",
    "adaptive_group_size_top_fraction",
    "sketch_enabled",
    "sketch_mode",
    "uses_diagonal",
    "uses_low_rank",
    "sketch_rank",
    "low_rank_weight",
    "normalize_error_terms",
    "benchmark",
    "calibration_source",
    "calibration_dataset",
    "calibration_samples",
    "calibration_packing",
    "sequence_length",
    "num_layers",
    "quantization_seconds",
    "collect_stats_seconds",
    "prepare_sketches_seconds",
    "layer_quantization_seconds",
    "layer_elapsed_sum_seconds",
    "other_quantization_seconds",
    "cuda_peak_memory_mb",
    "fp16_weight_mb",
    "estimated_quantized_mb",
    "estimated_compression_ratio",
    "loss",
    "perplexity",
    "tokens_per_second",
    "original_loss",
    "original_perplexity",
    "original_tokens_per_second",
    "quantized_loss",
    "quantized_perplexity",
    "quantized_tokens_per_second",
    "perplexity_delta",
    "perplexity_ratio",
    "throughput_ratio",
    "output_dir",
    "device",
    "cuda_available",
    "gpu_name",
]

EVALUATION_METRIC_FIELDS = [
    "run_id",
    "created_at",
    "run_name",
    "model_repo_id",
    "method",
    "model_variant",
    "benchmark",
    "bits",
    "group_size",
    "loss",
    "perplexity",
    "tokens_per_second",
    "eval_elapsed_seconds",
    "eval_batches",
    "tokens",
    "cuda_peak_memory_mb",
    "output_dir",
]

LAYER_METRIC_FIELDS = [
    "run_id",
    "run_name",
    "method",
    "model_repo_id",
    "bits",
    "group_size",
    "adaptive_sensitivity_score",
    "adaptive_group_size_applied",
    "layer_name",
    "sketch_mode",
    "uses_diagonal",
    "uses_low_rank",
    "input_channels",
    "output_channels",
    "scale_clip_search",
    "awq_alpha",
    "clip_ratio",
    "awq_scale_normalization",
    "awq_min_scale",
    "awq_max_scale",
    "search_score",
    "scale_score",
    "clip_score",
    "sketch_rank",
    "weight_numel",
    "layer_elapsed_seconds",
    "fp16_weight_mb",
    "estimated_quantized_mb",
    "estimated_compression_ratio",
    "outlier_strategy",
    "outlier_metric",
    "outlier_fraction",
    "outlier_scale_boost",
    "outlier_channels",
    "protected_channels",
]


def make_run_metadata(config: SketchAWQConfig, *, config_path: str | Path) -> dict[str, Any]:
    return {
        "run_id": uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "run_name": config.output.run_name,
        "config": config_to_dict(config),
        "environment": collect_environment_info(),
    }


def collect_environment_info() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
    }


def build_run_summary_row(report: dict[str, Any], config: SketchAWQConfig) -> dict[str, Any]:
    quant = report.get("quantization", {})
    timings = quant.get("timings", {})
    layers = quant.get("layers", [])
    fp16_weight_mb = _sum_layer_float(layers, "fp16_weight_mb")
    estimated_quantized_mb = _sum_layer_float(layers, "estimated_quantized_mb")
    evaluation = report.get("evaluation", {})
    original = _variant_evaluation(evaluation, "original")
    quantized = _variant_evaluation(evaluation, "quantized")
    quantized_ppl = quantized.get("perplexity", {})
    quantized_throughput = quantized.get("throughput", {})
    original_ppl = original.get("perplexity", {})
    original_throughput = original.get("throughput", {})
    original_perplexity = original_ppl.get("perplexity")
    quantized_perplexity = quantized_ppl.get("perplexity")
    original_tps = original_throughput.get("tokens_per_second")
    quantized_tps = quantized_throughput.get("tokens_per_second")
    metadata = report.get("metadata", {})
    environment = metadata.get("environment", {})
    return {
        "run_id": metadata.get("run_id"),
        "created_at": metadata.get("created_at"),
        "run_name": metadata.get("run_name", config.output.run_name),
        "method": quant.get("method"),
        "model_repo_id": config.model.repo_id,
        "model_revision": config.model.revision,
        "bits": config.quantization.bits,
        "group_size": config.quantization.group_size,
        "symmetric": config.quantization.symmetric,
        "scale_clip_search": config.quantization.scale_clip_search,
        "search_objective": config.quantization.search_objective,
        "awq_min_scale": config.quantization.awq_min_scale,
        "awq_max_scale": config.quantization.awq_max_scale,
        "output_error_rows": config.quantization.output_error_rows,
        "output_error_weight": config.quantization.output_error_weight,
        "hessian_error_weight": config.quantization.hessian_error_weight,
        "outlier_enabled": config.outliers.enabled,
        "outlier_strategy": config.outliers.strategy if config.outliers.active() else "none",
        "outlier_metric": config.outliers.metric,
        "outlier_fraction": config.outliers.fraction,
        "outlier_scale_boost": config.outliers.scale_boost,
        "outlier_min_channels": config.outliers.min_channels,
        "outlier_max_channels": config.outliers.max_channels,
        "outlier_protected_bits": config.outliers.protected_bits,
        "adaptive_enabled": config.adaptive.enabled,
        "adaptive_sensitivity_metric": config.adaptive.sensitivity_metric,
        "adaptive_sensitivity_top_fraction": config.adaptive.sensitivity_top_fraction,
        "adaptive_protected_fraction_min": config.adaptive.protected_fraction_min,
        "adaptive_protected_fraction_max": config.adaptive.protected_fraction_max,
        "adaptive_sensitive_group_size": config.adaptive.sensitive_group_size,
        "adaptive_group_size_top_fraction": config.adaptive.group_size_top_fraction,
        "sketch_enabled": config.sketch.enabled,
        "sketch_mode": config.sketch.effective_mode(),
        "uses_diagonal": config.sketch.uses_diagonal(),
        "uses_low_rank": config.sketch.uses_low_rank(),
        "sketch_rank": config.sketch.effective_rank(),
        "low_rank_weight": config.sketch.low_rank_weight,
        "normalize_error_terms": config.sketch.normalize_error_terms,
        "benchmark": config.evaluation.benchmark_name,
        "calibration_source": config.calibration.source,
        "calibration_dataset": config.calibration.dataset_name,
        "calibration_samples": config.calibration.num_samples,
        "calibration_packing": config.calibration.packing,
        "sequence_length": config.calibration.sequence_length,
        "num_layers": quant.get("num_layers"),
        "quantization_seconds": quant.get("elapsed_seconds"),
        "collect_stats_seconds": timings.get("collect_stats_seconds"),
        "prepare_sketches_seconds": timings.get("prepare_sketches_seconds"),
        "layer_quantization_seconds": timings.get("layer_quantization_seconds"),
        "layer_elapsed_sum_seconds": timings.get("layer_elapsed_sum_seconds"),
        "other_quantization_seconds": timings.get("other_quantization_seconds"),
        "cuda_peak_memory_mb": quant.get("cuda_peak_memory_mb"),
        "fp16_weight_mb": fp16_weight_mb,
        "estimated_quantized_mb": estimated_quantized_mb,
        "estimated_compression_ratio": _optional_ratio(fp16_weight_mb, estimated_quantized_mb),
        "loss": quantized_ppl.get("loss"),
        "perplexity": quantized_perplexity,
        "tokens_per_second": quantized_tps,
        "original_loss": original_ppl.get("loss"),
        "original_perplexity": original_perplexity,
        "original_tokens_per_second": original_tps,
        "quantized_loss": quantized_ppl.get("loss"),
        "quantized_perplexity": quantized_perplexity,
        "quantized_tokens_per_second": quantized_tps,
        "perplexity_delta": _optional_delta(quantized_perplexity, original_perplexity),
        "perplexity_ratio": _optional_ratio(quantized_perplexity, original_perplexity),
        "throughput_ratio": _optional_ratio(quantized_tps, original_tps),
        "output_dir": report.get("output_dir"),
        "device": config.runtime.device,
        "cuda_available": environment.get("cuda_available"),
        "gpu_name": environment.get("gpu_name"),
    }


def build_evaluation_metric_rows(
    report: dict[str, Any],
    config: SketchAWQConfig,
) -> list[dict[str, Any]]:
    metadata = report.get("metadata", {})
    rows: list[dict[str, Any]] = []
    for key in ("original", "quantized"):
        item = _variant_evaluation(report.get("evaluation", {}), key)
        if not item:
            continue
        perplexity = item.get("perplexity", {})
        throughput = item.get("throughput", {})
        rows.append(
            {
                "run_id": metadata.get("run_id"),
                "created_at": metadata.get("created_at"),
                "run_name": metadata.get("run_name", config.output.run_name),
                "model_repo_id": config.model.repo_id,
                "method": item.get("method"),
                "model_variant": item.get("variant", key),
                "benchmark": item.get("benchmark", config.evaluation.benchmark_name),
                "bits": "" if key == "original" else config.quantization.bits,
                "group_size": "" if key == "original" else config.quantization.group_size,
                "loss": perplexity.get("loss"),
                "perplexity": perplexity.get("perplexity"),
                "tokens_per_second": throughput.get("tokens_per_second"),
                "eval_elapsed_seconds": item.get("elapsed_seconds"),
                "eval_batches": perplexity.get("batches"),
                "tokens": throughput.get("tokens"),
                "cuda_peak_memory_mb": item.get("cuda_peak_memory_mb"),
                "output_dir": report.get("output_dir"),
            }
        )
    return rows


def build_layer_metric_rows(
    report: dict[str, Any],
    config: SketchAWQConfig,
) -> list[dict[str, Any]]:
    quant = report.get("quantization", {})
    run_id = report.get("metadata", {}).get("run_id")
    run_name = report.get("metadata", {}).get("run_name", config.output.run_name)
    rows = []
    for layer in quant.get("layers", []):
        rows.append(
            {
                "run_id": run_id,
                "run_name": run_name,
                "method": quant.get("method"),
                "model_repo_id": config.model.repo_id,
                "bits": config.quantization.bits,
                "group_size": layer.get("group_size", config.quantization.group_size),
                "adaptive_sensitivity_score": layer.get("adaptive_sensitivity_score"),
                "adaptive_group_size_applied": layer.get("adaptive_group_size_applied", False),
                "layer_name": layer.get("name"),
                "sketch_mode": layer.get("sketch_mode", config.sketch.effective_mode()),
                "uses_diagonal": layer.get("uses_diagonal", config.sketch.uses_diagonal()),
                "uses_low_rank": layer.get("uses_low_rank", config.sketch.uses_low_rank()),
                "input_channels": layer.get("input_channels"),
                "output_channels": layer.get("output_channels"),
                "scale_clip_search": layer.get(
                    "scale_clip_search",
                    config.quantization.scale_clip_search,
                ),
                "awq_alpha": layer.get("awq_alpha"),
                "clip_ratio": layer.get("clip_ratio"),
                "awq_scale_normalization": layer.get(
                    "awq_scale_normalization",
                    config.quantization.awq_scale_normalization,
                ),
                "awq_min_scale": layer.get("awq_min_scale", config.quantization.awq_min_scale),
                "awq_max_scale": layer.get("awq_max_scale", config.quantization.awq_max_scale),
                "search_score": layer.get("search_score"),
                "scale_score": layer.get("scale_score"),
                "clip_score": layer.get("clip_score"),
                "sketch_rank": layer.get("sketch_rank"),
                "weight_numel": layer.get("weight_numel"),
                "layer_elapsed_seconds": layer.get("layer_elapsed_seconds"),
                "fp16_weight_mb": layer.get("fp16_weight_mb"),
                "estimated_quantized_mb": layer.get("estimated_quantized_mb"),
                "estimated_compression_ratio": layer.get("estimated_compression_ratio"),
                "outlier_strategy": layer.get("outlier_strategy", "none"),
                "outlier_metric": layer.get("outlier_metric", config.outliers.metric),
                "outlier_fraction": layer.get("outlier_fraction", config.outliers.fraction),
                "outlier_scale_boost": layer.get(
                    "outlier_scale_boost",
                    config.outliers.scale_boost,
                ),
                "outlier_channels": layer.get("outlier_channels", 0),
                "protected_channels": layer.get("protected_channels", 0),
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(rows: list[dict[str, Any]], path: str | Path, *, fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plot_artifacts(
    report: dict[str, Any],
    config: SketchAWQConfig,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    summary_row = build_run_summary_row(report, config)
    evaluation_rows = build_evaluation_metric_rows(report, config)
    layer_rows = build_layer_metric_rows(report, config)

    (output_dir / "run_summary.json").write_text(
        json.dumps(summary_row, indent=2),
        encoding="utf-8",
    )
    write_csv([summary_row], output_dir / "run_summary.csv", fields=RUN_SUMMARY_FIELDS)
    write_csv(
        evaluation_rows,
        output_dir / "evaluation_metrics.csv",
        fields=EVALUATION_METRIC_FIELDS,
    )
    write_jsonl(evaluation_rows, output_dir / "evaluation_metrics.jsonl")
    write_csv(layer_rows, output_dir / "layer_metrics.csv", fields=LAYER_METRIC_FIELDS)
    write_jsonl(layer_rows, output_dir / "layer_metrics.jsonl")


def _variant_evaluation(evaluation: dict[str, Any], key: str) -> dict[str, Any]:
    if key in evaluation:
        return evaluation[key] or {}
    if key == "quantized" and "perplexity" in evaluation:
        return {
            "variant": "quantized_sketchawq",
            "method": "SketchAWQ-v0",
            "benchmark": "legacy",
            "perplexity": evaluation.get("perplexity", {}),
            "throughput": evaluation.get("throughput", {}),
        }
    return {}


def _optional_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _optional_ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in {None, 0}:
        return None
    return float(value) / float(baseline)


def _sum_layer_float(layers: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for layer in layers:
        value = layer.get(key)
        if value in {None, ""}:
            continue
        values.append(float(value))
    if not values:
        return None
    return float(sum(values))
