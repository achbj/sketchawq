from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sketchawq.config import SketchAWQConfig, load_config
from sketchawq.data.calibration import make_calibration_batches
from sketchawq.eval.inference import (
    benchmark_generation,
    build_inference_metric_row,
    write_inference_metrics,
)
from sketchawq.logging_utils import configure_logging
from sketchawq.models.load import choose_device, load_causal_lm
from sketchawq.quant.pipeline import quantize_model
from sketchawq.reporting.metrics import collect_environment_info
from sketchawq.seed import set_seed


def benchmark_from_config(
    config_path: str | Path,
    *,
    variant: str = "fp16",
    prompt_lengths: list[int] | None = None,
    batch_sizes: list[int] | None = None,
    generated_tokens: int = 128,
    warmup_runs: int = 1,
    model_cache_dir: str | Path | None = None,
    output_root: str | Path = "artifacts/inference",
    name: str | None = None,
    seed_text: str = "SketchAWQ inference benchmark prompt.",
) -> dict[str, Any]:
    config = load_config(config_path)
    configure_logging(config.runtime.log_level)
    set_seed(config.runtime.seed)
    device = choose_device(config.runtime.device)
    prompt_lengths = prompt_lengths or [128, 512, 2048]
    batch_sizes = batch_sizes or [1]
    variant = _normalize_variant(variant)

    tokenizer, model = load_causal_lm(config.model, cache_dir=model_cache_dir, device=device)
    quantization_seconds: float | None = None
    method = "FP16"
    weight_runtime_format = "fp16"
    bits: int | str | None = ""
    group_size: int | str | None = ""

    if variant == "sketchawq_dense":
        calibration_batches = make_calibration_batches(tokenizer, config.calibration, device=device)
        quant_summary = quantize_model(model, calibration_batches, config)
        quantization_seconds = quant_summary.elapsed_seconds
        method = quant_summary.method
        weight_runtime_format = "sketchawq_dense_dequantized"
        bits = config.quantization.bits
        group_size = config.quantization.group_size

    output_dir = _unique_output_dir(
        Path(output_root) / (name or _default_run_name(config, variant))
    )
    metadata = {
        "run_id": uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": output_dir.name,
        "config_path": str(config_path),
        "environment": collect_environment_info(),
    }
    rows = []
    for prompt_length in prompt_lengths:
        for batch_size in batch_sizes:
            metrics = benchmark_generation(
                model,
                tokenizer,
                prompt_length=prompt_length,
                generated_tokens=generated_tokens,
                batch_size=batch_size,
                device=device,
                warmup_runs=warmup_runs,
                seed_text=seed_text,
            )
            rows.append(
                build_inference_metric_row(
                    metrics,
                    metadata=metadata,
                    model_repo_id=config.model.repo_id,
                    model_revision=config.model.revision,
                    method=method,
                    model_variant=variant,
                    bits=bits,
                    group_size=group_size,
                    inference_backend="transformers_generate_loop",
                    weight_runtime_format=weight_runtime_format,
                    device=device,
                    quantization_seconds=quantization_seconds,
                    output_dir=output_dir,
                )
            )

    report = {
        "metadata": metadata,
        "config": _minimal_config_record(config),
        "variant": variant,
        "output_dir": str(output_dir),
        "metrics": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inference_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    write_inference_metrics(
        rows,
        output_dir,
        append_path=Path(output_root) / "inference_metrics.csv",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark generation inference throughput.")
    parser.add_argument("--config", required=True, help="SketchAWQ YAML config.")
    parser.add_argument(
        "--variant",
        choices=["fp16", "sketchawq-dense", "sketchawq_dense"],
        default="fp16",
        help="Model/runtime variant to benchmark.",
    )
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--output-root", default="artifacts/inference")
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed-text", default="SketchAWQ inference benchmark prompt.")
    args = parser.parse_args()

    report = benchmark_from_config(
        args.config,
        variant=args.variant,
        prompt_lengths=args.prompt_lengths,
        batch_sizes=args.batch_sizes,
        generated_tokens=args.generated_tokens,
        warmup_runs=args.warmup_runs,
        model_cache_dir=args.model_cache_dir,
        output_root=args.output_root,
        name=args.name,
        seed_text=args.seed_text,
    )
    print(f"Wrote {report['output_dir']}")


def _normalize_variant(variant: str) -> str:
    return variant.strip().lower().replace("-", "_")


def _default_run_name(config: SketchAWQConfig, variant: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{config.output.run_name}_{variant}_inference_{stamp}"


def _unique_output_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}_{index:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not choose a unique output directory for {path}")


def _minimal_config_record(config: SketchAWQConfig) -> dict[str, Any]:
    return {
        "model": {
            "repo_id": config.model.repo_id,
            "revision": config.model.revision,
            "dtype": config.model.dtype,
        },
        "quantization": {
            "bits": config.quantization.bits,
            "group_size": config.quantization.group_size,
        },
        "calibration": {
            "source": config.calibration.source,
            "dataset_name": config.calibration.dataset_name,
            "split": config.calibration.split,
            "num_samples": config.calibration.num_samples,
            "sequence_length": config.calibration.sequence_length,
            "packing": config.calibration.packing,
            "batch_size": config.calibration.batch_size,
        },
        "runtime": {
            "device": config.runtime.device,
            "seed": config.runtime.seed,
        },
    }


if __name__ == "__main__":
    main()
