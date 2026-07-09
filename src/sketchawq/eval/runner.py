from __future__ import annotations

import time
from typing import Any

import torch

from sketchawq.config import EvaluationConfig, ModelSelectionConfig
from sketchawq.eval.perplexity import evaluate_perplexity
from sketchawq.eval.throughput import measure_forward_tokens_per_second


def evaluate_model_variant(
    model: torch.nn.Module,
    batches,
    config: EvaluationConfig | ModelSelectionConfig,
    *,
    variant: str,
    method: str,
) -> dict[str, Any]:
    """Evaluate one model variant and return a normalized report block."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    result: dict[str, Any] = {
        "variant": variant,
        "method": method,
        "benchmark": config.benchmark_name,
        "perplexity": evaluate_perplexity(model, batches, max_batches=config.max_batches),
    }
    if config.measure_throughput:
        result["throughput"] = measure_forward_tokens_per_second(
            model,
            batches,
            warmup=config.throughput_warmup_batches,
            max_batches=config.max_batches,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        result["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1_000_000
    else:
        result["cuda_peak_memory_mb"] = None
    result["elapsed_seconds"] = time.perf_counter() - start
    return result
