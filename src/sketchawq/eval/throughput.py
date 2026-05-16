from __future__ import annotations

import time

import torch


def measure_forward_tokens_per_second(
    model: torch.nn.Module,
    batches,
    *,
    warmup: int = 1,
    max_batches: int | None = 8,
) -> dict[str, float]:
    model.eval()
    selected = list(batches) if max_batches is None else list(batches)[:max_batches]
    if not selected:
        return {"tokens_per_second": 0.0, "elapsed_seconds": 0.0, "tokens": 0.0}

    with torch.no_grad():
        for batch in selected[:warmup]:
            kwargs = batch.as_model_kwargs() if hasattr(batch, "as_model_kwargs") else batch
            model(**kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        tokens = 0
        for batch in selected:
            kwargs = batch.as_model_kwargs() if hasattr(batch, "as_model_kwargs") else batch
            model(**kwargs)
            tokens += int(kwargs["attention_mask"].sum().detach().cpu())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return {
        "tokens_per_second": float(tokens / max(elapsed, 1e-8)),
        "elapsed_seconds": elapsed,
        "tokens": float(tokens),
    }
