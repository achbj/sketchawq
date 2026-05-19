from __future__ import annotations

import csv
import inspect
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


INFERENCE_METRIC_FIELDS = [
    "run_id",
    "created_at",
    "run_name",
    "model_repo_id",
    "model_revision",
    "method",
    "model_variant",
    "bits",
    "group_size",
    "inference_backend",
    "weight_runtime_format",
    "measurement_kind",
    "repeat_index",
    "repeats",
    "prompt_length",
    "prompt_tokens",
    "generated_tokens",
    "decode_tokens",
    "total_tokens",
    "batch_size",
    "warmup_runs",
    "prefill_seconds",
    "decode_seconds",
    "generate_seconds",
    "time_to_first_token_ms",
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "generate_tokens_per_second",
    "prefill_seconds_min",
    "prefill_seconds_median",
    "prefill_seconds_max",
    "decode_seconds_min",
    "decode_seconds_median",
    "decode_seconds_max",
    "generate_seconds_min",
    "generate_seconds_median",
    "generate_seconds_max",
    "time_to_first_token_ms_min",
    "time_to_first_token_ms_median",
    "time_to_first_token_ms_max",
    "prefill_tokens_per_second_min",
    "prefill_tokens_per_second_median",
    "prefill_tokens_per_second_max",
    "decode_tokens_per_second_min",
    "decode_tokens_per_second_median",
    "decode_tokens_per_second_max",
    "generate_tokens_per_second_min",
    "generate_tokens_per_second_median",
    "generate_tokens_per_second_max",
    "peak_memory_mb_min",
    "peak_memory_mb_median",
    "peak_memory_mb_max",
    "peak_memory_mb",
    "device",
    "cuda_available",
    "gpu_name",
    "quantization_seconds",
    "output_dir",
]


@dataclass(frozen=True, slots=True)
class PromptBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def make_prompt_batch(
    tokenizer: Any,
    *,
    prompt_length: int,
    batch_size: int,
    device: torch.device,
    seed_text: str = "SketchAWQ inference benchmark prompt.",
) -> PromptBatch:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    token_ids = _encode_prompt_seed(tokenizer, seed_text)
    repeats = (prompt_length // len(token_ids)) + 1
    prompt_ids = (token_ids * repeats)[:prompt_length]
    input_ids = torch.tensor([prompt_ids] * batch_size, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return PromptBatch(input_ids=input_ids, attention_mask=attention_mask)


def benchmark_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    prompt_length: int,
    generated_tokens: int,
    batch_size: int,
    device: torch.device,
    warmup_runs: int = 1,
    seed_text: str = "SketchAWQ inference benchmark prompt.",
) -> dict[str, Any]:
    return benchmark_generation_repeated(
        model,
        tokenizer,
        prompt_length=prompt_length,
        generated_tokens=generated_tokens,
        batch_size=batch_size,
        device=device,
        warmup_runs=warmup_runs,
        repeats=1,
        seed_text=seed_text,
    )[0]


def benchmark_generation_repeated(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    prompt_length: int,
    generated_tokens: int,
    batch_size: int,
    device: torch.device,
    warmup_runs: int = 1,
    repeats: int = 1,
    seed_text: str = "SketchAWQ inference benchmark prompt.",
) -> list[dict[str, Any]]:
    if generated_tokens <= 0:
        raise ValueError("generated_tokens must be positive")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    model.eval()
    prompt = make_prompt_batch(
        tokenizer,
        prompt_length=prompt_length,
        batch_size=batch_size,
        device=device,
        seed_text=seed_text,
    )

    samples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for _ in range(warmup_runs):
            _run_generation_once(model, prompt, generated_tokens=generated_tokens, device=device)

        for repeat_index in range(1, repeats + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            result = _run_generation_once(
                model,
                prompt,
                generated_tokens=generated_tokens,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1_000_000
            else:
                peak_memory_mb = None
            metrics = _build_generation_metrics(
                result,
                prompt=prompt,
                generated_tokens=generated_tokens,
                batch_size=batch_size,
                warmup_runs=warmup_runs,
                peak_memory_mb=peak_memory_mb,
            )
            metrics["measurement_kind"] = "sample"
            metrics["repeat_index"] = repeat_index
            metrics["repeats"] = repeats
            samples.append(metrics)

    return samples


def summarize_generation_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must be non-empty")

    first = samples[0]
    summary: dict[str, Any] = {
        "measurement_kind": "summary",
        "repeat_index": "",
        "repeats": len(samples),
        "prompt_length": first.get("prompt_length"),
        "prompt_tokens": first.get("prompt_tokens"),
        "generated_tokens": first.get("generated_tokens"),
        "decode_tokens": first.get("decode_tokens"),
        "total_tokens": first.get("total_tokens"),
        "batch_size": first.get("batch_size"),
        "warmup_runs": first.get("warmup_runs"),
    }
    for field in (
        "prefill_seconds",
        "decode_seconds",
        "generate_seconds",
        "time_to_first_token_ms",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "generate_tokens_per_second",
        "peak_memory_mb",
    ):
        values = _numeric_values(samples, field)
        if not values:
            summary[field] = None
            summary[f"{field}_min"] = None
            summary[f"{field}_median"] = None
            summary[f"{field}_max"] = None
            continue
        median_value = float(statistics.median(values))
        summary[field] = median_value
        summary[f"{field}_min"] = min(values)
        summary[f"{field}_median"] = median_value
        summary[f"{field}_max"] = max(values)
    return summary


def _build_generation_metrics(
    result: dict[str, float],
    *,
    prompt: PromptBatch,
    generated_tokens: int,
    batch_size: int,
    warmup_runs: int,
    peak_memory_mb: float | None,
) -> dict[str, float | int | None]:
    prompt_length = int(prompt.input_ids.shape[1])
    prefill_tokens = int(prompt.input_ids.numel())
    decode_tokens = int(batch_size * generated_tokens)
    total_tokens = prefill_tokens + decode_tokens
    prefill_seconds = result["prefill_seconds"]
    decode_seconds = result["decode_seconds"]
    generate_seconds = prefill_seconds + decode_seconds
    return {
        "prompt_length": prompt_length,
        "prompt_tokens": prefill_tokens,
        "generated_tokens": generated_tokens,
        "decode_tokens": decode_tokens,
        "total_tokens": total_tokens,
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "generate_seconds": generate_seconds,
        "time_to_first_token_ms": prefill_seconds * 1000.0,
        "prefill_tokens_per_second": _safe_rate(prefill_tokens, prefill_seconds),
        "decode_tokens_per_second": _safe_rate(decode_tokens, decode_seconds),
        "generate_tokens_per_second": _safe_rate(total_tokens, generate_seconds),
        "peak_memory_mb": peak_memory_mb,
    }


def build_inference_metric_row(
    metrics: dict[str, Any],
    *,
    metadata: dict[str, Any],
    model_repo_id: str,
    model_revision: str | None,
    method: str,
    model_variant: str,
    bits: int | str | None,
    group_size: int | str | None,
    inference_backend: str,
    weight_runtime_format: str,
    device: torch.device,
    quantization_seconds: float | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    environment = metadata.get("environment", {})
    return {
        "run_id": metadata.get("run_id"),
        "created_at": metadata.get("created_at"),
        "run_name": metadata.get("run_name"),
        "model_repo_id": model_repo_id,
        "model_revision": model_revision,
        "method": method,
        "model_variant": model_variant,
        "bits": bits,
        "group_size": group_size,
        "inference_backend": inference_backend,
        "weight_runtime_format": weight_runtime_format,
        "measurement_kind": metrics.get("measurement_kind", "sample"),
        "repeat_index": metrics.get("repeat_index", ""),
        "repeats": metrics.get("repeats", 1),
        "prompt_length": metrics.get("prompt_length"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "generated_tokens": metrics.get("generated_tokens"),
        "decode_tokens": metrics.get("decode_tokens"),
        "total_tokens": metrics.get("total_tokens"),
        "batch_size": metrics.get("batch_size"),
        "warmup_runs": metrics.get("warmup_runs"),
        "prefill_seconds": metrics.get("prefill_seconds"),
        "decode_seconds": metrics.get("decode_seconds"),
        "generate_seconds": metrics.get("generate_seconds"),
        "time_to_first_token_ms": metrics.get("time_to_first_token_ms"),
        "prefill_tokens_per_second": metrics.get("prefill_tokens_per_second"),
        "decode_tokens_per_second": metrics.get("decode_tokens_per_second"),
        "generate_tokens_per_second": metrics.get("generate_tokens_per_second"),
        "prefill_seconds_min": metrics.get("prefill_seconds_min"),
        "prefill_seconds_median": metrics.get("prefill_seconds_median"),
        "prefill_seconds_max": metrics.get("prefill_seconds_max"),
        "decode_seconds_min": metrics.get("decode_seconds_min"),
        "decode_seconds_median": metrics.get("decode_seconds_median"),
        "decode_seconds_max": metrics.get("decode_seconds_max"),
        "generate_seconds_min": metrics.get("generate_seconds_min"),
        "generate_seconds_median": metrics.get("generate_seconds_median"),
        "generate_seconds_max": metrics.get("generate_seconds_max"),
        "time_to_first_token_ms_min": metrics.get("time_to_first_token_ms_min"),
        "time_to_first_token_ms_median": metrics.get("time_to_first_token_ms_median"),
        "time_to_first_token_ms_max": metrics.get("time_to_first_token_ms_max"),
        "prefill_tokens_per_second_min": metrics.get("prefill_tokens_per_second_min"),
        "prefill_tokens_per_second_median": metrics.get("prefill_tokens_per_second_median"),
        "prefill_tokens_per_second_max": metrics.get("prefill_tokens_per_second_max"),
        "decode_tokens_per_second_min": metrics.get("decode_tokens_per_second_min"),
        "decode_tokens_per_second_median": metrics.get("decode_tokens_per_second_median"),
        "decode_tokens_per_second_max": metrics.get("decode_tokens_per_second_max"),
        "generate_tokens_per_second_min": metrics.get("generate_tokens_per_second_min"),
        "generate_tokens_per_second_median": metrics.get("generate_tokens_per_second_median"),
        "generate_tokens_per_second_max": metrics.get("generate_tokens_per_second_max"),
        "peak_memory_mb_min": metrics.get("peak_memory_mb_min"),
        "peak_memory_mb_median": metrics.get("peak_memory_mb_median"),
        "peak_memory_mb_max": metrics.get("peak_memory_mb_max"),
        "peak_memory_mb": metrics.get("peak_memory_mb"),
        "device": str(device),
        "cuda_available": environment.get("cuda_available"),
        "gpu_name": environment.get("gpu_name"),
        "quantization_seconds": quantization_seconds,
        "output_dir": str(output_dir),
    }


def write_inference_metrics(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    append_path: str | Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inference_metrics.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    _write_csv(rows, output_dir / "inference_metrics.csv", append=False)
    if append_path is not None:
        _write_csv(rows, append_path, append=True)


def _run_generation_once(
    model: torch.nn.Module,
    prompt: PromptBatch,
    *,
    generated_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    accepts_use_cache = _model_accepts_kwarg(model, "use_cache")
    accepts_past = _model_accepts_kwarg(model, "past_key_values")

    prefill_kwargs: dict[str, Any] = {
        "input_ids": prompt.input_ids,
        "attention_mask": prompt.attention_mask,
    }
    if accepts_use_cache:
        prefill_kwargs["use_cache"] = True

    prefill_start = time.perf_counter()
    output = model(**prefill_kwargs)
    _synchronize(device)
    prefill_seconds = time.perf_counter() - prefill_start

    next_token = _next_token(output)
    past_key_values = getattr(output, "past_key_values", None)
    full_input_ids = prompt.input_ids
    full_attention_mask = prompt.attention_mask
    decode_start = time.perf_counter()

    for _ in range(generated_tokens):
        if past_key_values is not None and accepts_past:
            full_attention_mask = torch.cat(
                [full_attention_mask, torch.ones_like(next_token, device=device)],
                dim=1,
            )
            kwargs = {
                "input_ids": next_token,
                "attention_mask": full_attention_mask,
                "past_key_values": past_key_values,
            }
            if accepts_use_cache:
                kwargs["use_cache"] = True
            try:
                output = model(**kwargs)
            except (RuntimeError, TypeError):
                past_key_values = None
                full_input_ids = torch.cat([full_input_ids, next_token], dim=1)
                output = model(input_ids=full_input_ids, attention_mask=full_attention_mask)
        else:
            full_input_ids = torch.cat([full_input_ids, next_token], dim=1)
            full_attention_mask = torch.ones_like(full_input_ids, device=device)
            output = model(input_ids=full_input_ids, attention_mask=full_attention_mask)
        next_token = _next_token(output)
        past_key_values = getattr(output, "past_key_values", None)

    _synchronize(device)
    decode_seconds = time.perf_counter() - decode_start
    return {"prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds}


def _next_token(output: Any) -> torch.Tensor:
    logits = output.logits if hasattr(output, "logits") else output[0]
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def _encode_prompt_seed(tokenizer: Any, seed_text: str) -> list[int]:
    try:
        encoded = tokenizer(seed_text, add_special_tokens=False)
        token_ids = encoded["input_ids"]
    except Exception:
        token_ids = []
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.squeeze(0).tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = [int(token_id) for token_id in token_ids if int(token_id) >= 0]
    if token_ids:
        return token_ids
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return [int(token_id)]
    return [0]


def _model_accepts_kwarg(model: torch.nn.Module, name: str) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _write_csv(rows: list[dict[str, Any]], path: str | Path, *, append: bool) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists() or path.stat().st_size == 0
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INFERENCE_METRIC_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _safe_rate(count: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return float(count) / float(seconds)


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        values.append(float(value))
    return values


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
