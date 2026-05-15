from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sketchawq.config import ModelConfig


def parse_dtype(dtype: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return table[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_causal_lm(
    config: ModelConfig,
    *,
    cache_dir: str | Path | None = None,
    device: torch.device | None = None,
):
    dtype = parse_dtype(config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        config.repo_id,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.repo_id,
        revision=config.revision,
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    if device is not None:
        model.to(device)
    model.eval()
    return tokenizer, model
