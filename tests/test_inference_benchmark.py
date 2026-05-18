from __future__ import annotations

import csv
from types import SimpleNamespace

import torch

from sketchawq.eval.inference import (
    INFERENCE_METRIC_FIELDS,
    benchmark_generation,
    build_inference_metric_row,
    make_prompt_batch,
    write_inference_metrics,
)


class TinyTokenizer:
    eos_token_id = 1

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [max(ord(char) % 11, 1) for char in text if not char.isspace()]}


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.prefill_shapes: list[tuple[int, int]] = []
        self.decode_shapes: list[tuple[int, int]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        use_cache: bool = True,
        past_key_values=None,
    ):
        del attention_mask
        if past_key_values is None:
            self.prefill_shapes.append(tuple(input_ids.shape))
        else:
            self.decode_shapes.append(tuple(input_ids.shape))
        logits = torch.zeros(
            (*input_ids.shape, self.vocab_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
        next_ids = (input_ids % self.vocab_size).unsqueeze(-1)
        logits.scatter_(-1, next_ids, 1.0)
        past = (torch.zeros(1, device=input_ids.device),) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=past)


def test_prompt_batch_repeats_seed_to_requested_shape() -> None:
    batch = make_prompt_batch(
        TinyTokenizer(),
        prompt_length=9,
        batch_size=3,
        device=torch.device("cpu"),
        seed_text="abc",
    )

    assert batch.input_ids.shape == (3, 9)
    assert batch.attention_mask.shape == (3, 9)
    torch.testing.assert_close(batch.attention_mask, torch.ones(3, 9, dtype=torch.long))
    torch.testing.assert_close(batch.input_ids[0], batch.input_ids[1])


def test_generation_benchmark_reports_prefill_decode_and_generate_rates() -> None:
    model = TinyCausalLM()

    metrics = benchmark_generation(
        model,
        TinyTokenizer(),
        prompt_length=5,
        generated_tokens=3,
        batch_size=2,
        device=torch.device("cpu"),
        warmup_runs=0,
    )

    assert metrics["prompt_length"] == 5
    assert metrics["generated_tokens"] == 3
    assert metrics["batch_size"] == 2
    assert metrics["prefill_tokens_per_second"] > 0
    assert metrics["decode_tokens_per_second"] > 0
    assert metrics["generate_tokens_per_second"] > 0
    assert metrics["time_to_first_token_ms"] >= 0
    assert model.prefill_shapes == [(2, 5)]
    assert model.decode_shapes == [(2, 1), (2, 1), (2, 1)]


def test_inference_metric_row_and_files_include_runtime_format(tmp_path) -> None:
    metadata = {
        "run_id": "abc123",
        "created_at": "2026-05-18T00:00:00+00:00",
        "run_name": "tiny_inference",
        "environment": {"cuda_available": False, "gpu_name": None},
    }
    metrics = {
        "prompt_length": 4,
        "generated_tokens": 2,
        "batch_size": 1,
        "warmup_runs": 0,
        "prefill_seconds": 0.1,
        "decode_seconds": 0.2,
        "generate_seconds": 0.3,
        "time_to_first_token_ms": 100.0,
        "prefill_tokens_per_second": 40.0,
        "decode_tokens_per_second": 10.0,
        "generate_tokens_per_second": 20.0,
        "peak_memory_mb": None,
    }

    row = build_inference_metric_row(
        metrics,
        metadata=metadata,
        model_repo_id="tiny/model",
        model_revision=None,
        method="FP16",
        model_variant="fp16",
        bits="",
        group_size="",
        inference_backend="transformers_generate_loop",
        weight_runtime_format="fp16",
        device=torch.device("cpu"),
        quantization_seconds=None,
        output_dir=tmp_path,
    )
    write_inference_metrics([row], tmp_path, append_path=tmp_path / "global.csv")

    assert row["weight_runtime_format"] == "fp16"
    assert row["inference_backend"] == "transformers_generate_loop"
    assert row["gpu_name"] is None
    assert (tmp_path / "inference_metrics.json").exists()
    with (tmp_path / "inference_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == INFERENCE_METRIC_FIELDS
    with (tmp_path / "global.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["model_repo_id"] == "tiny/model"
