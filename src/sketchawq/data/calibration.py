from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import cycle, islice
from random import Random
from typing import Any

import torch
from datasets import load_dataset

from sketchawq.config import CalibrationConfig


_SYNTHETIC_TEXTS = [
    "Post-training quantization compresses a trained language model without retraining it.",
    "Activation-aware quantization protects channels that matter most for the model output.",
    "A Hessian sketch gives a cheap signal about which rounding errors are more harmful.",
    "Three-bit quantization is difficult because each weight has very few representable values.",
    "The goal is to keep the method simple, fast, and compatible with standard weight kernels.",
]


@dataclass(frozen=True, slots=True)
class CalibrationBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}


@dataclass(frozen=True, slots=True)
class TextBatchConfig:
    source: str
    dataset_name: str
    dataset_config: str
    split: str
    text_column: str
    num_samples: int | None
    sequence_length: int
    seed: int
    packing: bool
    batch_size: int = 1

    @classmethod
    def from_calibration(cls, config: CalibrationConfig) -> TextBatchConfig:
        return cls(
            source=config.source,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            split=config.split,
            text_column=config.text_column,
            num_samples=config.num_samples,
            sequence_length=config.sequence_length,
            seed=config.seed,
            packing=config.packing,
            batch_size=config.batch_size,
        )

    @classmethod
    def from_evaluation(cls, config: Any, *, seed: int) -> TextBatchConfig:
        return cls(
            source=config.source,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            split=config.split,
            text_column=config.text_column,
            num_samples=config.num_samples,
            sequence_length=config.sequence_length,
            seed=seed,
            packing=config.packing,
            batch_size=config.batch_size,
        )


def _iter_texts(config: TextBatchConfig, *, repeat_synthetic: bool = False) -> Iterable[str]:
    if config.source == "synthetic":
        if config.num_samples is None:
            source = cycle(_SYNTHETIC_TEXTS) if repeat_synthetic else iter(_SYNTHETIC_TEXTS)
        else:
            repeats = (config.num_samples // len(_SYNTHETIC_TEXTS)) + 1
            source = cycle(_SYNTHETIC_TEXTS) if repeat_synthetic else iter(
                (_SYNTHETIC_TEXTS * repeats)[: config.num_samples]
            )
        for text in source:
            stripped = text.strip()
            if stripped:
                yield stripped
        return

    if config.source != "dataset":
        raise ValueError("text batch source must be either 'synthetic' or 'dataset'")

    dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.split, streaming=True)
    for row in dataset:
        text = str(row.get(config.text_column, "")).strip()
        if text:
            yield text


def make_calibration_batches(
    tokenizer,
    config: CalibrationConfig,
    *,
    device: torch.device,
) -> list[CalibrationBatch]:
    return make_text_batches(
        tokenizer,
        TextBatchConfig.from_calibration(config),
        device=device,
    )


def make_text_batches(
    tokenizer,
    config: TextBatchConfig,
    *,
    device: torch.device,
) -> list[CalibrationBatch]:
    if config.packing:
        return _make_packed_text_batches(tokenizer, config, device=device)

    texts = (
        list(_iter_texts(config))
        if config.num_samples is None
        else list(islice(_iter_texts(config), config.num_samples))
    )
    batches: list[CalibrationBatch] = []
    batch_size = _validated_batch_size(config.batch_size)

    for i in range(0, len(texts), batch_size):
        text_batch = texts[i:i + batch_size]
        encoded = tokenizer(
            text_batch[0] if len(text_batch) == 1 else text_batch,
            max_length=config.sequence_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        batches.append(
            CalibrationBatch(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
            )
        )
    return batches


def _make_packed_text_batches(
    tokenizer,
    config: TextBatchConfig,
    *,
    device: torch.device,
) -> list[CalibrationBatch]:
    selected_chunks = _select_packed_chunks(tokenizer, config)
    if config.num_samples is not None and len(selected_chunks) < config.num_samples:
        raise RuntimeError(
            "Packed calibration produced "
            f"{len(selected_chunks)} chunks, but num_samples={config.num_samples}. "
            "Use a longer calibration split or reduce sequence_length."
        )

    batches: list[CalibrationBatch] = []
    batch_size = _validated_batch_size(config.batch_size)
    for i in range(0, len(selected_chunks), batch_size):
        chunk_batch = selected_chunks[i:i + batch_size]
        input_ids = torch.tensor(chunk_batch, dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        batches.append(CalibrationBatch(input_ids=input_ids, attention_mask=attention_mask))
    return batches


def _select_packed_chunks(tokenizer, config: TextBatchConfig) -> list[list[int]]:
    if config.num_samples is None:
        return list(_iter_packed_chunks(tokenizer, config))

    if config.source == "synthetic":
        return list(
            islice(
                _iter_packed_chunks(tokenizer, config, repeat_synthetic=True),
                config.num_samples,
            )
        )

    rng = Random(config.seed)
    pool_size = config.num_samples * 4
    pool: list[list[int]] = list(islice(_iter_packed_chunks(tokenizer, config), pool_size))
    rng.shuffle(pool)
    return pool[: config.num_samples]


def _iter_packed_chunks(
    tokenizer,
    config: TextBatchConfig,
    *,
    repeat_synthetic: bool = False,
) -> Iterable[list[int]]:
    buffer: list[int] = []
    eos_token_id = _separator_token_id(tokenizer)
    for text in _iter_texts(config, repeat_synthetic=repeat_synthetic):
        token_ids = _encode_token_ids(tokenizer, text)
        if not token_ids:
            continue
        buffer.extend(token_ids)
        if eos_token_id is not None:
            buffer.append(eos_token_id)
        while len(buffer) >= config.sequence_length:
            yield buffer[: config.sequence_length]
            del buffer[: config.sequence_length]


def _encode_token_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.squeeze(0).tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _separator_token_id(tokenizer) -> int | None:
    for attr in ("eos_token_id", "sep_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return int(token_id)
    return None


def _validated_batch_size(batch_size: int) -> int:
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("text batch_size must be positive")
    return batch_size


def iter_model_kwargs(batches: Iterable[CalibrationBatch]) -> Iterable[dict[str, torch.Tensor]]:
    for batch in batches:
        yield batch.as_model_kwargs()
