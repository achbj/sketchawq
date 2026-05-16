from __future__ import annotations

import torch

from sketchawq.config import CalibrationConfig, load_config
from sketchawq.data import calibration
from sketchawq.data.calibration import TextBatchConfig, make_calibration_batches, make_text_batches


class WhitespaceTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def __call__(
        self,
        text: str,
        *,
        max_length: int | None = None,
        truncation: bool = False,
        padding: str | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int] | torch.Tensor]:
        del add_special_tokens
        input_ids = [self._token_id(token) for token in text.split()]
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]

        attention_mask = [1] * len(input_ids)
        if padding == "max_length":
            if max_length is None:
                raise ValueError("max_length is required for max_length padding")
            pad_count = max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_count
            attention_mask = attention_mask + [0] * pad_count

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.tensor([attention_mask]),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _token_id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]


def test_unpacked_calibration_keeps_legacy_padding(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "_SYNTHETIC_TEXTS", ["alpha beta"])
    tokenizer = WhitespaceTokenizer()
    config = CalibrationConfig(
        source="synthetic",
        num_samples=1,
        sequence_length=5,
        packing=False,
    )

    batches = make_calibration_batches(tokenizer, config, device=torch.device("cpu"))

    assert len(batches) == 1
    assert batches[0].input_ids.shape == (1, 5)
    torch.testing.assert_close(batches[0].attention_mask, torch.tensor([[1, 1, 0, 0, 0]]))


def test_text_batches_support_evaluation_shaped_config(monkeypatch) -> None:
    monkeypatch.setattr(
        calibration,
        "load_dataset",
        lambda *args, **kwargs: [{"text": "alpha beta"}, {"text": "gamma delta"}],
    )
    tokenizer = WhitespaceTokenizer()
    config = TextBatchConfig(
        source="dataset",
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="validation",
        text_column="text",
        num_samples=1,
        sequence_length=3,
        seed=123,
        packing=False,
    )

    batches = make_text_batches(tokenizer, config, device=torch.device("cpu"))

    assert len(batches) == 1
    torch.testing.assert_close(batches[0].input_ids, torch.tensor([[1, 2, 0]]))
    torch.testing.assert_close(batches[0].attention_mask, torch.tensor([[1, 1, 0]]))


def test_unpacked_dataset_calibration_respects_num_samples(monkeypatch) -> None:
    monkeypatch.setattr(
        calibration,
        "load_dataset",
        lambda *args, **kwargs: [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}],
    )
    tokenizer = WhitespaceTokenizer()
    config = CalibrationConfig(
        source="dataset",
        num_samples=2,
        sequence_length=3,
        packing=False,
    )

    batches = make_calibration_batches(tokenizer, config, device=torch.device("cpu"))

    assert len(batches) == 2
    torch.testing.assert_close(batches[0].input_ids, torch.tensor([[1, 0, 0]]))
    torch.testing.assert_close(batches[1].input_ids, torch.tensor([[2, 0, 0]]))


def test_packed_calibration_builds_full_chunks_without_padding(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "_SYNTHETIC_TEXTS", ["alpha beta", "gamma", "delta epsilon zeta"])
    tokenizer = WhitespaceTokenizer()
    config = CalibrationConfig(
        source="synthetic",
        num_samples=2,
        sequence_length=4,
        seed=7,
        packing=True,
    )

    batches = make_calibration_batches(tokenizer, config, device=torch.device("cpu"))

    assert len(batches) == 2
    for batch in batches:
        assert batch.input_ids.shape == (1, 4)
        torch.testing.assert_close(batch.attention_mask, torch.ones((1, 4), dtype=torch.long))

    torch.testing.assert_close(batches[0].input_ids, torch.tensor([[1, 2, 99, 3]]))
    torch.testing.assert_close(batches[1].input_ids, torch.tensor([[99, 4, 5, 6]]))


def test_config_parses_calibration_packing(tmp_path) -> None:
    config_path = tmp_path / "packed.yaml"
    config_path.write_text(
        """
calibration:
  source: synthetic
  num_samples: 2
  sequence_length: 16
  packing: true
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.calibration.packing is True


def test_config_parses_evaluation_packing(tmp_path) -> None:
    config_path = tmp_path / "packed_eval.yaml"
    config_path.write_text(
        """
evaluation:
  use_calibration_data: false
  source: dataset
  num_samples: 8
  sequence_length: 16
  max_batches: 8
  packing: true
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.evaluation.packing is True


def test_config_accepts_uncapped_evaluation(tmp_path) -> None:
    config_path = tmp_path / "uncapped_eval.yaml"
    config_path.write_text(
        """
evaluation:
  use_calibration_data: false
  source: dataset
  num_samples: null
  sequence_length: 2048
  max_batches: null
  packing: true
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.evaluation.num_samples is None
    assert config.evaluation.max_batches is None


def test_packed_text_batches_with_no_sample_cap_use_all_chunks(monkeypatch) -> None:
    monkeypatch.setattr(
        calibration,
        "_SYNTHETIC_TEXTS",
        ["alpha beta", "gamma delta", "epsilon zeta"],
    )
    tokenizer = WhitespaceTokenizer()
    config = TextBatchConfig(
        source="synthetic",
        dataset_name="",
        dataset_config="",
        split="validation",
        text_column="text",
        num_samples=None,
        sequence_length=3,
        seed=123,
        packing=True,
    )

    batches = make_text_batches(tokenizer, config, device=torch.device("cpu"))

    assert len(batches) == 3
    torch.testing.assert_close(batches[0].input_ids, torch.tensor([[1, 2, 99]]))
    torch.testing.assert_close(batches[1].input_ids, torch.tensor([[3, 4, 99]]))
    torch.testing.assert_close(batches[2].input_ids, torch.tensor([[5, 6, 99]]))
