from __future__ import annotations

from types import SimpleNamespace

import torch

from sketchawq.data.calibration import CalibrationBatch
from sketchawq.eval.perplexity import evaluate_perplexity


class LogitOnlyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.labels: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask
        self.labels = None if labels is None else labels.detach().clone()
        logits = torch.full((*input_ids.shape, 6), -10.0, dtype=torch.float16)
        logits[:, 0, 2] = 10.0
        logits[:, 1, 3] = 10.0
        logits[:, 2, 5] = 10.0
        return SimpleNamespace(logits=logits, loss=torch.tensor(float("nan")))


def test_perplexity_computes_causal_loss_from_logits_instead_of_model_loss() -> None:
    model = LogitOnlyLM()
    batch = CalibrationBatch(
        input_ids=torch.tensor([[1, 2, 3, 5]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
    )

    result = evaluate_perplexity(model, [batch], max_batches=1)

    assert result["loss"] < 1e-4
    assert result["perplexity"] < 1.001
    assert model.labels is None


def test_perplexity_ignores_padding_after_causal_shift() -> None:
    model = LogitOnlyLM()
    batch = CalibrationBatch(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
    )

    result = evaluate_perplexity(model, [batch], max_batches=1)

    assert result["loss"] < 1e-4
