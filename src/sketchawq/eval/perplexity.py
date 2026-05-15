from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def evaluate_perplexity(model: torch.nn.Module, batches, *, max_batches: int) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for index, batch in enumerate(batches):
            if index >= max_batches:
                break
            kwargs = batch.as_model_kwargs() if hasattr(batch, "as_model_kwargs") else batch
            labels = kwargs["input_ids"].clone()
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)
            output = model(**kwargs)
            loss = _causal_lm_loss(output.logits, labels)
            if loss is not None:
                losses.append(float(loss.detach().cpu()))

    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    return {
        "loss": mean_loss,
        "perplexity": float(math.exp(min(mean_loss, 20.0))),
        "batches": float(len(losses)),
    }


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if not torch.any(shift_labels != -100):
        return None
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )
