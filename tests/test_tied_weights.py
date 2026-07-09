from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from sketchawq.models.layers import iter_named_linears
from sketchawq.models.tied import find_tied_linear_weights
from sketchawq.quant.loss import row_weighted_error_score


def test_tied_lm_head_is_linked_to_the_input_embedding() -> None:
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_inner=32,
        )
    )

    tied = find_tied_linear_weights(model, list(iter_named_linears(model)))

    lm_head = tied["lm_head"]
    assert lm_head.shared_with_input_embedding is True
    assert set(lm_head.weight_names) == {"lm_head.weight", "transformer.wte.weight"}


def test_row_weighted_error_prioritizes_frequent_embedding_rows() -> None:
    original = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    candidate = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    frequent_first = row_weighted_error_score(
        original,
        candidate,
        torch.tensor([100.0, 1.0]),
    )
    frequent_second = row_weighted_error_score(
        original,
        candidate,
        torch.tensor([1.0, 100.0]),
    )

    assert frequent_first > frequent_second
