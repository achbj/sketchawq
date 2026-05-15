from __future__ import annotations

from collections.abc import Iterator

import torch


def iter_named_linears(model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            yield name, module
