from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TiedLinearWeight:
    """Alias information for a quantized Linear weight."""

    linear_name: str
    weight_names: tuple[str, ...]
    shared_with_input_embedding: bool


def find_tied_linear_weights(
    model: torch.nn.Module,
    named_linears: Sequence[tuple[str, torch.nn.Linear]],
) -> dict[str, TiedLinearWeight]:
    """Find state-dict aliases for Linear weights, including tied token embeddings."""
    state_dict = model.state_dict()
    aliases_by_storage = _aliases_by_storage(state_dict)
    input_embedding_keys = _input_embedding_storage_keys(model)
    tied: dict[str, TiedLinearWeight] = {}

    for name, module in named_linears:
        weight_name = f"{name}.weight"
        tensor = state_dict.get(weight_name, module.weight.detach())
        storage_key = tensor_storage_key(tensor)
        weight_names = aliases_by_storage.get(storage_key, (weight_name,))
        tied[name] = TiedLinearWeight(
            linear_name=name,
            weight_names=weight_names,
            shared_with_input_embedding=storage_key in input_embedding_keys,
        )
    return tied


def base_tensor_aliases(
    state_dict: Mapping[str, torch.Tensor],
    *,
    excluded_names: set[str],
) -> dict[str, str]:
    """Return non-quantized aliases that can be restored from one saved tensor."""
    aliases: dict[str, str] = {}
    for names in _aliases_by_storage(state_dict).values():
        saved_names = [name for name in names if name not in excluded_names]
        if len(saved_names) <= 1:
            continue
        canonical = saved_names[0]
        aliases.update({name: canonical for name in saved_names[1:]})
    return aliases


def tensor_storage_key(tensor: torch.Tensor) -> tuple[object, ...]:
    """Identify exact tensor aliases without conflating unrelated storage views."""
    if tensor.is_meta:
        return (
            "meta",
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            str(tensor.dtype),
        )
    storage = tensor.untyped_storage()
    return (
        storage.data_ptr(),
        tensor.storage_offset(),
        tensor.numel(),
        tensor.element_size(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
    )


def _aliases_by_storage(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[tuple[object, ...], tuple[str, ...]]:
    grouped: dict[tuple[object, ...], list[str]] = defaultdict(list)
    for name, tensor in state_dict.items():
        grouped[tensor_storage_key(tensor)].append(name)
    return {key: tuple(names) for key, names in grouped.items()}


def _input_embedding_storage_keys(model: torch.nn.Module) -> set[tuple[object, ...]]:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_input_embeddings):
        return set()
    embedding = get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return set()
    return {tensor_storage_key(weight)}
