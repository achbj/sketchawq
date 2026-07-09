from __future__ import annotations

from dataclasses import dataclass

import torch

from sketchawq.quant.quantizer import QuantizedTensor, refine_quantized_scales


SEQUENTIAL_COMPENSATION_BLOCK_SIZE = 1024
SEQUENTIAL_COMPENSATION_ROW_CHUNK_SIZE = 4096


@dataclass(frozen=True, slots=True)
class SequentialCompensationResult:
    quantized: QuantizedTensor
    block_count: int
    updated_block_count: int


def sequentially_compensate_quantized_weight(
    weight: torch.Tensor,
    quantized: QuantizedTensor,
    *,
    input_scale: torch.Tensor | None,
    diagonal: torch.Tensor | None,
    factor: torch.Tensor | None,
    basis: torch.Tensor | None,
    low_rank_weight: float,
    refinement_steps: int,
    block_size: int | None = None,
) -> SequentialCompensationResult:
    """Apply a blockwise GPTQ-style update using a diagonal-plus-low-rank sketch.

    For a curvature model ``H = D + F F^T``, each already-quantized block
    shifts the remaining full-precision target by ``E_B H_BR H_RR^-1``. The
    Woodbury identity lets the update use only the existing rank-sized sketch,
    avoiding a dense Hessian or inverse-Hessian allocation.
    """
    if weight.ndim != 2 or tuple(weight.shape) != quantized.original_shape:
        raise ValueError("weight shape must match the quantized tensor")
    if not quantized.symmetric:
        raise ValueError("sequential compensation requires symmetric quantization")
    if refinement_steps <= 0:
        raise ValueError("sequential compensation requires positive refinement_steps")

    device = weight.device
    input_features = weight.shape[1]
    curvature_factor = _curvature_factor(
        factor=factor,
        basis=basis,
        low_rank_weight=low_rank_weight,
        input_features=input_features,
        device=device,
    )
    curvature_diagonal = _curvature_diagonal(
        diagonal,
        input_features=input_features,
        device=device,
    )
    safe_input_scale = _input_scale(
        input_scale,
        input_features=input_features,
        device=device,
    )
    block_size = _aligned_block_size(
        requested=block_size,
        group_size=quantized.group_size,
    )

    working = weight.detach().to(device=device, dtype=torch.float32).clone()
    inverse_diagonal_factor = curvature_factor / curvature_diagonal.unsqueeze(1)
    rank = curvature_factor.shape[1]
    remaining_gram = torch.eye(rank, device=device, dtype=torch.float32)
    remaining_gram.add_(curvature_factor.transpose(0, 1) @ inverse_diagonal_factor)

    quantized_blocks: list[QuantizedTensor] = []
    block_count = 0
    updated_block_count = 0
    for start in range(0, input_features, block_size):
        end = min(start + block_size, input_features)
        initial_block = _slice_quantized_tensor(quantized, start=start, end=end)
        if start == 0:
            block_quantized = initial_block
        else:
            block_quantized = refine_quantized_scales(
                working[:, start:end],
                initial_block,
                input_scale=safe_input_scale[start:end],
                diagonal=curvature_diagonal[start:end],
                basis=curvature_factor[start:end],
                low_rank_weight=1.0,
                steps=refinement_steps,
            )
        quantized_blocks.append(block_quantized)
        block_count += 1

        current_factor = curvature_factor[start:end]
        current_inverse_diagonal_factor = inverse_diagonal_factor[start:end]
        remaining_gram.sub_(
            current_factor.transpose(0, 1) @ current_inverse_diagonal_factor
        )
        if end >= input_features:
            continue

        future_projection = _future_projection(
            remaining_gram,
            inverse_diagonal_factor[end:],
        )
        _update_future_weight_targets(
            working,
            block_quantized,
            start=start,
            end=end,
            input_scale=safe_input_scale[start:end],
            factor=current_factor,
            future_projection=future_projection,
        )
        updated_block_count += 1

    return SequentialCompensationResult(
        quantized=_merge_quantized_blocks(quantized_blocks, original=quantized),
        block_count=block_count,
        updated_block_count=updated_block_count,
    )


def _curvature_factor(
    *,
    factor: torch.Tensor | None,
    basis: torch.Tensor | None,
    low_rank_weight: float,
    input_features: int,
    device: torch.device,
) -> torch.Tensor:
    values = factor
    if values is None and basis is not None and low_rank_weight > 0:
        values = basis * low_rank_weight**0.5
    if values is None or values.numel() == 0:
        raise ValueError("sequential compensation requires a non-empty low-rank factor")
    values = values.detach().to(device=device, dtype=torch.float32)
    if values.ndim != 2 or values.shape[0] != input_features:
        raise ValueError("low-rank factor must have shape [input_channels, rank]")
    if not torch.isfinite(values).all():
        raise ValueError("low-rank factor must be finite")
    return values


def _curvature_diagonal(
    diagonal: torch.Tensor | None,
    *,
    input_features: int,
    device: torch.device,
) -> torch.Tensor:
    if diagonal is None:
        values = torch.ones(input_features, device=device, dtype=torch.float32)
    else:
        values = diagonal.detach().to(device=device, dtype=torch.float32)
    if values.ndim != 1 or values.shape[0] != input_features:
        raise ValueError("diagonal must have one value per input channel")
    if not torch.isfinite(values).all():
        raise ValueError("diagonal must be finite")
    return values.clamp_min(1e-6)


def _input_scale(
    input_scale: torch.Tensor | None,
    *,
    input_features: int,
    device: torch.device,
) -> torch.Tensor:
    if input_scale is None:
        values = torch.ones(input_features, device=device, dtype=torch.float32)
    else:
        values = input_scale.detach().to(device=device, dtype=torch.float32)
    if values.ndim != 1 or values.shape[0] != input_features:
        raise ValueError("input_scale must have one value per input channel")
    return values.clamp_min(1e-6)


def _aligned_block_size(*, requested: int | None, group_size: int) -> int:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    target = SEQUENTIAL_COMPENSATION_BLOCK_SIZE if requested is None else int(requested)
    if target <= 0:
        raise ValueError("block_size must be positive")
    target = max(target, group_size)
    return ((target + group_size - 1) // group_size) * group_size


def _slice_quantized_tensor(
    quantized: QuantizedTensor,
    *,
    start: int,
    end: int,
) -> QuantizedTensor:
    if start < 0 or end <= start or end > quantized.original_shape[1]:
        raise ValueError("invalid quantized block range")
    if start % quantized.group_size:
        raise ValueError("sequential compensation blocks must start on group boundaries")
    group_start = start // quantized.group_size
    group_end = (end + quantized.group_size - 1) // quantized.group_size
    return QuantizedTensor(
        qweight=quantized.qweight[:, start:end],
        scale=quantized.scale[:, group_start:group_end],
        zero_point=None,
        bits=quantized.bits,
        group_size=quantized.group_size,
        original_shape=(quantized.original_shape[0], end - start),
        symmetric=True,
    )


def _future_projection(
    remaining_gram: torch.Tensor,
    inverse_diagonal_factor: torch.Tensor,
) -> torch.Tensor:
    gram = (remaining_gram + remaining_gram.transpose(0, 1)) * 0.5
    return torch.linalg.solve(gram, inverse_diagonal_factor.transpose(0, 1))


def _update_future_weight_targets(
    working: torch.Tensor,
    quantized_block: QuantizedTensor,
    *,
    start: int,
    end: int,
    input_scale: torch.Tensor,
    factor: torch.Tensor,
    future_projection: torch.Tensor,
) -> None:
    for row_start in range(0, working.shape[0], SEQUENTIAL_COMPENSATION_ROW_CHUNK_SIZE):
        row_end = min(row_start + SEQUENTIAL_COMPENSATION_ROW_CHUNK_SIZE, working.shape[0])
        candidate = _dequantize_rows(
            quantized_block,
            row_start=row_start,
            row_end=row_end,
            input_scale=input_scale,
            device=working.device,
        )
        error = working[row_start:row_end, start:end] - candidate
        correction = (error @ factor) @ future_projection
        working[row_start:row_end, end:].add_(correction)


def _dequantize_rows(
    quantized: QuantizedTensor,
    *,
    row_start: int,
    row_end: int,
    input_scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    qweight = quantized.qweight[row_start:row_end].to(device=device, dtype=torch.float32)
    width = qweight.shape[1]
    remainder = width % quantized.group_size
    pad = 0 if remainder == 0 else quantized.group_size - remainder
    if pad:
        qweight = torch.nn.functional.pad(qweight, (0, pad))
    grouped = qweight.reshape(qweight.shape[0], -1, quantized.group_size)
    scale = quantized.scale[row_start:row_end].to(device=device, dtype=torch.float32)
    values = (grouped * scale.unsqueeze(-1)).reshape(qweight.shape[0], -1)
    if pad:
        values = values[:, :-pad]
    return values / input_scale.unsqueeze(0)


def _merge_quantized_blocks(
    blocks: list[QuantizedTensor],
    *,
    original: QuantizedTensor,
) -> QuantizedTensor:
    if not blocks:
        raise ValueError("at least one quantized block is required")
    qweight = torch.cat([block.qweight for block in blocks], dim=1)
    scale = torch.cat([block.scale for block in blocks], dim=1)
    return QuantizedTensor(
        qweight=qweight,
        scale=scale,
        zero_point=None,
        bits=original.bits,
        group_size=original.group_size,
        original_shape=original.original_shape,
        symmetric=True,
    )
