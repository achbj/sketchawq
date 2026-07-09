from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class QuantizedTensor:
    qweight: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor | None
    bits: int
    group_size: int
    original_shape: tuple[int, ...]
    symmetric: bool

    def dequantize(self) -> torch.Tensor:
        q = self.qweight.float()
        padded, pad = _pad_last_dim(q, self.group_size)
        out_features, padded_in_features = padded.shape
        groups_per_row = padded_in_features // self.group_size
        grouped = padded.reshape(out_features, groups_per_row, self.group_size)
        scale = self.scale.to(device=grouped.device, dtype=grouped.dtype).unsqueeze(-1)
        if self.symmetric:
            values = grouped * scale
        else:
            assert self.zero_point is not None
            zero_point = self.zero_point.to(
                device=grouped.device,
                dtype=grouped.dtype,
            ).unsqueeze(-1)
            values = (grouped - zero_point) * scale
        values = values.reshape(out_features, padded_in_features)
        if pad:
            values = values[:, :-pad]
        return values.reshape(self.original_shape)


def _pad_last_dim(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    last_dim = weight.shape[-1]
    remainder = last_dim % group_size
    if remainder == 0:
        return weight, 0
    pad = group_size - remainder
    padded = torch.nn.functional.pad(weight, (0, pad))
    return padded, pad


def quantize_groupwise(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    symmetric: bool = True,
    clip_ratio: float = 1.0,
) -> QuantizedTensor:
    """Quantize a 2D linear weight matrix along the input-channel dimension."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight matrix, got shape {tuple(weight.shape)}")
    if not 0 < clip_ratio <= 1.0:
        raise ValueError(f"clip_ratio must be in (0, 1], got {clip_ratio}")

    original_shape = tuple(weight.shape)
    work = weight.float()
    padded, pad = _pad_last_dim(work, group_size)
    out_features, padded_in_features = padded.shape
    grouped = padded.reshape(out_features, padded_in_features // group_size, group_size)

    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        max_abs = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) * clip_ratio
        scale = max_abs / max(qmax, 1)
        clipped = grouped.clamp(-max_abs, max_abs)
        qweight = torch.round(clipped / scale).clamp(qmin, qmax)
        zero_point = None
    else:
        qmin = 0
        qmax = 2**bits - 1
        raw_min = grouped.amin(dim=-1, keepdim=True)
        raw_max = grouped.amax(dim=-1, keepdim=True)
        center = (raw_min + raw_max) * 0.5
        half_range = (raw_max - raw_min).clamp_min(1e-8) * 0.5 * clip_ratio
        min_val = center - half_range
        max_val = center + half_range
        scale = (max_val - min_val).clamp_min(1e-8) / qmax
        zero_point = torch.round(qmin - min_val / scale).clamp(qmin, qmax)
        clipped = grouped.clamp(min_val, max_val)
        qweight = torch.round(clipped / scale + zero_point).clamp(qmin, qmax)

    dequant_grouped = qweight.float() * scale if symmetric else (qweight - zero_point) * scale
    dequant = dequant_grouped.reshape(out_features, padded_in_features)
    if pad:
        dequant = dequant[:, :-pad]
        qweight = qweight.reshape(out_features, padded_in_features)[:, :-pad]
    else:
        qweight = qweight.reshape(out_features, padded_in_features)

    return QuantizedTensor(
        qweight=qweight.to(torch.int16),
        scale=scale.squeeze(-1).detach().cpu(),
        zero_point=None if zero_point is None else zero_point.squeeze(-1).detach().cpu(),
        bits=bits,
        group_size=group_size,
        original_shape=original_shape,
        symmetric=symmetric,
    )


def quantize_weight_with_input_scale(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
    clip_ratio: float,
    input_scale: torch.Tensor | None = None,
) -> QuantizedTensor:
    """Quantize a weight matrix, including AWQ's equivalent input scaling."""
    work = weight.float()
    if input_scale is not None:
        safe_scale = input_scale.to(device=work.device, dtype=work.dtype).clamp_min(1e-6)
        work = work * safe_scale.view(1, -1)
    return quantize_groupwise(
        work,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        clip_ratio=clip_ratio,
    )


def refine_quantized_scales(
    weight: torch.Tensor,
    quantized: QuantizedTensor,
    *,
    input_scale: torch.Tensor | None = None,
    diagonal: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    low_rank_weight: float = 0.15,
    steps: int = 3,
) -> QuantizedTensor:
    """Refit symmetric group scales against a diagonal-plus-low-rank sketch."""
    if steps <= 0:
        return quantized
    if not quantized.symmetric:
        raise ValueError("Sketch-conditioned scale refinement requires symmetric quantization")
    if weight.ndim != 2 or tuple(weight.shape) != quantized.original_shape:
        raise ValueError("weight shape must match the quantized tensor")
    if low_rank_weight < 0:
        raise ValueError("low_rank_weight must be non-negative")

    device = weight.device
    work = weight.detach().to(device=device, dtype=torch.float32)
    qweight = quantized.qweight.detach().to(device=device, dtype=torch.float32)
    padded_weight, pad = _pad_last_dim(work, quantized.group_size)
    padded_qweight, _ = _pad_last_dim(qweight, quantized.group_size)
    out_features, padded_in_features = padded_weight.shape
    groups_per_row = padded_in_features // quantized.group_size
    grouped_weight = padded_weight.reshape(
        out_features,
        groups_per_row,
        quantized.group_size,
    )
    grouped_qweight = padded_qweight.reshape_as(grouped_weight)

    if input_scale is None:
        safe_input_scale = torch.ones(weight.shape[1], device=device, dtype=torch.float32)
    else:
        safe_input_scale = input_scale.detach().to(
            device=device,
            dtype=torch.float32,
        ).clamp_min(1e-6)
    safe_input_scale = _pad_vector(
        safe_input_scale,
        padded_in_features,
        fill_value=1.0,
    ).reshape(groups_per_row, quantized.group_size)

    if diagonal is None:
        diagonal_weights = torch.ones(weight.shape[1], device=device, dtype=torch.float32)
    else:
        diagonal_weights = diagonal.detach().to(device=device, dtype=torch.float32).clamp_min(0)
    diagonal_weights = _pad_vector(
        diagonal_weights,
        padded_in_features,
        fill_value=0.0 if pad else 1.0,
    ).reshape(groups_per_row, quantized.group_size)

    grouped_basis = None
    if basis is not None and basis.numel() > 0 and low_rank_weight > 0:
        basis_work = basis.detach().to(device=device, dtype=torch.float32)
        if basis_work.ndim != 2 or basis_work.shape[0] != weight.shape[1]:
            raise ValueError("basis must have shape [input_channels, rank]")
        if pad:
            basis_work = torch.nn.functional.pad(basis_work, (0, 0, 0, pad))
        grouped_basis = basis_work.reshape(
            groups_per_row,
            quantized.group_size,
            basis_work.shape[1],
        )

    qmin = -(2 ** (quantized.bits - 1))
    qmax = 2 ** (quantized.bits - 1) - 1
    valid_mask = torch.ones(
        padded_in_features,
        device=device,
        dtype=torch.float32,
    )
    if pad:
        valid_mask[-pad:] = 0
    grouped_valid_mask = valid_mask.reshape(groups_per_row, quantized.group_size)

    for _ in range(steps):
        scales = _fit_symmetric_group_scales(
            grouped_weight,
            grouped_qweight,
            safe_input_scale,
            diagonal_weights,
            grouped_basis,
            low_rank_weight=low_rank_weight,
        )
        scaled_target = grouped_weight * safe_input_scale.unsqueeze(0)
        grouped_qweight = torch.round(
            scaled_target / scales.unsqueeze(-1)
        ).clamp(qmin, qmax)
        grouped_qweight = grouped_qweight * grouped_valid_mask.unsqueeze(0)

    scales = _fit_symmetric_group_scales(
        grouped_weight,
        grouped_qweight,
        safe_input_scale,
        diagonal_weights,
        grouped_basis,
        low_rank_weight=low_rank_weight,
    )
    flat_qweight = grouped_qweight.reshape(out_features, padded_in_features)
    if pad:
        flat_qweight = flat_qweight[:, :-pad]

    return QuantizedTensor(
        qweight=flat_qweight.to(torch.int16),
        scale=scales.detach().cpu(),
        zero_point=None,
        bits=quantized.bits,
        group_size=quantized.group_size,
        original_shape=quantized.original_shape,
        symmetric=True,
    )


def refine_quantized_scales_globally(
    weight: torch.Tensor,
    quantized: QuantizedTensor,
    *,
    input_scale: torch.Tensor | None = None,
    diagonal: torch.Tensor | None = None,
    factor: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    low_rank_weight: float = 0.15,
    steps: int = 3,
) -> QuantizedTensor:
    """Refit scales while preserving the low-rank residual across all groups."""
    if steps <= 0:
        return quantized
    if not quantized.symmetric:
        raise ValueError("Global sketch scale refinement requires symmetric quantization")
    if weight.ndim != 2 or tuple(weight.shape) != quantized.original_shape:
        raise ValueError("weight shape must match the quantized tensor")
    if low_rank_weight < 0:
        raise ValueError("low_rank_weight must be non-negative")

    device = weight.device
    work = weight.detach().to(device=device, dtype=torch.float32)
    qweight = quantized.qweight.detach().to(device=device, dtype=torch.float32)
    padded_weight, pad = _pad_last_dim(work, quantized.group_size)
    padded_qweight, _ = _pad_last_dim(qweight, quantized.group_size)
    out_features, padded_in_features = padded_weight.shape
    groups_per_row = padded_in_features // quantized.group_size
    grouped_weight = padded_weight.reshape(
        out_features,
        groups_per_row,
        quantized.group_size,
    )
    grouped_qweight = padded_qweight.reshape_as(grouped_weight)
    grouped_input_scale = _grouped_input_scale(
        input_scale,
        input_features=weight.shape[1],
        padded_input_features=padded_in_features,
        group_size=quantized.group_size,
        device=device,
    )
    grouped_diagonal = _grouped_diagonal(
        diagonal,
        input_features=weight.shape[1],
        padded_input_features=padded_in_features,
        group_size=quantized.group_size,
        pad=pad,
        device=device,
    )
    grouped_factor = _grouped_low_rank_factor(
        factor=factor,
        basis=basis,
        low_rank_weight=low_rank_weight,
        input_features=weight.shape[1],
        padded_input_features=padded_in_features,
        group_size=quantized.group_size,
        device=device,
    )
    scales = quantized.scale.detach().to(device=device, dtype=torch.float32).clone()
    expected_scale_shape = (out_features, groups_per_row)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(f"scale must have shape {expected_scale_shape}")

    qmin = -(2 ** (quantized.bits - 1))
    qmax = 2 ** (quantized.bits - 1) - 1
    valid_mask = torch.ones(padded_in_features, device=device, dtype=torch.float32)
    if pad:
        valid_mask[-pad:] = 0
    grouped_valid_mask = valid_mask.reshape(groups_per_row, quantized.group_size)

    objective = _global_refinement_objective(
        grouped_weight,
        grouped_qweight,
        scales,
        grouped_input_scale,
        grouped_diagonal,
        grouped_factor,
    )
    for _ in range(steps):
        previous_scales = scales.clone()
        residual = _global_low_rank_residual(
            grouped_weight,
            grouped_qweight,
            scales,
            grouped_input_scale,
            grouped_factor,
        )
        for group_index in range(groups_per_row):
            group_weight = grouped_weight[:, group_index, :]
            group_qweight = grouped_qweight[:, group_index, :]
            group_input_scale = grouped_input_scale[group_index]
            group_diagonal = grouped_diagonal[group_index]
            group_factor = (
                None if grouped_factor is None else grouped_factor[group_index]
            )
            previous_error = group_weight - (
                scales[:, group_index].unsqueeze(-1)
                * group_qweight
                / group_input_scale.unsqueeze(0)
            )
            residual_without = (
                None if residual is None else residual - previous_error @ group_factor
            )

            # Two scale-rounding updates keep each integer group tied to its scale.
            for _ in range(2):
                scales[:, group_index] = _fit_global_group_scale(
                    group_weight,
                    group_qweight,
                    group_input_scale,
                    group_diagonal,
                    group_factor,
                    residual_without,
                )
                group_qweight = _round_group_qweight(
                    group_weight,
                    group_input_scale,
                    scales[:, group_index],
                    group_valid_mask=grouped_valid_mask[group_index],
                    qmin=qmin,
                    qmax=qmax,
                )
            grouped_qweight[:, group_index, :] = group_qweight
            if residual is not None:
                updated_error = group_weight - (
                    scales[:, group_index].unsqueeze(-1)
                    * group_qweight
                    / group_input_scale.unsqueeze(0)
                )
                residual = residual_without + updated_error @ group_factor

        updated_objective = _global_refinement_objective(
            grouped_weight,
            grouped_qweight,
            scales,
            grouped_input_scale,
            grouped_diagonal,
            grouped_factor,
        )
        if not torch.isfinite(updated_objective) or updated_objective > objective:
            scales = previous_scales
            _round_all_groups(
                grouped_weight,
                grouped_qweight,
                scales,
                grouped_input_scale,
                grouped_valid_mask,
                qmin=qmin,
                qmax=qmax,
            )
            break
        objective = updated_objective

    flat_qweight = grouped_qweight.reshape(out_features, padded_in_features)
    if pad:
        flat_qweight = flat_qweight[:, :-pad]
    return QuantizedTensor(
        qweight=flat_qweight.to(torch.int16),
        scale=scales.detach().cpu(),
        zero_point=None,
        bits=quantized.bits,
        group_size=quantized.group_size,
        original_shape=quantized.original_shape,
        symmetric=True,
    )


def _grouped_input_scale(
    input_scale: torch.Tensor | None,
    *,
    input_features: int,
    padded_input_features: int,
    group_size: int,
    device: torch.device,
) -> torch.Tensor:
    if input_scale is None:
        values = torch.ones(input_features, device=device, dtype=torch.float32)
    else:
        values = input_scale.detach().to(device=device, dtype=torch.float32).clamp_min(1e-6)
    return _pad_vector(values, padded_input_features, fill_value=1.0).reshape(-1, group_size)


def _grouped_diagonal(
    diagonal: torch.Tensor | None,
    *,
    input_features: int,
    padded_input_features: int,
    group_size: int,
    pad: int,
    device: torch.device,
) -> torch.Tensor:
    if diagonal is None:
        values = torch.ones(input_features, device=device, dtype=torch.float32)
    else:
        values = diagonal.detach().to(device=device, dtype=torch.float32).clamp_min(0)
    return _pad_vector(
        values,
        padded_input_features,
        fill_value=0.0 if pad else 1.0,
    ).reshape(-1, group_size)


def _grouped_low_rank_factor(
    *,
    factor: torch.Tensor | None,
    basis: torch.Tensor | None,
    low_rank_weight: float,
    input_features: int,
    padded_input_features: int,
    group_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    values = factor
    if values is None and basis is not None and basis.numel() > 0 and low_rank_weight > 0:
        values = basis * low_rank_weight**0.5
    if values is None or values.numel() == 0:
        return None
    values = values.detach().to(device=device, dtype=torch.float32)
    if values.ndim != 2 or values.shape[0] != input_features:
        raise ValueError("low-rank factor must have shape [input_channels, rank]")
    if padded_input_features > input_features:
        values = torch.nn.functional.pad(values, (0, 0, 0, padded_input_features - input_features))
    return values.reshape(-1, group_size, values.shape[1])


def _global_refinement_objective(
    grouped_weight: torch.Tensor,
    grouped_qweight: torch.Tensor,
    scales: torch.Tensor,
    grouped_input_scale: torch.Tensor,
    grouped_diagonal: torch.Tensor,
    grouped_factor: torch.Tensor | None,
) -> torch.Tensor:
    diagonal_error = torch.zeros((), device=grouped_weight.device, dtype=grouped_weight.dtype)
    residual = (
        None
        if grouped_factor is None
        else torch.zeros(
            grouped_weight.shape[0],
            grouped_factor.shape[-1],
            device=grouped_weight.device,
            dtype=grouped_weight.dtype,
        )
    )
    for group_index in range(grouped_weight.shape[1]):
        error = grouped_weight[:, group_index, :] - (
            scales[:, group_index].unsqueeze(-1)
            * grouped_qweight[:, group_index, :]
            / grouped_input_scale[group_index].unsqueeze(0)
        )
        diagonal_error = diagonal_error + (
            error.square() * grouped_diagonal[group_index].unsqueeze(0)
        ).sum()
        if residual is not None:
            residual.add_(error @ grouped_factor[group_index])
    return diagonal_error if residual is None else diagonal_error + residual.square().sum()


def _global_low_rank_residual(
    grouped_weight: torch.Tensor,
    grouped_qweight: torch.Tensor,
    scales: torch.Tensor,
    grouped_input_scale: torch.Tensor,
    grouped_factor: torch.Tensor | None,
) -> torch.Tensor | None:
    if grouped_factor is None:
        return None
    residual = torch.zeros(
        grouped_weight.shape[0],
        grouped_factor.shape[-1],
        device=grouped_weight.device,
        dtype=grouped_weight.dtype,
    )
    for group_index in range(grouped_weight.shape[1]):
        error = grouped_weight[:, group_index, :] - (
            scales[:, group_index].unsqueeze(-1)
            * grouped_qweight[:, group_index, :]
            / grouped_input_scale[group_index].unsqueeze(0)
        )
        residual.add_(error @ grouped_factor[group_index])
    return residual


def _fit_global_group_scale(
    group_weight: torch.Tensor,
    group_qweight: torch.Tensor,
    group_input_scale: torch.Tensor,
    group_diagonal: torch.Tensor,
    group_factor: torch.Tensor | None,
    residual_without: torch.Tensor | None,
) -> torch.Tensor:
    dequant_vector = group_qweight / group_input_scale.unsqueeze(0)
    numerator = (
        group_diagonal.unsqueeze(0) * dequant_vector * group_weight
    ).sum(dim=-1)
    denominator = (
        group_diagonal.unsqueeze(0) * dequant_vector.square()
    ).sum(dim=-1)
    if group_factor is not None:
        if residual_without is None:
            raise ValueError("low-rank residual is required for a low-rank factor")
        vector_projection = dequant_vector @ group_factor
        target_projection = group_weight @ group_factor
        numerator = numerator + (
            vector_projection * (residual_without + target_projection)
        ).sum(dim=-1)
        denominator = denominator + vector_projection.square().sum(dim=-1)
    return (numerator / denominator.clamp_min(1e-12)).clamp_min(1e-8)


def _round_group_qweight(
    group_weight: torch.Tensor,
    group_input_scale: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_valid_mask: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    scaled_target = group_weight * group_input_scale.unsqueeze(0)
    qweight = torch.round(scaled_target / scale.unsqueeze(-1)).clamp(qmin, qmax)
    return qweight * group_valid_mask.unsqueeze(0)


def _round_all_groups(
    grouped_weight: torch.Tensor,
    grouped_qweight: torch.Tensor,
    scales: torch.Tensor,
    grouped_input_scale: torch.Tensor,
    grouped_valid_mask: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
) -> None:
    for group_index in range(grouped_weight.shape[1]):
        grouped_qweight[:, group_index, :] = _round_group_qweight(
            grouped_weight[:, group_index, :],
            grouped_input_scale[group_index],
            scales[:, group_index],
            group_valid_mask=grouped_valid_mask[group_index],
            qmin=qmin,
            qmax=qmax,
        )


def _fit_symmetric_group_scales(
    grouped_weight: torch.Tensor,
    grouped_qweight: torch.Tensor,
    grouped_input_scale: torch.Tensor,
    grouped_diagonal: torch.Tensor,
    grouped_basis: torch.Tensor | None,
    *,
    low_rank_weight: float,
) -> torch.Tensor:
    dequant_vectors = grouped_qweight / grouped_input_scale.unsqueeze(0)
    diagonal = grouped_diagonal.unsqueeze(0)
    numerator = (diagonal * dequant_vectors * grouped_weight).sum(dim=-1)
    denominator = (diagonal * dequant_vectors.square()).sum(dim=-1)

    if grouped_basis is not None and low_rank_weight > 0:
        vector_projection = torch.einsum(
            "ogc,gcr->ogr",
            dequant_vectors,
            grouped_basis,
        )
        weight_projection = torch.einsum(
            "ogc,gcr->ogr",
            grouped_weight,
            grouped_basis,
        )
        numerator = numerator + low_rank_weight * (
            vector_projection * weight_projection
        ).sum(dim=-1)
        denominator = denominator + low_rank_weight * vector_projection.square().sum(
            dim=-1
        )

    return (numerator / denominator.clamp_min(1e-12)).clamp_min(1e-8)


def _pad_vector(
    values: torch.Tensor,
    target_length: int,
    *,
    fill_value: float,
) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("expected a one-dimensional input-channel vector")
    if values.shape[0] > target_length:
        raise ValueError("input-channel vector is longer than the padded weight")
    if values.shape[0] == target_length:
        return values
    return torch.nn.functional.pad(
        values,
        (0, target_length - values.shape[0]),
        value=fill_value,
    )


def dequantize_weight_with_input_scale(
    quantized: QuantizedTensor,
    *,
    input_scale: torch.Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Dequantize a qweight tensor and undo AWQ's equivalent input scaling."""
    dequantized = quantized.dequantize()
    if device is not None:
        dequantized = dequantized.to(device=device)
    dequantized = dequantized.float()
    if input_scale is not None:
        safe_scale = input_scale.to(
            device=dequantized.device,
            dtype=dequantized.dtype,
        ).clamp_min(1e-6)
        dequantized = dequantized / safe_scale.view(1, -1)
    if dtype is not None:
        dequantized = dequantized.to(dtype=dtype)
    return dequantized


def fake_quantize_weight(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
    clip_ratio: float,
    input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return dequantized weights after groupwise quantization.

    If input_scale is provided, we simulate AWQ's equivalent scaling:
    quantize W * s, then divide the dequantized result by s.
    """
    quantized = quantize_weight_with_input_scale(
        weight,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        clip_ratio=clip_ratio,
        input_scale=input_scale,
    )
    return dequantize_weight_with_input_scale(
        quantized,
        input_scale=input_scale,
        device=weight.device,
        dtype=weight.dtype,
    )


def fake_quantize_weight_with_sketch_refinement(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
    clip_ratio: float,
    input_scale: torch.Tensor | None = None,
    diagonal: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    low_rank_weight: float = 0.15,
    refinement_steps: int = 3,
) -> torch.Tensor:
    quantized = quantize_weight_with_input_scale(
        weight,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        clip_ratio=clip_ratio,
        input_scale=input_scale,
    )
    quantized = refine_quantized_scales(
        weight,
        quantized,
        input_scale=input_scale,
        diagonal=diagonal,
        basis=basis,
        low_rank_weight=low_rank_weight,
        steps=refinement_steps,
    )
    return dequantize_weight_with_input_scale(
        quantized,
        input_scale=input_scale,
        device=weight.device,
        dtype=weight.dtype,
    )
