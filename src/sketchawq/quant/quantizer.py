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
