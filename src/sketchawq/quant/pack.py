from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

from sketchawq.quant.quantizer import QuantizedTensor


_PACK_CHUNK_VALUES = 1_000_000


@dataclass(frozen=True, slots=True)
class PackedQuantizedTensor:
    packed_weight: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor | None
    bits: int
    group_size: int
    original_shape: tuple[int, ...]
    symmetric: bool

    def unpack_qweight(self) -> torch.Tensor:
        unsigned = unpack_unsigned(
            self.packed_weight,
            bits=self.bits,
            count=prod(self.original_shape),
        ).reshape(self.original_shape)
        if self.symmetric:
            qmin = -(2 ** (self.bits - 1))
            return (unsigned.to(torch.int16) + qmin).reshape(self.original_shape)
        return unsigned.to(torch.int16).reshape(self.original_shape)

    def dequantize(self) -> torch.Tensor:
        return QuantizedTensor(
            qweight=self.unpack_qweight(),
            scale=self.scale,
            zero_point=self.zero_point,
            bits=self.bits,
            group_size=self.group_size,
            original_shape=self.original_shape,
            symmetric=self.symmetric,
        ).dequantize()


def pack_unsigned(values: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Pack non-negative integer values into uint8 bytes.

    Values are packed least-significant-bit first. The implementation works in
    chunks so layer-sized tensors can be packed without one giant arange.
    """
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in the range [1, 8]")
    flat = values.detach().to(device="cpu", dtype=torch.int64).flatten()
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    if int(flat.min()) < 0 or int(flat.max()) >= 2**bits:
        raise ValueError(f"values must be in [0, {2**bits - 1}] for {bits}-bit packing")

    total_bits = flat.numel() * bits
    packed = torch.zeros((total_bits + 7) // 8, dtype=torch.int16)
    for start in range(0, flat.numel(), _PACK_CHUNK_VALUES):
        end = min(start + _PACK_CHUNK_VALUES, flat.numel())
        indexes = torch.arange(start, end, dtype=torch.int64)
        chunk = flat[start:end]
        bit_positions = indexes * bits
        byte_indexes = bit_positions // 8
        bit_offsets = bit_positions % 8

        low = ((chunk << bit_offsets) & 0xFF).to(torch.int16)
        packed.scatter_add_(0, byte_indexes, low)

        spans_next_byte = bit_offsets + bits > 8
        if torch.any(spans_next_byte):
            high = (chunk[spans_next_byte] >> (8 - bit_offsets[spans_next_byte])).to(torch.int16)
            packed.scatter_add_(0, byte_indexes[spans_next_byte] + 1, high)
    return packed.to(torch.uint8)


def unpack_unsigned(packed: torch.Tensor, *, bits: int, count: int) -> torch.Tensor:
    """Unpack uint8 bytes produced by :func:`pack_unsigned`."""
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in the range [1, 8]")
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return torch.empty(0, dtype=torch.uint8)

    packed_cpu = packed.detach().to(device="cpu", dtype=torch.uint8).flatten()
    required_bytes = (count * bits + 7) // 8
    if packed_cpu.numel() < required_bytes:
        raise ValueError(f"packed tensor has {packed_cpu.numel()} bytes, expected {required_bytes}")

    packed_int = packed_cpu.to(torch.int64)
    values = torch.empty(count, dtype=torch.uint8)
    mask = (1 << bits) - 1
    for start in range(0, count, _PACK_CHUNK_VALUES):
        end = min(start + _PACK_CHUNK_VALUES, count)
        indexes = torch.arange(start, end, dtype=torch.int64)
        bit_positions = indexes * bits
        byte_indexes = bit_positions // 8
        bit_offsets = bit_positions % 8

        chunk = packed_int[byte_indexes] >> bit_offsets
        spans_next_byte = bit_offsets + bits > 8
        if torch.any(spans_next_byte):
            chunk[spans_next_byte] |= (
                packed_int[byte_indexes[spans_next_byte] + 1]
                << (8 - bit_offsets[spans_next_byte])
            )
        values[start:end] = (chunk & mask).to(torch.uint8)
    return values


def pack_quantized_tensor(quantized: QuantizedTensor) -> PackedQuantizedTensor:
    """Pack a groupwise quantized tensor's integer qweights into bytes."""
    qweight = quantized.qweight.detach().to(device="cpu", dtype=torch.int16)
    if quantized.symmetric:
        qmin = -(2 ** (quantized.bits - 1))
        unsigned = qweight - qmin
    else:
        unsigned = qweight
    return PackedQuantizedTensor(
        packed_weight=pack_unsigned(unsigned, bits=quantized.bits),
        scale=quantized.scale.detach().cpu(),
        zero_point=None if quantized.zero_point is None else quantized.zero_point.detach().cpu(),
        bits=quantized.bits,
        group_size=quantized.group_size,
        original_shape=quantized.original_shape,
        symmetric=quantized.symmetric,
    )
