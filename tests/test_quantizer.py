from __future__ import annotations

import torch

from sketchawq.quant.pack import pack_quantized_tensor, pack_unsigned, unpack_unsigned
from sketchawq.quant.quantizer import fake_quantize_weight, quantize_groupwise


def test_groupwise_quantization_preserves_shape() -> None:
    weight = torch.randn(7, 13)
    quantized = quantize_groupwise(weight, bits=3, group_size=8, symmetric=True)
    assert quantized.qweight.shape == weight.shape
    assert quantized.dequantize().shape == weight.shape


def test_asymmetric_clip_ratio_shrinks_around_group_center() -> None:
    weight = torch.tensor([[-0.1, 0.5, 0.2, 0.3]])

    quantized = quantize_groupwise(weight, bits=4, group_size=4, symmetric=False, clip_ratio=0.8)

    assert quantized.zero_point is not None
    torch.testing.assert_close(quantized.scale, torch.tensor([[0.032]]))
    torch.testing.assert_close(quantized.zero_point, torch.tensor([[1.0]]))


def test_awq_fake_quantization_preserves_shape() -> None:
    weight = torch.randn(5, 11)
    scales = torch.linspace(0.5, 1.5, steps=11)
    dequantized = fake_quantize_weight(
        weight,
        bits=4,
        group_size=4,
        symmetric=True,
        clip_ratio=0.95,
        input_scale=scales,
    )
    assert dequantized.shape == weight.shape


def test_pack_unsigned_outputs_bytes() -> None:
    values = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    packed = pack_unsigned(values, bits=3)
    assert packed.dtype == torch.uint8
    assert packed.numel() == 3


def test_pack_unsigned_round_trips_non_byte_aligned_values() -> None:
    values = torch.arange(37, dtype=torch.int16) % 8

    packed = pack_unsigned(values, bits=3)
    unpacked = unpack_unsigned(packed, bits=3, count=values.numel())

    torch.testing.assert_close(unpacked, values.to(torch.uint8))


def test_pack_quantized_tensor_round_trips_signed_qweights() -> None:
    weight = torch.randn(5, 13)
    quantized = quantize_groupwise(weight, bits=3, group_size=8, symmetric=True)

    packed = pack_quantized_tensor(quantized)

    assert packed.packed_weight.dtype == torch.uint8
    assert packed.packed_weight.numel() < quantized.qweight.numel()
    torch.testing.assert_close(packed.unpack_qweight(), quantized.qweight)
    torch.testing.assert_close(packed.dequantize(), quantized.dequantize())
