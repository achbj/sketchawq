from __future__ import annotations

import torch

from sketchawq.quant.loss import weighted_error_score
from sketchawq.quant.pack import pack_quantized_tensor, pack_unsigned, unpack_unsigned
from sketchawq.quant.quantizer import (
    fake_quantize_weight,
    fake_quantize_weight_with_sketch_refinement,
    quantize_groupwise,
    quantize_weight_with_input_scale,
    refine_quantized_scales,
    refine_quantized_scales_globally,
)


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


def test_sketch_refinement_reduces_weighted_error_without_changing_bit_budget() -> None:
    weight = torch.tensor([[1.4, 2.0, 2.5, 3.0]])
    diagonal = torch.tensor([100.0, 1.0, 1.0, 1.0])
    initial = quantize_weight_with_input_scale(
        weight,
        bits=3,
        group_size=4,
        symmetric=True,
        clip_ratio=1.0,
    )

    refined = refine_quantized_scales(
        weight,
        initial,
        diagonal=diagonal,
        basis=None,
        low_rank_weight=0.0,
        steps=4,
    )

    initial_score = weighted_error_score(
        weight,
        initial.dequantize(),
        diagonal=diagonal,
        normalize_terms=False,
    )
    refined_score = weighted_error_score(
        weight,
        refined.dequantize(),
        diagonal=diagonal,
        normalize_terms=False,
    )

    assert refined_score < initial_score
    assert refined.bits == initial.bits == 3
    assert refined.group_size == initial.group_size == 4
    assert refined.qweight.shape == initial.qweight.shape
    assert int(refined.qweight.min()) >= -4
    assert int(refined.qweight.max()) <= 3


def test_sketch_refined_fake_quantization_uses_input_scale_and_low_rank_basis() -> None:
    weight = torch.tensor([[0.7, 1.4, 2.2, 3.0]])
    input_scale = torch.tensor([0.5, 1.0, 1.5, 2.0])
    diagonal = torch.tensor([8.0, 1.0, 1.0, 1.0])
    basis = torch.tensor([[1.0], [1.0], [0.0], [0.0]])

    candidate = fake_quantize_weight_with_sketch_refinement(
        weight,
        bits=3,
        group_size=4,
        symmetric=True,
        clip_ratio=1.0,
        input_scale=input_scale,
        diagonal=diagonal,
        basis=basis,
        low_rank_weight=0.5,
        refinement_steps=3,
    )

    assert candidate.shape == weight.shape
    assert torch.isfinite(candidate).all()


def test_global_scale_refinement_reduces_coupled_factor_error_without_extra_storage() -> None:
    torch.manual_seed(0)
    weight = torch.randn(3, 8) * torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5, 0.8, 2.5, 0.7])
    factor = torch.tensor(
        [
            [1.0, 0.2],
            [0.4, 0.1],
            [0.2, 1.0],
            [0.1, 0.5],
            [1.0, 0.3],
            [0.3, 0.8],
            [0.7, 0.2],
            [0.2, 0.6],
        ]
    )
    initial = quantize_weight_with_input_scale(
        weight,
        bits=3,
        group_size=2,
        symmetric=True,
        clip_ratio=0.85,
    )

    refined = refine_quantized_scales_globally(
        weight,
        initial,
        diagonal=torch.ones(weight.shape[1]),
        factor=factor,
        steps=3,
    )

    initial_score = weighted_error_score(
        weight,
        initial.dequantize(),
        diagonal=torch.ones(weight.shape[1]),
        factor=factor,
        normalize_terms=False,
    )
    refined_score = weighted_error_score(
        weight,
        refined.dequantize(),
        diagonal=torch.ones(weight.shape[1]),
        factor=factor,
        normalize_terms=False,
    )

    assert refined_score < initial_score
    assert refined.bits == initial.bits
    assert refined.group_size == initial.group_size
    assert refined.qweight.shape == initial.qweight.shape
    assert refined.scale.shape == initial.scale.shape


def test_zero_refinement_steps_preserve_the_original_quantized_tensor() -> None:
    weight = torch.randn(3, 7)
    initial = quantize_groupwise(weight, bits=4, group_size=4, symmetric=True)

    refined = refine_quantized_scales(
        weight,
        initial,
        diagonal=torch.ones(weight.shape[1]),
        steps=0,
    )

    assert refined is initial


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
