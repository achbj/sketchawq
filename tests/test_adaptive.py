from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from sketchawq.config import SketchAWQConfig, load_config
from sketchawq.data.calibration import CalibrationBatch
from sketchawq.quant.adaptive import LayerAdaptiveInput, build_layer_adaptive_plans
from sketchawq.quant.hessian_sketch import HessianSketch
from sketchawq.quant.pipeline import quantize_model


def test_adaptive_plans_shift_budget_and_group_size_to_sensitive_layers() -> None:
    config = SketchAWQConfig()
    config.quantization.group_size = 128
    config.outliers.enabled = True
    config.outliers.strategy = "protect"
    config.outliers.fraction = 0.02
    config.adaptive.enabled = True
    config.adaptive.sensitive_group_size = 64
    config.adaptive.group_size_top_fraction = 0.5
    config.adaptive.sensitivity_top_fraction = 0.125
    config.adaptive.protected_fraction_min = 0.005
    config.adaptive.protected_fraction_max = 0.04

    calm = LayerAdaptiveInput(
        name="calm",
        weight=torch.ones(2, 8),
        mean_abs_activation=torch.ones(8),
        sketch=HessianSketch(diagonal=torch.ones(8), basis=None, rank=0, mode="diagonal"),
    )
    sensitive_weight = torch.ones(2, 8)
    sensitive_weight[:, 7] = 12.0
    sensitive = LayerAdaptiveInput(
        name="sensitive",
        weight=sensitive_weight,
        mean_abs_activation=torch.ones(8),
        sketch=HessianSketch(
            diagonal=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 16.0]),
            basis=None,
            rank=0,
            mode="diagonal",
        ),
    )

    plans = build_layer_adaptive_plans([calm, sensitive], config)

    assert plans["sensitive"].group_size == 64
    assert plans["calm"].group_size == 128
    assert plans["sensitive"].outlier_fraction > config.outliers.fraction
    assert plans["calm"].outlier_fraction < config.outliers.fraction


def test_adaptive_config_parses_from_yaml(tmp_path) -> None:
    path = tmp_path / "adaptive.yaml"
    path.write_text(
        """
adaptive:
  enabled: true
  sensitive_group_size: 64
  group_size_top_fraction: 0.25
  sensitivity_metric: hessian-weight
  sensitivity_top_fraction: 0.02
  protected_fraction_min: 0.005
  protected_fraction_max: 0.04
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.adaptive.enabled is True
    assert config.adaptive.sensitive_group_size == 64
    assert config.adaptive.group_size_top_fraction == 0.25
    assert config.adaptive.sensitivity_metric == "hessian_weight"
    assert config.adaptive.sensitivity_top_fraction == 0.02
    assert config.adaptive.protected_fraction_min == 0.005
    assert config.adaptive.protected_fraction_max == 0.04


def test_pipeline_applies_adaptive_group_size_and_reports_effective_fraction() -> None:
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
    input_ids = torch.randint(0, 32, (1, 16))
    batch = CalibrationBatch(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    config = SketchAWQConfig()
    config.quantization.bits = 4
    config.quantization.group_size = 8
    config.sketch.rank = 2
    config.sketch.max_rows_per_batch = 8
    config.outliers.enabled = True
    config.outliers.strategy = "protect"
    config.outliers.fraction = 0.25
    config.adaptive.enabled = True
    config.adaptive.sensitive_group_size = 4
    config.adaptive.group_size_top_fraction = 1.0
    config.adaptive.protected_fraction_min = 0.125
    config.adaptive.protected_fraction_max = 0.5

    report = quantize_model(model, [batch], config)

    assert "-adaptive" in report.method
    assert report.method.endswith("-tied-embedding-v1")
    assert report.layers
    assert all(layer.group_size == 4 for layer in report.layers)
    assert all(0.125 <= layer.outlier_fraction <= 0.5 for layer in report.layers)
    assert all(layer.adaptive_sensitivity_score > 0 for layer in report.layers)
    assert {layer.group_size for layer in report.packed_layers} == {4}
