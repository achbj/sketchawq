from __future__ import annotations

import json

import torch
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel

from sketchawq.config import SketchAWQConfig
from sketchawq.data.calibration import CalibrationBatch
from sketchawq.models.save import save_quantized_artifacts
from sketchawq.quant.pipeline import quantize_model
from sketchawq.reporting.metrics import make_run_metadata


def test_save_quantized_artifacts_writes_packed_model_instead_of_dense_simulation(tmp_path) -> None:
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
    config.output.save_model = True

    quantization = quantize_model(model, [batch], config)
    output_dir = tmp_path / "run"
    report = {
        "metadata": make_run_metadata(config, config_path="tiny.yaml"),
        "output_dir": str(output_dir),
        "quantization": quantization.to_dict(),
    }

    save_quantized_artifacts(
        model=model,
        tokenizer=None,
        config=config,
        report=report,
        output_dir=output_dir,
        packed_layers=quantization.packed_layers,
    )

    manifest = json.loads((output_dir / "packed_manifest.json").read_text(encoding="utf-8"))
    tensors = load_file(output_dir / "packed_model.safetensors")

    assert manifest["format"] == "sketchawq-packed-v1"
    assert manifest["layers"]
    first_layer = manifest["layers"][0]
    assert first_layer["packed_weight"] in tensors
    assert first_layer["scale"] in tensors
    assert set(first_layer["weight_names"]) == {"lm_head.weight", "transformer.wte.weight"}
    assert "lm_head.weight" not in tensors
    assert "transformer.wte.weight" not in tensors
    assert not (output_dir / "model").exists()
