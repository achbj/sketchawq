from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import save_file

from sketchawq.config import SketchAWQConfig, config_to_dict
from sketchawq.quant.pipeline import PackedLayerArtifact
from sketchawq.reporting.metrics import write_plot_artifacts


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_quantized_artifacts(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    config: SketchAWQConfig,
    report: dict[str, Any],
    output_dir: str | Path,
    packed_layers: Sequence[PackedLayerArtifact] = (),
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(config_to_dict(config), output_dir / "config.json")
    save_json(report, output_dir / "quantization_report.json")
    write_plot_artifacts(report, config, output_dir)

    if config.output.save_model:
        if packed_layers:
            save_packed_model_artifacts(
                model=model,
                config=config,
                packed_layers=packed_layers,
                output_dir=output_dir,
            )
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir / "tokenizer")
        if config.output.save_dense_simulation:
            model.save_pretrained(output_dir / "model")
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir / "model")


def save_packed_model_artifacts(
    *,
    model: torch.nn.Module,
    config: SketchAWQConfig,
    packed_layers: Sequence[PackedLayerArtifact],
    output_dir: str | Path,
) -> None:
    """Save a packed integer-qweight checkpoint plus a JSON manifest.

    Non-quantized tensors keep their original state-dict names. Quantized linear
    weights are replaced by packed byte tensors plus scales and AWQ input scales.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quantized_weight_names = {f"{layer.name}.weight" for layer in packed_layers}
    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name in quantized_weight_names:
            continue
        tensors[name] = tensor.detach().cpu().contiguous()

    manifest_layers: list[dict[str, Any]] = []
    for index, layer in enumerate(packed_layers):
        prefix = f"quantized_layers.{index}"
        packed_key = f"{prefix}.packed_weight"
        scale_key = f"{prefix}.scale"
        input_scale_key = f"{prefix}.input_scale"
        tensors[packed_key] = layer.packed_weight.detach().cpu().contiguous()
        tensors[scale_key] = layer.scale.detach().cpu().contiguous()
        tensors[input_scale_key] = layer.input_scale.detach().cpu().contiguous()

        zero_point_key = None
        if layer.zero_point is not None:
            zero_point_key = f"{prefix}.zero_point"
            tensors[zero_point_key] = layer.zero_point.detach().cpu().contiguous()

        protected_indices_key = None
        protected_weight_key = None
        if layer.protected_weight is not None and layer.protected_indices.numel() > 0:
            protected_indices_key = f"{prefix}.protected_indices"
            protected_weight_key = f"{prefix}.protected_weight"
            tensors[protected_indices_key] = layer.protected_indices.detach().cpu().contiguous()
            tensors[protected_weight_key] = layer.protected_weight.detach().cpu().contiguous()

        manifest_layers.append(
            {
                "name": layer.name,
                "weight": f"{layer.name}.weight",
                "packed_weight": packed_key,
                "scale": scale_key,
                "zero_point": zero_point_key,
                "input_scale": input_scale_key,
                "protected_indices": protected_indices_key,
                "protected_weight": protected_weight_key,
                "bits": layer.bits,
                "group_size": layer.group_size,
                "symmetric": layer.symmetric,
                "original_shape": list(layer.original_shape),
            }
        )

    save_file(tensors, output_dir / "packed_model.safetensors")
    save_json(
        {
            "format": "sketchawq-packed-v1",
            "model_repo_id": config.model.repo_id,
            "model_revision": config.model.revision,
            "dtype": config.model.dtype,
            "layers": manifest_layers,
            "base_tensor_count": len(tensors)
            - sum(_layer_tensor_count(layer) for layer in manifest_layers),
        },
        output_dir / "packed_manifest.json",
    )


def _layer_tensor_count(layer: dict[str, Any]) -> int:
    keys = (
        "packed_weight",
        "scale",
        "zero_point",
        "input_scale",
        "protected_indices",
        "protected_weight",
    )
    return sum(1 for key in keys if layer.get(key))
