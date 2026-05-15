# Architecture

SketchAWQ is split into a reusable package and a thin experiment layer.

## Package Flow

1. `sketchawq.models.load` loads a Hugging Face causal LM and tokenizer.
2. `sketchawq.data.calibration` builds calibration/evaluation batches.
3. `sketchawq.quant.observer` collects per-linear activation statistics.
4. `sketchawq.quant.hessian_sketch` builds diagonal or diagonal-plus-low-rank sketches.
5. `sketchawq.quant.awq_scale_search` searches AWQ input scales and clipping ratios.
6. `sketchawq.quant.pipeline` applies quantization and emits packed layer artifacts.
7. `sketchawq.models.save` writes reports plus `packed_model.safetensors` and `packed_manifest.json`.

## Current Runtime Boundary

The packed format is a storage/export format. Evaluation still runs normal dense linear layers after dequantization. Real inference speedups need a `SketchAWQLinear` runtime with CUDA/Triton kernels or integration with an existing serving stack.

## Experiment Layer

`configs/experiments.yaml` is now the single list of named targets. Keep the catalog short:

- smoke checks that run quickly
- canonical paper rows
- active baseline comparisons

Large ablation grids belong in the old archive or in generated experiment outputs, not in the main working tree.
