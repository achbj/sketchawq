# SketchAWQ Simple Architecture

This is the short version of the **current executable SketchAWQ pipeline**.

It intentionally shows what the code does today. It does **not** include the old refinement-holdout / accept-revert stage because that stage is not implemented.

## Main Flow

```mermaid
flowchart TD
    A[Model + tokenizer + YAML config] --> B[Calibration batches]
    A --> C[Evaluation batches]

    B --> D[Calibration forward pass]
    D --> E[Collect per-linear statistics\nmean |X|, mean X^2, X^T X Omega]

    E --> F[Build Hessian sketch\nH_diag + optional low-rank Q]
    F --> G[Optional adaptive layer plan]
    F --> H[Score/select sensitive input channels]
    G --> H

    H --> I[Search AWQ scale alpha\nand clip ratio gamma]
    I --> J[Groupwise low-bit quantization\n2 / 3 / 4 / 8 bit]

    J --> K[Dequantize + undo AWQ input scaling]
    K --> L[Restore protected channels if configured]
    L --> M[Dense quantized model]

    C --> N[Optional original-model evaluation]
    M --> O[Quantized-model evaluation\nPPL / optional throughput]

    J --> P{Save packed model?}
    P -->|yes| Q[Bit-pack qweights to uint8]
    Q --> R[packed_model.safetensors\npacked_manifest.json]
```

## Step by Step

### 1. Load the model and data

`models/load.py` loads the Hugging Face causal LM and tokenizer.

`data/calibration.py` independently creates:

- calibration batches for PTQ statistics/search,
- evaluation batches for perplexity/throughput evaluation.

The number of samples and sequence length are configurable. There is no fixed architectural value such as `128 x 2048`.

### 2. Observe activations

`quant/observer.py` attaches hooks to each `torch.nn.Linear` and collects:

```text
E[|X|]          activation magnitude
E[X^2]          diagonal Hessian signal
X^T (X Omega)   streaming low-rank Hessian signal
```

It can also keep a small sample of activation rows for output-aware search.

### 3. Build the Hessian sketch

`quant/hessian_sketch.py` converts the statistics into:

```text
H_diag
+
optional low-rank basis Q from QR(H Omega)
```

The sketch mode can be disabled, diagonal-only, or diagonal-plus-low-rank.

### 4. Optionally adapt sensitive layers

`quant/adaptive.py` can assign different effective group sizes and outlier fractions to sensitive layers when `adaptive.enabled=true`.

This is optional. Group sizes are config values, not hard-coded to only 64 and 128.

### 5. Select sensitive/outlier channels

`quant/outliers.py` scores input channels using activation, Hessian, weight, or combined signals.

Two strategies exist:

```text
scale    -> boost AWQ scaling for selected channels
protect  -> restore selected columns from the original weight
```

### 6. Search scale and clipping

`quant/awq_scale_search.py` and `quant/clip_search.py` choose the AWQ input scale and clipping ratio.

The code supports:

```text
joint search:
    search all (alpha, gamma) pairs

or

greedy search:
    choose alpha first
    then choose gamma
```

Candidate error can be:

```text
weighted_error
output_error
hybrid
```

The weighted objective uses the diagonal Hessian signal plus the optional low-rank `Q` projection.

### 7. Quantize the weight

`quant/quantizer.py` performs groupwise quantization along the input-channel dimension.

For AWQ scaling the effective operation is:

```text
W
 -> W * s
 -> groupwise low-bit quantization
 -> dequantization
 -> divide by s
```

The dense dequantized result is copied back into the original linear layer.

### 8. Evaluate

`eval/perplexity.py`, `eval/runner.py`, and `eval/throughput.py` evaluate the dense quantized model.

The standard evaluation path measures the **dequantized dense model**, not direct packed INT2/INT3/INT4 kernel execution.

### 9. Optionally pack and export

When `output.save_model=true`:

```text
integer qweights
    -> bit packing
    -> uint8 bytes
    -> packed_model.safetensors
    -> packed_manifest.json
```

`quant/pack.py` performs the packing and `models/save.py` writes the artifacts.

Protected-channel side data is also stored when the `protect` outlier strategy is active.

## What Is Not in the Current Pipeline

The current code does **not** contain:

```text
iterative scale refinement
refinement_guard.py
separate refinement holdout batch
refined-MSE-vs-baseline accept/revert decision
packed low-bit CUDA/Triton inference kernel
```

So the simplest accurate mental model is:

```text
Calibration
   ↓
Activation statistics
   ↓
Hessian sketch
   ↓
Sensitive-channel handling
   ↓
AWQ scale + clip search
   ↓
Groupwise quantization
   ↓
Dense dequantized evaluation
   └──────── optional packed export
```
