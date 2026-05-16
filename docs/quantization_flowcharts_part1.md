# Quantization Algorithm Flowcharts – Part 1: GPTQ & AWQ

---

## GPTQ — Greedy Layer-wise Quantization with Inverse Hessian

### Step 1 — Load Model & Data
```
┌──────────────────────────────────────────────────────┐
│  Load model in FP16/BF16 (HuggingFace AutoModel)    │
│  • All Linear layers: weight W ∈ ℝ^{O×I}            │
│    - O = output channels (rows of W)                 │
│    - I = input channels (columns, also "fan-in")     │
│  Load calibration data: 128 sequences × 2048 tokens │
└──────────────────────────────────────────────────────┘
```

### Step 2 — Collect Hessian Per Layer (forward hooks)
```
┌──────────────────────────────────────────────────────────┐
│  For each Linear layer, register a forward hook.        │
│  Hook fires when the layer is called during forward pass │
│                                                          │
│  On each batch, hook receives X ∈ ℝ^{T×I}              │
│  (T = tokens in this batch, I = input channels)         │
│                                                          │
│  Accumulate:  H += 2 · X^T X   (shape I×I)             │
│  After all batches:  H /= num_batches                   │
│                                                          │
│  WHY X^T X?  The squared output error of a linear       │
│  layer when weight changes by ΔW is:                    │
│    ΔL ≈ Tr(ΔW H ΔW^T) / 2                              │
│  where H = 2 · E[X^T X] is the empirical Hessian.      │
│  This tells us which input dimensions matter MOST.      │
└──────────────────────────────────────────────────────────┘
```

### Step 3 — Invert Hessian with Damping
```
┌──────────────────────────────────────────────────────┐
│  H_inv = (H + λI)^{-1}                              │
│  λ = damping = 0.01 × mean(diag(H))                 │
│                                                      │
│  WHY damping?  H may be rank-deficient (singular).   │
│  Adding λI makes it invertible and numerically       │
│  stable.  λ is proportional to the average diagonal  │
│  so it adapts to the scale of each layer's Hessian.  │
│                                                      │
│  Cholesky decomposition of H_inv is computed once    │
│  and reused for all output rows.                     │
└──────────────────────────────────────────────────────┘
```

### Step 4 — Group Columns into Quantization Groups
```
┌──────────────────────────────────────────────────────┐
│  group_size = 128 (typical)                          │
│  W is split along input dimension into groups of 128 │
│  Each group gets its own scale (and zero if asym.)   │
│                                                      │
│  WHY groups?  A single global scale per row would    │
│  cause large error for outlier weights.  Groups let  │
│  each region of the row adapt its quantization range.│
└──────────────────────────────────────────────────────┘
```

### Step 5 — Column-wise OBQ Quantization Loop
```
┌──────────────────────────────────────────────────────────┐
│  For each column q = 0 … I-1 (ordered by Hessian diag): │
│                                                          │
│  a) Quantize column q of W:                             │
│       w_q    = W[:, q]          (all output rows)        │
│       scale  = max(|w_q|) / (2^{b-1} - 1)  [symmetric] │
│       w_q^Q  = round(w_q / scale) · scale              │
│       error  = w_q - w_q^Q                              │
│                                                          │
│  b) Propagate error to remaining columns q+1…I-1:       │
│       W[:, q+1:] -= error ⊗ (H_inv[q, q+1:] / H_inv[q,q]) │
│                                                          │
│  WHY this update?  It's the closed-form optimal weight   │
│  correction for remaining columns given that column q    │
│  is already fixed (OBQ / OBS theory from 1993).         │
│  It uses the row of H_inv to spread the quantization     │
│  error in the direction that minimally increases loss.   │
└──────────────────────────────────────────────────────────┘
```

### Step 6 — Final Output
```
┌──────────────────────────────────────────────────────┐
│  W_quantized: int4 weights (packed to uint8/int32)  │
│  scale:  FP16 tensor, shape [O, I/group_size]       │
│  zero_point: FP16 tensor (only for asymmetric)      │
│                                                      │
│  Inference:  W_fp16 ≈ dequant(W_quantized, scale)   │
│  Model weights replaced in-place, ready for serving. │
└──────────────────────────────────────────────────────┘
```

---

## AWQ — Activation-Aware Weight Quantization

### Step 1 — Load Model & Calibration Data
```
┌──────────────────────────────────────────────────────┐
│  Load FP16 model                                    │
│  Calibration: ~128 sequences (Pile / wikitext)      │
│  Layers to quantize: ALL Linear layers including    │
│   • q_proj, k_proj, v_proj  (attention queries etc) │
│   • o_proj                  (attention output)      │
│   • gate_proj, up_proj, down_proj (FFN / MLP)       │
└──────────────────────────────────────────────────────┘
```

### Step 2 — Collect Per-Channel Activation Statistics
```
┌──────────────────────────────────────────────────────────┐
│  Hook fires for each linear layer on each batch.        │
│  X ∈ ℝ^{T×I}  (tokens × input channels)               │
│                                                          │
│  mean_abs[i] = (1/T) Σ_t |X[t, i]|   for i=0…I-1      │
│                                                          │
│  This gives one scalar per INPUT CHANNEL representing   │
│  how "active" / "large" that channel's activations are. │
│  Channels with large mean_abs are "salient" – they      │
│  contribute disproportionately to the output and are    │
│  the hardest to quantize accurately.                    │
└──────────────────────────────────────────────────────────┘
```

### Step 3 — Compute Input-Channel Scale (α search)
```
┌──────────────────────────────────────────────────────────────┐
│  AWQ's key insight: scale the WEIGHT columns before         │
│  quantizing so the quantizer step-size adapts to the        │
│  activation magnitude.  The scaling is "equivalent"         │
│  because at inference time you divide activations by the    │
│  same scale, leaving the output unchanged.                  │
│                                                             │
│  For each candidate α ∈ {0, 0.25, 0.5, 0.75, 1.0}:       │
│                                                             │
│    act   = mean_abs / mean(mean_abs)   (normalized)        │
│    wcol  = mean_col_abs(W)             (per-input-col)     │
│    wcol  = wcol / mean(wcol)           (normalized)        │
│                                                             │
│    scale[i] = act[i]^α / wcol[i]^(1-α)                    │
│    scale    = clamp(scale, 0.25, 4.0)                      │
│    scale    = scale / geomean(scale)   (re-normalize)      │
│                                                             │
│  α=0 → scale based only on weight (ignores activations)    │
│  α=1 → scale based only on activations (ignores weights)   │
│  α=0.5 → balanced (typical best)                           │
│                                                             │
│  WHY divide by weight strength?  A channel whose weight    │
│  column is already large doesn't need more boost;          │
│  large weights already land in bigger quantization bins.   │
└──────────────────────────────────────────────────────────────┘
```

### Step 4 — Fake-Quantize and Score Each Candidate
```
┌──────────────────────────────────────────────────────────┐
│  For each (α, clip_ratio) combination:                  │
│                                                         │
│  W_scaled = W * scale.view(1, -1)   (multiply cols)    │
│  W_Q      = quantize_groupwise(W_scaled, bits, group)  │
│  W_deq    = dequantize(W_Q) / scale.view(1, -1)        │
│                                                         │
│  score = MSE(W, W_deq)    (simple reconstruction error) │
│         OR  Hessian-weighted: (W-W_deq)^2 · diag(H)   │
│                                                         │
│  Pick (α*, clip*) that minimizes score.                 │
└──────────────────────────────────────────────────────────┘
```

### Step 5 — Quantize with Best Scale
```
┌──────────────────────────────────────────────────────┐
│  Apply best scale to W, quantize, replace weights.  │
│  Store scale as per-channel multiplier alongside    │
│  the usual per-group quantization scale.            │
│                                                     │
│  At inference:                                      │
│   x_in  = x / scale          (fuse into prev layer) │
│   out   = x_in @ W_Q_deq     (effectively x@W)      │
│  (In practice the input scaling is fused into the   │
│  previous linear layer's output, so no extra ops.)  │
└──────────────────────────────────────────────────────┘
```

### Step 6 — Output
```
┌──────────────────────────────────────────────────────┐
│  int4 weights (packed)                              │
│  per-group scale (FP16)                             │
│  per-channel input scale (FP16, 1 per input dim)   │
│  Model saved in AWQ format, loadable via autoawq.  │
└──────────────────────────────────────────────────────┘
```
