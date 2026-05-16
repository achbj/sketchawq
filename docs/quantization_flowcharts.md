# Quantization Algorithm Flowcharts

> **Purpose**: A build-it-yourself reference. Every step explains *what* happens and *why*, so you could re-implement any of these methods from scratch.

---

## 1 · GPTQ — Greedy Layer-wise Quantization with Optimal Brain Surgeon

### Step 1 — Load Model & Calibration Data
```
Load FP16/BF16 model (e.g. LLaMA, Qwen)
  • All nn.Linear layers: weight W ∈ ℝ^{O×I}
      O = out_features (rows)   — how many output neurons
      I = in_features  (cols)   — how many input channels the layer reads
Calibration: 128 sequences × 2048 tokens from Pile / wikitext
```

### Step 2 — Collect Hessian via Forward Hooks
```
For each Linear layer, register a forward hook.
Hook fires every forward pass, receives input X ∈ ℝ^{T×I}.

Accumulate:   H += 2 · X^T X      (shape I×I)
After N batches:  H /= N

WHY X^T X?
  The second-order Taylor expansion of the loss around the
  current weight W gives:
      ΔL ≈ ½ Tr(ΔW · H · ΔW^T)
  where H = 2·E[X^T X] is the Gauss-Newton / Fisher approx.
  H tells us which input dimensions MATTER: large H_ii means
  perturbing column i of W causes big output errors.
```

### Step 3 — Damp & Invert the Hessian
```
λ  = 0.01 × mean(diag(H))          ← adaptive damping
H' = H + λI                         ← makes H invertible
H_inv = cholesky_inverse(cholesky(H'))

WHY damping?
  Calibration is finite; H will be rank-deficient.
  Adding λI regularises it. Using λ proportional to mean
  diagonal keeps damping scale-invariant across layers.
```

### Step 4 — Group Columns for Quantization
```
group_size = 128  (typical default)
W is split along the I dimension into chunks of 128 weights.
Each chunk gets its OWN scale (and zero-point for asymmetric).

WHY groups?
  A single per-row scale is dominated by the largest weight,
  wasting dynamic range for all smaller ones. Groups give
  local adaptivity without per-weight overhead.
```

### Step 5 — OBQ Column-Sweep Loop
```
For q = 0 … I-1   (each input channel / weight column):

  a) Quantize column q:
       w_q    = W[:, q]
       scale  = max(|w_q|) / (2^{b-1}-1)   [symmetric]
       w_q^Q  = round(w_q / scale) · scale
       err    = w_q - w_q^Q

  b) Propagate error to columns q+1…I-1:
       W[:, q+1:] -= err ⊗ (H_inv[q, q+1:] / H_inv[q,q])

  WHY this update?
    OBS theory: the optimal weight change to minimize ΔL
    after fixing column q is δW = -err · H_inv[q,:] / H_inv[q,q].
    This redistributes quantization error to remaining columns
    in the direction that least increases the loss.
```

### Step 6 — Output
```
W_int4   : int4 packed weights (uint8, 2 values/byte)
scale    : FP16, shape [O, I/group_size]
zero_pt  : FP16, shape [O, I/group_size]  (asymmetric only)

Inference: W_fp16 ≈ (W_int4 - zero_pt) * scale  per group
```

---

## 2 · AWQ — Activation-Aware Weight Quantization

### Step 1 — Load Model & Data
```
Load FP16 model.
Calibration: ~128 sequences.
Target layers: ALL Linear layers (q/k/v/o_proj, gate/up/down_proj).
```

### Step 2 — Collect Per-Channel Activation Magnitude
```
Hook receives X ∈ ℝ^{T×I}.

mean_abs[i] = (1/T) Σ_t |X[t, i]|    for i = 0…I-1

WHY mean_abs?
  Channels with large activations contribute disproportionately
  to output. Their weights need finer quantization resolution.
```

### Step 3 — Compute Input-Channel Scale (α grid search)
```
For each α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}:

  act   = mean_abs / mean(mean_abs)         normalize
  wcol  = mean_col_abs(W) / mean(...)       per-col weight norm
  s     = act[i]^α / wcol[i]^(1-α)
  s     = clamp(s, 0.25, 4.0)
  s     = s / geomean(s)                    re-center at 1

  α=0 : weight-only scaling
  α=1 : activation-only scaling
  Dividing by weight strength prevents over-scaling channels
  whose weights are already large.
  geomean re-centering keeps the overall scale of W unchanged.
```

### Step 4 — Fake-Quantize & Score
```
W_scaled = W * s.view(1,-1)
W_Q      = quantize_groupwise(W_scaled, bits, group_size)
W_deq    = dequantize(W_Q) / s.view(1,-1)
score    = MSE(W, W_deq)

Pick (α*, clip*) with lowest score.
```

### Step 5 — Apply & Save
```
W_int4 = quantize_groupwise(W * s*, bits, group, clip*)
Store input_scale = s*  (shape [I])

At inference: s* is fused into the previous layer's output
so there is no extra runtime cost.
```

---

## 3 · QuIP — Quantization with Incoherence Processing

### Step 1-2 — Load & Collect Hessian (same as GPTQ)
```
H = 2·E[X^T X] per layer, shape I×I.
```

### Step 3 — Random Orthogonal Pre-rotation (KEY STEP)
```
G_L ∈ ℝ^{O×O}, G_R ∈ ℝ^{I×I}  random Gaussian
U, _ = QR(G_L)   →  orthogonal U
V, _ = QR(G_R)   →  orthogonal V

W_rot = U^T · W · V
H_rot = V^T · H · V

WHY?
  A random orthogonal rotation INCOHERENCES the matrix:
  it spreads energy uniformly. Max entry scales O(1/√n)
  instead of O(1). Uniform quantizers are optimal for
  flat distributions. The rotation is lossless and can
  be folded into adjacent layers at inference.
```

### Step 4 — Quantize Rotated Weight (LDLQ / GPTQ column sweep)
```
Apply GPTQ's OBQ loop to W_rot using H_rot_inv.
No outlier channels → 2-bit quantization is feasible.
```

### Step 5 — Inference
```
x_rot = x @ V
out   = x_rot @ W_Q_deq.T
out   = U @ out
(U, V folded into RMSNorm / adjacent linears at runtime)
```

---

## 4 · QuIP# — QuIP + E8 Lattice Codebook

Identical to QuIP through Step 3. Then:

### Step 4a — E8 Lattice Vector Quantization
```
Group 8 rotated weights as a vector in ℝ^8.
Round to nearest E8 lattice point.

WHY E8?
  Vector quantizers beat scalar quantizers at equal bit-rate.
  E8 is the densest packing in 8D.
  2 bits/weight → codebook of 256 entries per 8 weights.
  Fast nearest-neighbor via integer arithmetic (no lookup table).
  Combined with incoherence rotation, entries are ~uniform → E8 optimal.
```

### Step 4b — LDLQ error propagation (same as GPTQ)
```
Column-sweep OBQ update applied to E8-quantized columns.
```

### Output
```
int2 E8-coded weights  +  scale per 8-weight block
U, V stored as random seeds (saves memory)
```

---

## 5 · SketchAWQ — Our Method

> **Core idea**: AWQ-style equivalent input scaling, scored by a
> Hessian *sketch* (diagonal + low-rank approximation), plus optional
> outlier channel protection and adaptive per-layer configuration.
> No full Hessian inversion. No column-by-column sweep. Fast.

### Step 1 — Load Model & Data
```
AutoModelForCausalLM (FP16)
iter_named_linears(model) → all (name, nn.Linear) pairs

Layers covered (Qwen / LLaMA):
  layers[i].self_attn.q_proj   query   W ∈ ℝ^{H×D}
  layers[i].self_attn.k_proj   key     W ∈ ℝ^{H×D}
  layers[i].self_attn.v_proj   value   W ∈ ℝ^{H×D}
  layers[i].self_attn.o_proj   out     W ∈ ℝ^{D×H}
  layers[i].mlp.gate_proj      gate    W ∈ ℝ^{F×D}
  layers[i].mlp.up_proj        up      W ∈ ℝ^{F×D}
  layers[i].mlp.down_proj      down    W ∈ ℝ^{D×F}

D = hidden dim, H = num_heads×head_dim, F = FFN intermediate dim
```

### Step 2 — Attach Hooks & Run Calibration
```
ActivationObserver.attach(model)
  → register forward hook on every named linear

for batch in calibration_batches:
    observer.set_attention_mask(batch["attention_mask"])
    model(**batch)    ← fires all hooks

Padding tokens excluded: only unmasked positions counted.
```

### Step 3 — Three Statistics Accumulated Per Layer
```
Hook receives X ∈ ℝ^{T×I}  (tokens × input channels)

A) Mean absolute activation (AWQ-style saliency):
   abs_sum += |X|.sum(dim=0)     shape [I]
   count   += T
   mean_abs = abs_sum / count

   WHY: identifies channels with large average activation.

B) Mean squared activation (diagonal Hessian proxy):
   sq_sum += X².sum(dim=0)        shape [I]
   mean_sq = sq_sum / count

   WHY: H_ii = E[X_i²] exactly.  This IS the diagonal of H.
   No I×I matrix needed — just accumulate a length-I vector.

C) Streaming Hessian sketch (low-rank):
   ω ∈ ℝ^{I×r}  fixed random Gaussian, seeded per layer, ÷√r

   X_sub = subsample(X, max_rows_per_batch)
   sketch_sum += X_sub^T @ (X_sub @ ω)    accumulates H·ω

   WHY: Full H = X^T X is O(I²) memory per layer.
   H·ω ∈ ℝ^{I×r} (r≪I) gives enough information to extract
   the top-r eigenvectors without ever forming H explicitly.
```

### Step 4 — Build Hessian Sketch Per Layer
```
build_hessian_sketch(stats, sketch_config):

A) Diagonal:
   diagonal = mean_sq.clamp_min(floor)
   diagonal = diagonal / mean(diagonal)   → normalized to ≈1
   shape [I]

   WHY normalize?  Makes diagonal values comparable across
   layers with different activation scales.

B) Low-rank basis (mode = "diagonal_low_rank"):
   H_ω = sketch_sum / sketch_count   ∈ ℝ^{I×r}
   Q, _ = torch.linalg.qr(H_ω)      Q ∈ ℝ^{I×r}

   WHY QR?  H·ω is in the column space of H.
   QR orthonormalizes it → Q ≈ top-r eigenvectors of H
   without O(I³) cost.

Result: HessianSketch(diagonal=[I], basis=[I×r])
```

### Step 5 — Adaptive Layer Plans (optional)
```
If adaptive.enabled:

  sensitivity_score per layer:
    scores = score_outlier_channels(W, mean_abs, sketch, metric)
    top1%  = topk(scores, ceil(I×0.01))
    sensitivity_score = mean(top1% values)

  Top 25% most sensitive layers:
    → group_size = sensitive_group_size (e.g. 64 instead of 128)
    → finer quantization where it matters most

  outlier_fraction per layer:
    raw = base_fraction × (layer_score / mean_score)
    clamp to [fraction_min, fraction_max]
```

### Step 6 — Select Outlier Channels
```
select_outlier_channels(W, mean_abs, sketch, outlier_config)

Score each input channel i:
  "activation"        score[i] = normalize(mean_abs[i])
  "hessian"           score[i] = normalize(diagonal[i])
  "weight"            score[i] = normalize(mean_col_abs(W[:,i]))
  "activation_weight" score[i] = norm(act[i]) × norm(wcol[i])
  "hessian_weight"    score[i] = norm(diag[i]) × norm(wcol[i])

WHY multiply two signals?
  A channel hurts only if activation AND weight are both large.
  The product catches channels that are outliers in both senses.

k = ceil(I × outlier_fraction)
indices = topk(scores, k)   → k most salient input channels
```

### Step 7 — Joint Search over (α, clip_ratio)
```
For each α ∈ {0.0, 0.25, 0.5, 0.75}:

  ── SCALE ────────────────────────────────────────────────
  act  = mean_abs / mean(mean_abs)
  wcol = mean_col_abs(W) / mean(mean_col_abs(W))
  s    = act^α / wcol^(1-α)
  s    = clamp(s, 0.25, 4.0)
  s    = s / geomean(s)

  ── OUTLIER BOOST (strategy="scale") ────────────────────
  s[outlier_indices] *= scale_boost   (e.g. ×2.0)
  s = s / geomean(s)

  WHY boost?  Larger s[i] → smaller quantization step for
  column i → finer resolution where it matters most.

  For each clip_ratio ∈ {1.0, 0.98, 0.95, 0.9, 0.85, 0.8}:

    ── FAKE QUANTIZE ──────────────────────────────────────
    W_scaled = W * s.view(1,-1)

    Per group g of group_size weights:
      SYMMETRIC:
        max_abs  = max(|W_scaled[:,g]|) × clip_ratio
        scale_g  = max_abs / (2^{b-1} - 1)
        qW       = round(W_scaled[:,g] / scale_g)
        qW       = clamp(qW, -2^{b-1}, 2^{b-1}-1)
        W_deq[:,g] = qW × scale_g / s[g_indices]

      ASYMMETRIC (0-centered range):
        center  = (min + max) / 2
        half    = (max - min) / 2 × clip_ratio
        scale_g = (2 × half) / (2^b - 1)
        zero_g  = round(−min / scale_g)
        qW      = round(W_scaled[:,g] / scale_g + zero_g)
        W_deq[:,g] = (qW − zero_g) × scale_g / s[g]

    WHY clip_ratio < 1?  Trades clipping error on extreme
    values for finer resolution on the majority of weights.

    ── OUTLIER RESTORE (strategy="protect") ──────────────
    W_deq[:,outlier_idx] = W[:,outlier_idx]   (exact FP16)

    ── SCORE CANDIDATE ───────────────────────────────────
    error = W - W_deq

    objective="weighted_error":
      diag_score = mean(error² × diagonal.view(1,-1))
      lr_score   = mean((error @ Q)²)
      score      = diag_score + low_rank_weight × lr_score

      WHY diagonal weight?
        Approximates Tr(ΔW H ΔW^T).  Channels with H_ii large
        damage the output more when their weights are wrong.

      WHY low-rank term?
        Diagonal misses off-diagonal structure.  (error @ Q)²
        measures projection onto H's dominant eigenvectors,
        capturing correlated cross-channel amplification.

    objective="output_error":
      score = mean(((W - W_deq) @ X_sample^T)²)
      Direct output reconstruction on sampled rows.

    objective="hybrid":
      score = h_weight × weighted_score + o_weight × out_score

  Track best (α, clip, score).
```

### Step 8 — Final Quantization
```
W_work = W × scales*.view(1,-1)
quantize_groupwise(W_work, bits, group_size, symmetric, clip*)
  → QuantizedTensor(.qweight, .scale, .zero_pt)

dequantize → W_deq  (FP32, original W space)

apply_outlier_policy:
  "scale"   → no restore (boost already helped)
  "protect" → W_deq[:,outlier_idx] = W[:,outlier_idx]

module.weight.data.copy_(W_deq)   ← in-place replace
```

### Step 9 — Pack to Bits
```
Symmetric: shift int4 from signed [-8,7] to unsigned [0,15]
Pack two 4-bit values per byte (LSB-first)
→ PackedQuantizedTensor (uint8 packed_weight)

PackedLayerArtifact:
  packed_weight      uint8   O×I/2 bytes
  scale              FP32    O×(I/group_size)
  zero_point         FP32    (asymmetric only)
  input_scale        FP32    [I]   AWQ channel scale
  protected_indices  int64   outlier column indices
  protected_weight   FP16    original weights for those cols
```

### Step 10 — Evaluate & Report
```
Perplexity on validation split (WikiText-2 / C4).
Per-layer LayerQuantizationSummary:
  bits, group_size, sketch_mode, sketch_rank
  awq_alpha, clip_ratio, search_score
  outlier_strategy, outlier_fraction, outlier_channels
  compression_ratio, MB saved, elapsed_seconds
```

---

## Side-by-Side Comparison

| Property | GPTQ | AWQ | QuIP | QuIP# | **SketchAWQ** |
|---|---|---|---|---|---|
| Hessian type | Full I×I inverse | None (act proxy) | Full I×I inverse | Full I×I inverse | Diagonal + rank-r sketch |
| Calibration memory | O(I²) / layer | O(I) | O(I²) / layer | O(I²) / layer | O(I·r), r≪I |
| Error correction mechanism | OBQ column sweep | None | OBQ column sweep | OBQ column sweep | Score-guided scale search |
| Weight rotation | No | No | Random orthogonal | Random orthogonal | No |
| Scale / clip search | No | Grid over α | No | No | Joint grid (α, clip) |
| Outlier handling | None | None | Via rotation | Via rotation | Score + scale boost / protect |
| Quantizer | Scalar uniform | Scalar uniform | Scalar uniform | E8 lattice (vector) | Scalar uniform (groupwise) |
| Adaptive per-layer config | No | No | No | No | Yes (group_size + outlier_frac) |
| Speed | Slow (I sweeps) | Fast | Slow | Slow | Fast |
| Best bit-widths | 3–4 | 4 | 2–3 | 2 | 3–4 |
