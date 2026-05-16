## SketchAWQ — Our Method

> **Core idea**: AWQ's equivalent input scaling + a Hessian _sketch_
> (diagonal + low-rank) to score candidates, plus optional outlier
> channel protection and adaptive per-layer configuration.

### Step 1 — Load Model & Data

```
┌────────────────────────────────────────────────────────────────┐
│  Load FP16 model (AutoModelForCausalLM)                       │
│  Collect calibration batches: e.g. 128 seqs × 512 tokens     │
│  iter_named_linears(model) → list of (name, nn.Linear)        │
│                                                               │
│  Named linears cover ALL projection layers:                   │
│   model.layers[i].self_attn.q_proj   ← query projection      │
│   model.layers[i].self_attn.k_proj   ← key projection        │
│   model.layers[i].self_attn.v_proj   ← value projection      │
│   model.layers[i].self_attn.o_proj   ← attention output      │
│   model.layers[i].mlp.gate_proj      ← FFN gate (SwiGLU)     │
│   model.layers[i].mlp.up_proj        ← FFN up-projection     │
│   model.layers[i].mlp.down_proj      ← FFN down-projection   │
│                                                               │
│  Each has weight W ∈ ℝ^{O×I}                                 │
│   O = out_features (how many neurons this layer writes to)    │
│   I = in_features  (how many input dimensions this layer reads│
└────────────────────────────────────────────────────────────────┘
```

### Step 2 — Register Hooks & Run Calibration Forward Passes

```
┌────────────────────────────────────────────────────────────────┐
│  ActivationObserver.attach(model):                            │
│   For every named linear layer, register a forward hook.      │
│   The hook fires BEFORE the linear computation, receiving X.  │
│                                                               │
│  For each calibration batch:                                  │
│   model(**batch)  → triggers all hooks                        │
│                                                               │
│  Each hook receives X ∈ ℝ^{T×I}  (tokens × input channels)   │
│  (possibly multi-dimensional batch that is flattened to 2D)   │
│                                                               │
│  Attention mask is applied: padding tokens are dropped so     │
│  their activations don't pollute statistics.                  │
└────────────────────────────────────────────────────────────────┘
```

### Step 3 — Accumulate Three Statistics in the Hook

```
┌────────────────────────────────────────────────────────────────┐
│  For each batch's flat activation matrix X ∈ ℝ^{T×I}:        │
│                                                               │
│  A) Mean absolute activation (AWQ-style saliency):           │
│     abs_sum += |X|.sum(dim=0)    → shape [I]                 │
│     count   += T                                             │
│     → mean_abs = abs_sum / count  (computed later)           │
│                                                               │
│  B) Mean squared activation (diagonal Hessian proxy):        │
│     sq_sum  += X².sum(dim=0)     → shape [I]                 │
│     → mean_square = sq_sum / count  (used for diag of H)     │
│                                                               │
│  C) Streaming Hessian sketch (low-rank approximation):       │
│     ω ∈ ℝ^{I×r}  random Gaussian test matrix (fixed seed)   │
│     X_sub = random_sample(X, max_rows_per_batch)             │
│     projected = X_sub @ ω          → shape [T', r]           │
│     update    = X_sub^T @ projected → shape [I×I] reduced    │
│     sketch_sum += update  (accumulates H·ω across batches)   │
│                                                               │
│  WHY sketch?  Computing full H = X^TX is O(I²) memory.       │
│  Instead we accumulate H·ω (shape I×r, r≪I) without ever    │
│  forming the full I×I Hessian. From H·ω we can extract       │
│  the top-r eigenvectors via QR decomposition.                │
└────────────────────────────────────────────────────────────────┘
```

### Step 4 — Build Hessian Sketch Per Layer

```
┌────────────────────────────────────────────────────────────────┐
│  build_hessian_sketch(stats, sketch_config):                  │
│                                                               │
│  A) Diagonal term:                                            │
│     diagonal = mean_square.clamp_min(floor)                  │
│     diagonal = diagonal / mean(diagonal)   (normalize to ~1) │
│     Shape: [I]                                               │
│                                                               │
│     WHY mean_square as diagonal?                             │
│     H_ii = E[X_i²] — the diagonal of the Hessian equals the │
│     mean squared activation for input channel i.             │
│     Channels with large H_ii cause large output errors when  │
│     their weight values are perturbed by quantization.       │
│                                                               │
│  B) Low-rank basis (optional, rank r):                        │
│     H_ω = sketch_sum / sketch_count   → H·ω ∈ ℝ^{I×r}       │
│     Q, _ = QR(H_ω)                   → Q ∈ ℝ^{I×r}          │
│     (orthonormal columns spanning top-r subspace of H)       │
│                                                               │
│     WHY QR?  H·ω is already in the column-space of H.        │
│     QR gives an orthonormal basis for this subspace,         │
│     which is a good approximation of the top-r eigenvectors  │
│     of H without computing an expensive full eigendecomp.    │
│                                                               │
│  Result: HessianSketch(diagonal=[I], basis=[I×r])            │
└────────────────────────────────────────────────────────────────┘
```

### Step 5 — Adaptive Layer Plans (optional)

```
┌────────────────────────────────────────────────────────────────┐
│  If adaptive.enabled:                                         │
│                                                               │
│  For each layer compute sensitivity_score:                    │
│   scores = score_outlier_channels(W, mean_abs, sketch, metric)│
│   top_k  = top 1% of scores (most sensitive channels)        │
│   sensitivity_score = mean(top_k scores)                     │
│                                                               │
│  Rank all layers by sensitivity_score.                        │
│                                                               │
│  Top group_size_top_fraction (e.g. 25%) of layers:           │
│   → use sensitive_group_size (e.g. 64 instead of 128)        │
│   → smaller group = finer-grained quantization = less error  │
│                                                               │
│  outlier_fraction per layer:                                  │
│   raw = base_fraction × (layer_score / mean_score)           │
│   clamp to [protected_fraction_min, protected_fraction_max]  │
│   → sensitive layers protect more outlier channels           │
└────────────────────────────────────────────────────────────────┘
```

### Step 6 — Select Outlier Channels Per Layer

```
┌────────────────────────────────────────────────────────────────┐
│  select_outlier_channels(W, mean_abs, sketch, outlier_config) │
│                                                               │
│  Score every input channel i using chosen metric:            │
│                                                               │
│  metric="activation":       score[i] = mean_abs[i]           │
│  metric="hessian":          score[i] = diagonal[i]           │
│  metric="weight":           score[i] = mean_col_abs(W[:,i])  │
│  metric="activation_weight":score[i] = norm(act) × norm(wgt) │
│  metric="hessian_weight":   score[i] = norm(diag) × norm(wgt)│
│                                                               │
│  WHY multiply hessian × weight?  A channel is a problem only │
│  if BOTH its activation is large (H_ii is big) AND its       │
│  weight column has large values (the two amplify each other). │
│                                                               │
│  k = ceil(I × outlier_fraction)                              │
│  indices = topk(scores, k)   → top-k most salient channels   │
│                                                               │
│  These k channels are "outliers" that need special treatment. │
└────────────────────────────────────────────────────────────────┘
```

### Step 7 — Search Best AWQ Scale (α) and Clip Ratio (joint search)

```
┌────────────────────────────────────────────────────────────────┐
│  For each α ∈ {0.0, 0.25, 0.5, 0.75}:                       │
│                                                               │
│   SCALE COMPUTATION:                                          │
│    act   = mean_abs / mean(mean_abs)    (normalize)          │
│    wcol  = mean_col_abs(W) / mean(mean_col_abs(W))           │
│    raw_s = act^α / wcol^(1-α)          (per-channel scale)   │
│    s     = clamp(raw_s, min=0.25, max=4.0)                   │
│    s     = s / geomean(s)              (re-center at 1.0)    │
│                                                               │
│    WHY geomean normalization?  Keeps the overall scale of W  │
│    unchanged; only the relative balance between channels      │
│    shifts.  Prevents the entire weight from shrinking.       │
│                                                               │
│   OUTLIER SCALE BOOST (strategy="scale"):                    │
│    For each outlier channel i in selection.indices:          │
│     s[i] *= scale_boost  (e.g. 2.0)                         │
│    Then re-normalize via geomean.                            │
│    WHY?  A larger input scale means the quantizer step-size  │
│    for that column is smaller → finer resolution for the     │
│    channels that matter most.                               │
│                                                               │
│   For each clip_ratio ∈ {1.0, 0.98, 0.95, 0.9, 0.85, 0.8}: │
│                                                               │
│    FAKE QUANTIZE:                                            │
│     W_scaled = W * s.view(1,-1)   (multiply each column by s)│
│     quantize_groupwise(W_scaled, bits, group_size, clip)     │
│      → within each group of `group_size` weights:           │
│        max_abs = max(|W_scaled[:,g]|) × clip_ratio           │
│        scale_g = max_abs / (2^{b-1} - 1)   [symmetric]      │
│        OR for asymmetric:                                    │
│          center  = (min+max)/2                               │
│          half    = (max-min)/2 × clip_ratio                  │
│          scale_g = (2×half) / (2^b - 1)                     │
│          zero_pt = round(-min/scale_g)                       │
│        qweight = round(W_scaled / scale_g)  clamped to int  │
│     dequant: W_deq = qweight × scale_g / s.view(1,-1)       │
│                                                               │
│    OUTLIER PROTECTION (strategy="protect"):                  │
│     W_deq[:,outlier_indices] = W[:,outlier_indices]  (exact) │
│                                                               │
│    SCORE THE CANDIDATE:                                       │
│     error = W - W_deq                                        │
│                                                               │
│     objective="weighted_error":                              │
│      diag_score = mean( error² × diagonal.view(1,-1) )      │
│      low_rank_score = mean( (error @ Q)² )                   │
│      score = diag_score + low_rank_weight × low_rank_score  │
│                                                               │
│      WHY weighted by diagonal?  error²·H_ii approximates     │
│      the true output error ΔL ≈ Tr(ΔW·H·ΔW^T).  Channels   │
│      with high H_ii contribute more to the loss, so their   │
│      weight errors should be penalized more heavily.         │
│                                                               │
│      WHY low-rank term?  The diagonal only captures H's      │
│      diagonal.  The low-rank term (error @ Q)² captures      │
│      the Hessian's dominant off-diagonal structure: how      │
│      correlated input channels amplify each other's errors.  │
│                                                               │
│     objective="output_error":                                │
│      score = mean( ((W-W_deq) @ X_sample^T)² )              │
│      Directly measures output activation reconstruction error│
│      on sampled calibration rows X_sample ∈ ℝ^{T'×I}.      │
│                                                               │
│     objective="hybrid":                                      │
│      score = h_weight × weighted_score + o_weight × out_score│
│                                                               │
│  Pick (α*, clip*) → lowest score → best_result              │
└────────────────────────────────────────────────────────────────┘
```

### Step 8 — Final Quantization with Best Parameters

```
┌────────────────────────────────────────────────────────────────┐
│  quantize_weight_with_input_scale(W, bits, group_size,        │
│                                   symmetric, clip*, scales*)  │
│   W_work = W × scales*.view(1,-1)                            │
│   quantize_groupwise(W_work, …) → QuantizedTensor:            │
│    .qweight  = int4 integers  (shape [O, I])                  │
│    .scale    = FP32 per-group scale  (shape [O, I/group_size])│
│    .zero_pt  = FP32 per-group zero   (asymmetric only)        │
│                                                               │
│  dequantize → W_deq (FP32, back in original W space)         │
│                                                               │
│  apply_outlier_policy(W, W_deq, selection, config):          │
│   "scale"  → no restoration, the boosted scale already helped │
│   "protect"→ W_deq[:,outlier_idx] = W[:,outlier_idx]  (FP16) │
└────────────────────────────────────────────────────────────────┘
```

### Step 9 — Pack & Store

```
┌────────────────────────────────────────────────────────────────┐
│  pack_quantized_tensor(QuantizedTensor):                      │
│   Shift signed int4 → unsigned (0…15)                        │
│   Pack two 4-bit values per byte using bit manipulation      │
│   → PackedQuantizedTensor with uint8 packed_weight           │
│                                                               │
│  PackedLayerArtifact per layer:                              │
│   .packed_weight  uint8 tensor  ~O×I/2 bytes (4-bit)        │
│   .scale          FP32 tensor   O×(I/group_size)             │
│   .zero_point     FP32 tensor   (asymmetric only)            │
│   .input_scale    FP32 tensor   shape [I]  (AWQ channel scale)│
│   .protected_indices  int64     indices of FP16 columns       │
│   .protected_weight   FP16      original weights for those   │
│                                                               │
│  module.weight.data.copy_(W_deq)   (in-place weight update)  │
│  → model now has fake-quantized FP16 weights for eval        │
└────────────────────────────────────────────────────────────────┘
```

### Step 10 — Evaluate & Report

```
┌────────────────────────────────────────────────────────────────┐
│  Run perplexity evaluation on validation split.               │
│  Report per-layer stats:                                      │
│   • bits, group_size, sketch_mode, sketch_rank               │
│   • alpha, clip_ratio, search_score                          │
│   • outlier_strategy, outlier_fraction, outlier_channels      │
│   • compression ratio, MB saved                              │
│  Save packed artifacts for deployment.                       │
└────────────────────────────────────────────────────────────────┘
```

---

## Side-by-Side Comparison

| Property               | GPTQ              | AWQ                     | QuIP              | QuIP#                          | **SketchAWQ**                   |
| ---------------------- | ----------------- | ----------------------- | ----------------- | ------------------------------ | ------------------------------- |
| Hessian type           | Full I×I inverse  | None (activation proxy) | Full I×I inverse  | Full I×I inverse               | Diagonal + low-rank sketch      |
| Calibration data use   | Build H           | Build mean_abs          | Build H           | Build H                        | Build mean_abs + sketch         |
| Error propagation      | OBQ column sweep  | No                      | OBQ column sweep  | OBQ column sweep               | No (AWQ-style equivalent scale) |
| Weight rotation        | No                | No                      | Random orthogonal | Random orthogonal + E8 lattice | No                              |
| Scale search           | No                | Grid search over α      | No                | No                             | Joint grid search (α, clip)     |
| Outlier handling       | None (basic)      | None (basic)            | Via rotation      | Via rotation                   | Explicit score + scale/protect  |
| Quantizer              | Scalar uniform    | Scalar uniform          | Scalar uniform    | E8 lattice vector              | Scalar uniform (groupwise)      |
| Memory for calibration | O(I²) per layer   | O(I)                    | O(I²) per layer   | O(I²) per layer                | O(I·r) sketch, r≪I              |
| Typical speed          | Slow (per-column) | Fast                    | Slow              | Slow                           | Fast (no column sweep)          |
| Bit-widths             | 2–8               | 4                       | 2–4               | 2 (best)                       | 2–8                             |
