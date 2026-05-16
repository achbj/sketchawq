# Quantization Algorithm Flowcharts – Part 2: QuIP, QuIP#, SketchAWQ

---

## QuIP — Quantization with Incoherence Processing

### Step 1 — Load Model

```
┌──────────────────────────────────────────────────────┐
│  Load FP16/BF16 model                               │
│  Calibration: ~256 sequences (RedPajama / wikitext) │
│  Target: aggressive 2-bit or 3-bit quantization     │
└──────────────────────────────────────────────────────┘
```

### Step 2 — Compute Hessian (same as GPTQ)

```
┌──────────────────────────────────────────────────────┐
│  H = 2 · E[X^T X] per layer  (shape I×I)           │
│  Same forward-hook accumulation as GPTQ.            │
└──────────────────────────────────────────────────────┘
```

### Step 3 — Incoherence Pre-processing (the key idea)

```
┌──────────────────────────────────────────────────────────────┐
│  PROBLEM: weight matrices have "outlier" rows/cols that      │
│  cluster large values, making uniform quantization bad.      │
│                                                              │
│  QuIP's SOLUTION: apply random orthogonal rotation to       │
│  spread ("incohere") the weight energy uniformly.           │
│                                                              │
│  Compute random orthogonal matrix U ∈ ℝ^{O×O} and          │
│                               V ∈ ℝ^{I×I}                  │
│  (from random Gaussian matrix via QR decomposition)         │
│                                                              │
│  W_rotated = U^T · W · V                                    │
│                                                              │
│  WHY?  After rotation, no single weight is dominant.        │
│  The max element scales as O(1/√n) rather than O(1).        │
│  Uniform quantizers work near-optimally on such "flat"      │
│  distributions.  The rotation is orthogonal so it is        │
│  equivalent (lossless) — it can be folded into the          │
│  adjacent layers at inference time.                         │
│                                                              │
│  The Hessian is also rotated: H_rot = V^T · H · V           │
└──────────────────────────────────────────────────────────────┘
```

### Step 4 — Quantize Rotated Weight (LDLQ / GPTQ-style)

```
┌──────────────────────────────────────────────────────────────┐
│  Quantize W_rotated using LDLQ (like GPTQ column sweep)     │
│  on the rotated Hessian H_rot.                              │
│                                                              │
│  Same OBQ error-propagation loop as GPTQ but applied        │
│  to the pre-rotated, incoherent weight matrix.              │
│  Because the energy is spread out, 2-bit quantization       │
│  becomes feasible without catastrophic error.               │
└──────────────────────────────────────────────────────────────┘
```

### Step 5 — Store & Inference

```
┌──────────────────────────────────────────────────────────────┐
│  Store: int2/int3 packed weights + rotation matrices U, V   │
│                                                              │
│  Inference: apply V to input, matmul with W_Q, apply U.     │
│  U and V can be fused into neighboring layers (e.g. the     │
│  RMS-norm preceding the linear layer) to eliminate          │
│  extra matrix multiplications at runtime.                   │
└──────────────────────────────────────────────────────────────┘
```

---

## QuIP# — QuIP with Lattice Codebooks

### Same as QuIP but replaces Step 4 with a lattice codebook quantizer.

### Step 4a — E8 Lattice Codebook Quantization

```
┌──────────────────────────────────────────────────────────────┐
│  Instead of scalar uniform quantization, QuIP# uses the     │
│  E8 lattice: the densest known packing in 8 dimensions.     │
│                                                              │
│  Weight groups of 8 scalars are treated as a vector in ℝ^8. │
│  That vector is rounded to the nearest E8 lattice point.    │
│                                                              │
│  WHY E8?  Vector quantizers achieve lower distortion than    │
│  scalar quantizers at the same bit-rate.  The E8 lattice    │
│  has 2^{16} / 8 = 256 codebook entries per 8 values = 2 bits │
│  per weight, matching the incoherence pre-processing scale. │
│                                                              │
│  Combined with the incoherence rotation from QuIP, this     │
│  gives state-of-the-art 2-bit accuracy on large LLMs.      │
│                                                              │
│  Nearest E8 neighbor is found by a fast integer algorithm   │
│  (Micciancio & Nicolosi 2008), not by brute-force search.  │
└──────────────────────────────────────────────────────────────┘
```

### Step 4b — Hessian-Weighted LDLQ Correction

```
┌──────────────────────────────────────────────────────────────┐
│  Identical OBQ error-propagation loop as GPTQ but using     │
│  the E8-quantized columns.  Each column's quantization      │
│  error is propagated to remaining columns via H_inv.        │
└──────────────────────────────────────────────────────────────┘
```

### Output

```
┌──────────────────────────────────────────────────────────────┐
│  int2 E8-coded weights (packed)                            │
│  Scale per 8-weight group                                  │
│  Orthogonal rotation matrices U, V (stored as Hadamard or  │
│  random matrices using a seed to save memory)              │
└──────────────────────────────────────────────────────────────┘
```

---
