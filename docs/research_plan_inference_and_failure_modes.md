# SketchAWQ Forward Research Plan

Last updated: 2026-05-18

## Research Thesis

SketchAWQ is no longer just a "can we beat GPTQ PPL" project. The current evidence says:

- Quantization is much faster than GPTQ under the fair Qwen3-4B setup.
- PPL is close enough to be scientifically useful, especially at 3-bit.
- The current throughput metric is still dense dequantized simulation, not true packed int3/int4 inference.
- The approximation works empirically, but we do not yet understand where it fails.

The next research question is:

> Where exactly does the SketchAWQ approximation fail, and can the packed representation deliver real inference speed?

This splits the project into two connected tracks.

## Track A - Real Inference

Goal: prove whether SketchAWQ compression can improve real serving speed, not only quantization time or storage.

### A1. Freeze An Honest Inference Benchmark

Add a benchmark that reports separate metrics for:

- prefill tokens/s
- decode tokens/s
- end-to-end generate tokens/s
- time to first token
- peak VRAM
- model load time
- weight runtime format

Required benchmark dimensions:

| Dimension | Default |
| --- | --- |
| Model | Qwen/Qwen3-4B first, then Qwen/Qwen3-8B |
| Prompt lengths | 128, 512, 2048 |
| Generated tokens | 128 |
| Batch sizes | 1, 4, 8 |
| Dtype baseline | FP16/BF16 |
| Quant baselines | GPTQ backend, SketchAWQ dense simulation, SketchAWQ packed/backend |
| Hardware | record exact `gpu_name` every time |

Required output fields:

- `inference_backend`
- `weight_runtime_format`
- `prefill_tokens_per_second`
- `decode_tokens_per_second`
- `generate_tokens_per_second`
- `time_to_first_token_ms`
- `peak_memory_mb`
- `prompt_tokens`
- `generated_tokens`
- `batch_size`
- `gpu_name`

### A2. Separate Evaluation Throughput From Inference Throughput

The current board column `Tokens/s` is evaluation throughput with dense dequantized weights. Keep it, but do not treat it as an inference claim.

New naming rule:

| Metric | Meaning |
| --- | --- |
| `eval_tokens_per_second` | PPL/evaluation loop speed |
| `inference_prefill_tokens_per_second` | serving prefill speed |
| `inference_decode_tokens_per_second` | serving decode speed |
| `inference_generate_tokens_per_second` | full generation speed |

### A3. Short-Term Backend Strategy

Start with 4-bit before 3-bit.

Reason:

- int4 kernels are more common.
- GPTQ/AWQ ecosystems already have int4 packed paths.
- int3 support is much less standard and should not block the first inference result.

Candidate paths:

1. Export SketchAWQ scales and quantized weights into a backend-compatible int4 format.
2. Add a simple packed int4 Linear module for controlled benchmarking.
3. Optimize later with Triton or CUDA only after correctness and speed baselines exist.

### A4. Minimum Acceptance Criteria

A packed inference milestone is only valid if:

- outputs are numerically close to dense simulation for fixed inputs
- PPL matches the dense simulated quantized model within a small tolerance
- the benchmark records prompt length, generated tokens, batch size, backend, and GPU
- speed is compared against FP16 and GPTQ on the same GPU
- dense simulation and packed inference are clearly separated in the report

## Track B - Approximation Failure Modes

Goal: move from empirical success to explanation.

We need to identify:

- architecture sensitivity
- layer sensitivity
- bit-width sensitivity
- calibration sensitivity

### B1. Architecture Sensitivity

Question:

> Which architectures are naturally friendly or hostile to SketchAWQ?

Compare the same fair setup across:

- Qwen3-1.7B
- Qwen3-4B
- Qwen3-8B
- Mistral-7B
- optionally Qwen3-14B and Qwen3-32B after the pipeline is stable

Keep these fixed:

- C4 calibration
- WikiText2 packed validation eval
- group size 128 for GPTQ fairness
- 3-bit and 4-bit
- same sequence length where feasible
- same evaluation protocol

Record:

- PPL gap vs GPTQ
- PPL gap vs FP16
- quantization time ratio vs GPTQ
- compression ratio
- layer score distributions
- search choice distributions

Decision rule:

- If one architecture fails badly, do not tune around it immediately. First compare its layer sensitivity and search-choice plots against an architecture where SketchAWQ works.

### B2. Layer Sensitivity

Question:

> Which layers cause the PPL gap?

Primary diagnostic:

1. Quantize the full model.
2. Restore one layer or block to FP16.
3. Re-evaluate PPL.
4. Rank layers/blocks by PPL rescue.

Variants:

- restore one attention projection family at a time
- restore one MLP projection family at a time
- restore full transformer blocks
- compare early, middle, and late layer sensitivity

Expected outputs:

- `layer_rescue.csv`
- `block_rescue.csv`
- heatmap by layer index and module type
- correlation table between rescue value and existing search metrics

Key correlations to test:

- rescue value vs weighted error score
- rescue value vs output error score
- rescue value vs activation mean absolute value
- rescue value vs Hessian diagonal mass
- rescue value vs low-rank residual
- rescue value vs outlier count
- rescue value vs selected alpha and clip ratio

### B3. Bit-Width Sensitivity

Question:

> Why can 3-bit be very close while 4-bit still lags in some runs?

Compare, for each model:

- 3-bit group 128
- 4-bit group 128
- optional 4-bit group 64 as non-fair "best quality" reference

Record:

- selected alpha distribution
- selected clip distribution
- protected channel count
- search objective score
- PPL gap vs GPTQ
- compression ratio
- quantization time

Decision rule:

- If 4-bit underperforms while 3-bit is close, inspect whether 4-bit search is clipping too aggressively or choosing a narrow alpha band.

### B4. Calibration Sensitivity

Question:

> Is SketchAWQ learning stable activation/Hessian structure, or is it fragile to calibration data?

Run a matrix:

| Axis | Values |
| --- | --- |
| Dataset | C4, WikiText train, mixed C4+WikiText |
| Samples | 128, 256, 512 |
| Sequence length | 1024, 2048 |
| Packing | packed, padded only for ablation |
| Seed | 42, one alternate seed |
| Calibration batch size | 1, 4, 8 where memory allows |

Keep evaluation fixed during sensitivity tests:

- WikiText2 validation packed eval2048 for main comparison
- C4 validation packed eval2048 as confirmation

Decision rule:

- A calibration change is real only if it improves or preserves PPL on a held-out evaluation source, not only the final WikiText board.

## Comparison Plan

### Main Comparison Table

For each model and bit-width:

| Method | Purpose |
| --- | --- |
| FP16/BF16 | quality anchor |
| GPTQ | main fair baseline |
| SketchAWQ dense simulation | current quality/compression method |
| SketchAWQ packed/backend | real inference method, once available |
| AWQ or LLMCompressor AWQ | secondary baseline after GPTQ |

### Required Views

1. Paper PPL board
   - lower PPL is better
   - include quantization time, GPU, compression

2. Inference board
   - higher prefill/decode/generate tokens/s is better
   - include backend and runtime format

3. Failure-mode board
   - layer/block rescue
   - sensitivity heatmaps
   - calibration matrix

## Immediate Next Steps

### Step 1 - Add Inference Benchmark Skeleton

- [x] Add a benchmark command or script that can run FP16 and dense simulated SketchAWQ first.
- [x] Record prompt length, generated tokens, batch size, GPU, and runtime format.
- [ ] Run the first A100 Qwen3-4B FP16 and SketchAWQ dense-simulation matrix.
- [ ] Add GPTQ backend benchmarking once saved/loadable GPTQ model handling is settled.
- [ ] Do not claim packed speed yet.

Canonical command shape:

```bash
sketchawq inference configs/paper/qwen3_4b_4bit.yaml \
  --variant fp16 \
  --prompt-lengths 128 512 2048 \
  --batch-sizes 1 4 \
  --generated-tokens 128 \
  --output-root artifacts/inference
```

For dense simulated SketchAWQ, use:

```bash
sketchawq inference configs/paper/qwen3_4b_4bit.yaml \
  --variant sketchawq-dense \
  --prompt-lengths 128 512 2048 \
  --batch-sizes 1 4 \
  --generated-tokens 128 \
  --output-root artifacts/inference
```

### Step 2 - Add Experiment Ledger Discipline

- Every run gets a complete entry in `docs/experiment_ledger.md`.
- The entry must include config path, artifact path, git state, GPU, calibration setup, evaluation setup, timing, PPL, and decision.

### Step 3 - Add Layer Rescue Tooling

- Start with Qwen3-4B 4-bit because the gap is visible but the model is manageable.
- Restore one block at a time to FP16 and run the same evaluation.
- Produce `block_rescue.csv`.

### Step 4 - Decide Backend Direction

- Investigate whether SketchAWQ int4 artifacts can be mapped into an existing GPTQ/AWQ runtime format.
- If yes, implement export/conversion.
- If no, implement a minimal packed int4 Linear benchmark path.

### Step 5 - Compare Again

Use this decision table:

| Outcome | Decision |
| --- | --- |
| Packed SketchAWQ is faster and PPL holds | inference track becomes the main paper contribution |
| Packed SketchAWQ is not faster | keep speed claim limited to quantization time and compression |
| Layer rescue reveals a small set of bad layers | add targeted protection or mixed precision |
| Failures are diffuse across many layers | improve objective/sketch rather than adding layer exceptions |
| Calibration changes dominate PPL | study stability before adding new search knobs |

## Non-Goals For Now

- Do not chase a tiny GPTQ PPL win by adding many model-specific knobs.
- Do not report dense simulation throughput as inference speed.
- Do not tune Qwen3-4B only and call it a general result.
- Do not mix hardware when making close timing claims.
- Do not promote any recipe without a held-out robustness check.
