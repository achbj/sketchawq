# SketchAWQ Experiment Ledger

Last updated: 2026-05-19

This is the detailed experiment record. The goal is to preserve every important run decision, not just the final pretty board.

Rule: if a result affects our interpretation, it gets logged here with enough detail to reproduce or reject it later.

## Required Entry Schema

Every new experiment entry should include:

- Date and local time.
- Short experiment name.
- Status: planned, running, done, rejected, or frozen.
- Git commit or branch, plus whether the worktree was dirty.
- Command or script used.
- Config paths.
- Artifact paths.
- Model repo, revision, dtype, trust-remote-code flag.
- Method and backend.
- Bits, group size, symmetric/asymmetric setting.
- Search objective, alpha candidates, clip candidates, scale range.
- Sketch mode, rank, low-rank weight, row sampling, max rows per batch.
- Outlier policy and protected channel settings.
- Calibration dataset, split, text column, packing, sequence length, samples, batch size, seed.
- Evaluation dataset, split, packing, sequence length, samples, max batches, batch size.
- GPU name, CUDA availability, peak memory if recorded.
- Quantization time, stats time, stats forward time, layer time, stats batches, stats tokens.
- FP16 loss/PPL, quantized loss/PPL, PPL delta and ratio.
- Evaluation throughput, with a label saying whether it is dense simulation or packed inference.
- Model size, quantized size, compression ratio.
- Search choice plots or relevant plot paths.
- Decision and next action.

## Current Interpretation

As of 2026-05-18:

- SketchAWQ has a strong quantization-time advantage over GPTQ on Qwen3-4B.
- The latest A100 batch-8 run reduces stats forwards from 256 to 32 and cuts quantization time to about 58s.
- PPL remains close but not always better than GPTQ.
- The `Tokens/s` values in paper boards are dense dequantized simulation throughput, not real packed inference.
- Next work should split into real inference benchmarking and approximation failure diagnostics.

## Experiment Timeline

### 2026-05-17 16:32:59 - Qwen3-4B Fair Balanced Baseline

Status:
- Historical baseline.

Artifact:
- `artifacts/modal/qwen3_4b_3_4bit_paper_eval_20260517_163259`

Purpose:
- Establish Qwen3-4B 3-bit/4-bit fair comparison against GPTQ with group size 128.

Shared setup:
- Model: `Qwen/Qwen3-4B`
- Calibration: C4 train, packed, 256 samples, sequence length 2048
- Evaluation: WikiText2 validation packed eval2048
- Runtime format for throughput: dense dequantized simulation for SketchAWQ

Rows:

| Bits | Method | GPU | PPL | FP16 PPL | Quant time | Stats time | Layer time | Eval tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | SketchAWQ | NVIDIA A10 | 17.7010 | 15.3766 | 231.48s | 208.62s | 22.17s | 5419.2 |
| 3 | GPTQ | NVIDIA A10 | 17.8771 | 15.3766 | 1012.91s | n/a | n/a | 5147.0 |
| 4 | SketchAWQ | NVIDIA A10 | 15.8668 | 15.3766 | 208.41s | 184.37s | 23.59s | 5394.1 |
| 4 | GPTQ | NVIDIA A10G | 15.6277 | 15.3760 | 1095.40s | n/a | n/a | 5165.4 |

Decision:
- SketchAWQ was much faster to quantize.
- 3-bit was better than GPTQ in this specific board.
- 4-bit was behind GPTQ in PPL.
- Stats collection dominated SketchAWQ runtime.

Next action:
- Investigate stats collection overhead.

### 2026-05-17 17:13:52 - Slower Repeat / Hardware Variation

Status:
- Historical timing comparison.

Artifact:
- `artifacts/modal/qwen3_4b_3_4bit_paper_eval_20260517_171352`

Rows:

| Bits | Method | GPU | PPL | FP16 PPL | Quant time | Stats time | Layer time | Eval tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | SketchAWQ | NVIDIA A10G | 17.7184 | 15.3760 | 276.51s | 251.97s | 23.82s | 5122.6 |
| 3 | GPTQ | NVIDIA A10 | 17.8771 | 15.3766 | 1022.08s | n/a | n/a | 5046.8 |
| 4 | SketchAWQ | NVIDIA A10G | 16.0137 | 15.3760 | 237.41s | 212.85s | 24.10s | 5126.6 |
| 4 | GPTQ | NVIDIA A10G | 15.6277 | 15.3760 | 1091.10s | n/a | n/a | 5167.1 |

Decision:
- Runtime increased materially.
- GPU class and stats collection needed to be logged explicitly.

Next action:
- Add/report GPU name and attack stats collection sync overhead.

### 2026-05-17 19:22:25 - GPU-Resident Stats Accumulators

Status:
- Accepted with caution.

Artifact:
- `artifacts/modal/qwen3_4b_3_4bit_paper_eval_20260517_192225`

Change:
- Keep `LayerStats` accumulators on the model device during calibration.
- Use `float32` accumulators.
- Move stats to CPU once after collection.

Rows:

| Bits | Method | GPU | PPL | FP16 PPL | Quant time | Stats time | Layer time | Eval tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | SketchAWQ | NVIDIA A10G | 17.7015 | 15.3760 | 188.70s | 165.42s | 22.48s | 5128.1 |
| 3 | GPTQ | NVIDIA A10G | 17.6530 | 15.3760 | 1080.35s | n/a | n/a | 5059.7 |
| 4 | SketchAWQ | NVIDIA A10 | 16.0346 | 15.3766 | 165.70s | 143.34s | 21.87s | 5537.2 |
| 4 | GPTQ | NVIDIA A10 | 15.8146 | 15.3766 | 1010.34s | n/a | n/a | 5297.5 |

Decision:
- Stats time improved about 20-22% compared with the old balanced run.
- Not enough to explain the full runtime gap.
- 4-bit PPL was confounded by search-grid changes.

Next action:
- Keep GPU-resident stats accumulators.
- Do not over-interpret quality from this mixed change.

### 2026-05-17 20:33:58 - Tensor-Score Search Experiment

Status:
- Rejected.

Artifact:
- `artifacts/modal/qwen3_4b_3_4bit_paper_eval_20260517_203358`

Change:
- Tried keeping candidate scores on GPU during search and syncing less often.

Rows:

| Bits | Method | GPU | PPL | FP16 PPL | Quant time | Stats time | Layer time | Eval tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | SketchAWQ | NVIDIA A10G | 17.7015 | 15.3760 | 200.57s | 174.82s | 24.93s | 5120.6 |
| 3 | GPTQ | NVIDIA A10 | 17.8771 | 15.3766 | 1111.93s | n/a | n/a | 4855.4 |
| 4 | SketchAWQ | NVIDIA A10G | 16.0291 | 15.3760 | 187.14s | 165.02s | 21.55s | 5121.5 |
| 4 | GPTQ | NVIDIA A10 | 15.8146 | 15.3766 | 1023.15s | n/a | n/a | 5177.1 |

Decision:
- Did not improve runtime.
- Increased complexity without a clear benefit.

Next action:
- Revert tensor-score/GPU-cache experiment.
- Focus on calibration batching and stats forward path.

### 2026-05-17 20:59:33 - Post-Revert Timing Repeat

Status:
- Historical repeat.

Artifact:
- `artifacts/modal/qwen3_4b_3_4bit_paper_eval_20260517_205933`

Rows:

| Bits | Method | GPU | PPL | FP16 PPL | Quant time | Stats time | Layer time | Eval tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | SketchAWQ | NVIDIA A10G | 17.7015 | 15.3760 | 188.47s | 165.43s | 22.22s | 5130.9 |
| 3 | GPTQ | NVIDIA A10 | 17.8771 | 15.3766 | 1111.93s | n/a | n/a | 4855.4 |
| 4 | SketchAWQ | NVIDIA A10G | 16.0291 | 15.3760 | 188.17s | 165.71s | 21.88s | 5121.8 |
| 4 | GPTQ | NVIDIA A10 | 15.8146 | 15.3766 | 1023.15s | n/a | n/a | 5177.1 |

Decision:
- Confirms layer search was not the main bottleneck.
- Stats still dominated runtime.

Next action:
- Add stats-phase instrumentation.

### 2026-05-17 22:05:20 - Stats-Phase Cleanup

Status:
- Accepted.

Artifact:
- `artifacts/modal/qwen3_4b_3_4bit_paper_eval_20260517_220520`

Change:
- Use `torch.inference_mode()`.
- Pass `use_cache=False` when supported.
- Skip attention-mask filtering when packed calibration has all-ones masks.

Rows:

| Bits | Method | GPU | PPL | FP16 PPL | Quant time | Stats time | Layer time | Eval tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | SketchAWQ | NVIDIA A10 | 17.6992 | 15.3766 | 170.71s | 147.46s | 22.49s | 5398.0 |
| 3 | GPTQ | NVIDIA A10G | 17.6530 | 15.3760 | 1131.47s | n/a | n/a | 5070.5 |
| 4 | SketchAWQ | NVIDIA A10G | 16.0291 | 15.3760 | 188.67s | 165.87s | 22.22s | 5129.2 |
| 4 | GPTQ | NVIDIA A10 | 15.8146 | 15.3766 | 1022.01s | n/a | n/a | 5323.9 |

Decision:
- Low-risk cleanup was kept.
- GPU variation still made close timing comparisons noisy.

Next action:
- Run on A100 to separate hardware ceiling from algorithm cost.

### 2026-05-17 22:54:27 - Instrumented A10G Reference

Status:
- Reference run.

Artifact:
- `artifacts/modal/qwen3_4b_3_4bit_paper_eval_20260517_225427`

Rows:

| Bits | Method | GPU | PPL | Quant time | Stats time | Stats forward | Stats batches | Stats tokens | Layer time | Eval tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | SketchAWQ | NVIDIA A10G | 17.7015 | 173.35s | 147.57s | 146.37s | 256 | 524288 | 24.77s | 5121.6 |
| 3 | GPTQ | NVIDIA A10G | 17.6530 | 1077.72s | n/a | n/a | n/a | n/a | n/a | 5066.4 |
| 4 | SketchAWQ | NVIDIA A10G | 16.0291 | 167.44s | 144.73s | 144.20s | 256 | 524288 | 22.13s | 5130.5 |
| 4 | GPTQ | NVIDIA A10G | 15.6277 | 1101.10s | n/a | n/a | n/a | n/a | n/a | 5165.8 |

Decision:
- `stats_forward_seconds` nearly equals `collect_stats_seconds`.
- The bottleneck is real model forward plus hook work, not reporting or final CPU transfer.

Next action:
- Use A100 for fair speed conclusions.

### 2026-05-18 12:22:09 - A100 Hardware Scaling

Status:
- Accepted as the main hardware conclusion.

Artifact:
- `artifacts/modal/qwen3_4b_3_4bit_paper_eval_20260518_122209`

Rows:

| Bits | Method | GPU | PPL | FP16 PPL | Quant time | Stats time | Stats forward | Stats batches | Stats tokens | Layer time | Eval tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | SketchAWQ | NVIDIA A100-SXM4-40GB | 17.6985 | 15.3760 | 78.70s | 65.57s | 64.73s | 256 | 524288 | 12.29s | 16511.5 |
| 3 | GPTQ | NVIDIA A100-SXM4-40GB | 17.6109 | 15.3760 | 904.35s | n/a | n/a | n/a | n/a | n/a | 14204.7 |
| 4 | SketchAWQ | NVIDIA A100-SXM4-40GB | 16.0282 | 15.3760 | 65.62s | 54.94s | 54.47s | 256 | 524288 | 10.24s | 16277.8 |
| 4 | GPTQ | NVIDIA A100-SXM4-40GB | 15.8523 | 15.3760 | 853.17s | n/a | n/a | n/a | n/a | n/a | 15655.9 |

Decision:
- A10/A10G was a major speed limiter.
- SketchAWQ quantization was 11.49x faster than GPTQ for 3-bit and 13.00x faster for 4-bit in this run.
- PPL remained slightly behind GPTQ.

Next action:
- Stop treating broad speed as the main blocker.
- Focus on real inference and approximation failure modes.

### 2026-05-18 12:52:49 - A100 Quality Variant

Status:
- Needs config diff attribution before being used as a claim.

Artifact:
- `artifacts/modal/qwen3_4b_3_4bit_paper_eval_20260518_125249`

Rows:

| Bits | Method | GPU | PPL | FP16 PPL | Quant time | Stats time | Stats forward | Stats batches | Stats tokens | Layer time | Eval tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | SketchAWQ | NVIDIA A100-SXM4-40GB | 17.6287 | 15.3760 | 69.44s | 56.98s | 55.99s | 256 | 524288 | 11.26s | 16619.3 |
| 3 | GPTQ | NVIDIA A100-SXM4-40GB | 17.6109 | 15.3760 | 889.03s | n/a | n/a | n/a | n/a | n/a | 14101.6 |
| 4 | SketchAWQ | NVIDIA A100-SXM4-40GB | 15.9361 | 15.3760 | 89.42s | 74.64s | 72.65s | 256 | 524288 | 12.81s | 16571.1 |
| 4 | GPTQ | NVIDIA A100-SXM4-40GB | 15.8523 | 15.3760 | 1094.75s | n/a | n/a | n/a | n/a | n/a | 14999.3 |

Decision:
- This is the best observed Qwen3-4B SketchAWQ quality among the A100 runs listed here.
- It is not yet the canonical claim until the exact config delta is documented.

Next action:
- Compare its `quantization_report.json` and config metadata against `20260518_122209` and `20260518_134520`.

### 2026-05-18 13:45:20 - Config-Controlled Calibration Batch Size 8

Status:
- Accepted as the current speed-oriented Qwen3-4B run.

Artifact:
- `artifacts/modal/qwen3_4b_3_4bit_paper_eval_20260518_134520`

Change:
- Added `calibration.batch_size: 8` to Qwen3-4B SketchAWQ configs.
- Data budget stayed fixed: 256 packed C4 chunks x 2048 tokens.
- Stats batches dropped from 256 to 32.
- Observer now scales sketch row budget by runtime batch dimension so batching does not silently weaken the sketch.

Rows:

| Bits | Method | GPU | Calib batch | Eval batch | PPL | FP16 PPL | Quant time | Stats time | Stats forward | Stats batches | Stats tokens | Layer time | Eval tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | SketchAWQ | NVIDIA A100-SXM4-40GB | 8 | 1 | 17.6994 | 15.3760 | 58.27s | 44.58s | 43.36s | 32 | 524288 | 12.20s | 16639.1 |
| 3 | GPTQ | NVIDIA A100-SXM4-40GB | 1 | 1 | 17.6109 | 15.3760 | 910.96s | n/a | n/a | n/a | n/a | n/a | 14177.0 |
| 4 | SketchAWQ | NVIDIA A100-SXM4-40GB | 8 | 1 | 16.0957 | 15.3760 | 58.65s | 45.33s | 44.01s | 32 | 524288 | 11.76s | 16435.8 |
| 4 | GPTQ | NVIDIA A100-SXM4-40GB | 1 | 1 | 15.8523 | 15.3760 | 895.03s | n/a | n/a | n/a | n/a | n/a | 15576.7 |

Decision:
- Calibration batching produced the expected speed improvement.
- 3-bit remained close to GPTQ.
- 4-bit PPL moved farther behind GPTQ than the A100 quality variant.
- This strengthens the decision to study approximation failure modes instead of only optimizing PPL.

Next action:
- Add inference benchmarking.
- Add layer/block rescue diagnostics.
- Investigate why the `20260518_125249` quality variant is better than the batch-8 run.

## Cross-Experiment Observations

### Speed

- Qwen3-4B SketchAWQ quantization time moved from 208-231s on early A10 runs to about 58s on A100 with calibration batch size 8.
- GPTQ remained around 850-1100s on comparable Qwen3-4B runs.
- Stats collection is still the dominant SketchAWQ phase, but batching reduced the number of stats forwards from 256 to 32.

### Quality

- 3-bit is consistently close to GPTQ and sometimes better on older boards.
- 4-bit remains behind GPTQ in the latest Qwen3-4B A100 board.
- The best A100 quality variant in this ledger is `20260518_125249`, but its exact config delta must be documented before it becomes a canonical result.

### Throughput

- Paper-board `Tokens/s` is dense dequantized evaluation throughput.
- It is useful for sanity checks, but it is not a packed inference claim.
- The next benchmark must split evaluation throughput from true inference prefill/decode throughput.

### Hardware

- Do not compare close timing differences across A10, A10G, and A100.
- Use A100 rows for current speed claims.
- Always preserve `gpu_name` in run summaries and plots.

## Backlog Of Experiments To Add

### 2026-05-24 - Accuracy Rescue Guardrails

Status:
- Planned / tooling started.

Git:
- Branch: `main`
- Pre-guardrail anchor commit: `8d05aab`
- Guardrail setup head: `af1e850`
- Setup contains GPTQ paper-style setup, 2-bit diagnostics, accuracy-rescue
  docs, and effective-bpw reporting changes.

Purpose:
- Start the path toward near-lossless 4-bit and 3-bit quality close to current
  4-bit without overfitting one final WikiText2 board or hiding extra storage in
  protected channels and small groups.

Guardrails:
- Track `estimated_effective_bits_per_weight` with every SketchAWQ summary and
  layer metric.
- Promote only fixed recipes that hold under a confirmation board, not a single
  tuned result.
- Treat 2-bit as a diagnostic lane until a rotation/vector/codebook-style 2-bit
  mechanism exists.

Initial commands:
- `scripts/run_paper_eval.sh qwen3_4b 2 --method both --gpu A100`
- `scripts/run_paper_eval.sh qwen3_4b 3 --method both --gpu A100`
- `scripts/run_paper_eval.sh qwen3_4b 4 --method both --gpu A100`
- `scripts/sync_paper_eval_results.sh qwen3_4b all --method both`

Decision:
- Do not begin broad rescue grids until the Qwen3-4B fair 2/3/4 rows are
  re-frozen under the same calibration/evaluation contract.

Next action:
- Build the layer failure map from `layer_metrics.csv` and select one narrow
  budgeted W4 rescue experiment.

### 2026-05-18 - Inference Benchmark Harness Added

Status:
- Done.

Change:
- Added `sketchawq inference`.
- Added a reusable generation benchmark that reports prefill, decode, and total generate throughput.
- Added runtime-format fields so dense simulation cannot be confused with packed int4/int3 inference.

Current supported variants:
- `fp16`
- `sketchawq-dense`

Output files:
- per-run `inference_report.json`
- per-run `inference_metrics.json`
- per-run `inference_metrics.csv`
- appendable global `artifacts/inference/inference_metrics.csv`

Decision:
- This is now the canonical harness for inference measurements.
- The first real run should be Qwen3-4B FP16 on A100, then Qwen3-4B SketchAWQ dense simulation.

Next action:
- Run the first A100 inference matrix with prompt lengths 128/512/2048, generated tokens 128, and batch sizes 1/4.

### 2026-05-18 - First Qwen3-4B A100 Inference Matrix

Status:
- Done, but not frozen for claims.

Artifacts:
- `artifacts/modal/inference/inference/qwen3_4b_4bit_qwen_gptq_fair_balanced_packed_eval2048_fp16_inference_20260518_183049`
- `artifacts/modal/inference/inference/qwen3_4b_4bit_qwen_gptq_fair_balanced_packed_eval2048_sketchawq_dense_inference_20260518_183159`
- `artifacts/modal/inference/inference/qwen3_4b_3bit_qwen_gptq_fair_balanced_packed_eval2048_sketchawq_dense_inference_20260518_183241`
- `artifacts/modal/inference/inference/qwen3_4b_3bit_qwen_gptq_fair_balanced_packed_eval2048_fp16_inference_20260518_183548`
- Each run directory contains its own `inference_metrics.csv`; use those per-run CSVs as the source of truth.
- The appendable root CSV at `artifacts/modal/inference/inference/inference_metrics.csv` currently contains only a subset, so do not treat it as the complete merged board until append/volume behavior is checked.

Setup:
- Model: `Qwen/Qwen3-4B`
- GPU: `NVIDIA A100-SXM4-40GB`
- Prompt lengths: 128, 512, 2048
- Batch sizes: 1, 4
- Generated tokens: 128
- Warmup runs: 1
- Backend label: `transformers_generate_loop`
- Runtime formats: `fp16`, `sketchawq_dense_dequantized`

Key rows:

| Bits/config | Variant | Prompt | Batch | Prefill tok/s | Decode tok/s | Generate tok/s | Peak MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-bit config | FP16 | 128 | 1 | 2403.7 | 19.50 | 38.69 | 8111.9 |
| 4-bit config | SketchAWQ dense | 128 | 1 | 2385.3 | 19.12 | 37.94 | 8120.3 |
| 4-bit config | FP16 | 2048 | 4 | 17358.8 | 75.80 | 1204.48 | 11792.8 |
| 4-bit config | SketchAWQ dense | 2048 | 4 | 17347.8 | 73.59 | 1171.45 | 11801.2 |
| 3-bit config | FP16 | 128 | 1 | 3522.9 | 28.36 | 56.27 | 8111.9 |
| 3-bit config | SketchAWQ dense | 128 | 1 | 2608.9 | 20.88 | 41.42 | 8120.3 |
| 3-bit config | FP16 | 2048 | 4 | 17834.9 | 109.08 | 1689.11 | 11792.8 |
| 3-bit config | SketchAWQ dense | 2048 | 4 | 17063.0 | 78.65 | 1245.28 | 11801.2 |

Ratio summary:

| Pair | Prefill ratio | Decode ratio | Generate ratio |
| --- | ---: | ---: | ---: |
| 4-bit dense / matching FP16 median | 0.987x | 0.983x | 0.983x |
| 3-bit dense / matching FP16 median | 0.793x | 0.730x | 0.735x |
| 3-bit-config FP16 / 4-bit-config FP16 median | 1.351x | 1.469x | 1.463x |

Decision:
- The 4-bit dense simulated run behaves approximately like FP16, which is expected because it still executes dense dequantized weights.
- The 3-bit dense-vs-FP16 ratio is not reliable as a method conclusion because the two FP16 runs for the same Qwen3-4B base model differ by about 1.35-1.47x median depending on metric.
- Therefore this first matrix validates the harness and reinforces that dense simulation is not a packed inference speed story, but it is too noisy for final throughput claims.

Next action:
- Repeat the inference benchmark with more warmup or repeated measurements and summarize median/min/max.
- Then add a GPTQ backend inference row or move toward an int4 packed SketchAWQ backend/export path.

### 2026-05-19 - Repeat-Aware Inference Benchmark Tooling

Status:
- Done, tooling only. New repeated A100 results still need to be run.

Change:
- Added measured repeats to inference benchmarking.
- Each prompt/batch shape now warms up once and then records `N` measured sample rows.
- When `repeats > 1`, each run also writes a per-shape summary row with median/min/max timing and throughput.
- Added `measurement_kind`, `repeat_index`, `repeats`, `prompt_tokens`, `decode_tokens`, and `total_tokens` to inference metrics.
- Added an inference summary board that reads per-run `inference_metrics.csv` files, ignores incomplete root aggregate CSVs by default, and writes median/min/max CSV, JSON, and HTML.
- Updated the Modal inference entrypoint and `scripts/run_inference_eval.sh`; the helper script now defaults to `--repeats 3`.

Commands:
- Modal run shape:
  `scripts/run_inference_eval.sh qwen3_4b 4 --variant fp16 --gpu A100 --repeats 3`
- Dense SketchAWQ shape:
  `scripts/run_inference_eval.sh qwen3_4b 4 --variant sketchawq-dense --gpu A100 --repeats 3`
- Summary after download:
  `PYTHONPATH=src python -m sketchawq.experiments.summarize_inference artifacts/modal/inference/inference --output-dir artifacts/inference_summary --name qwen3_4b_a100_repeat3`

Validation:
- Focused test suite passed:
  `pytest tests/test_inference_benchmark.py tests/test_inference_summary.py tests/test_cli.py tests/test_paper_eval_scripts.py tests/test_modal_entrypoints.py`
- Result: 44 passed.
- Smoke-tested the summary builder on the first downloaded inference matrix.
- Smoke output:
  `artifacts/inference_summary/qwen3_4b_a100_first_matrix.csv`
  `artifacts/inference_summary/qwen3_4b_a100_first_matrix.json`
  `artifacts/inference_summary/qwen3_4b_a100_first_matrix.html`

Decision:
- Future inference claims should use the summary board medians, not single sample rows.
- Keep dense simulation clearly labeled as `sketchawq_dense_dequantized`; it is still not packed int4/int3 inference.

Next action:
- Re-run the Qwen3-4B A100 inference matrix with `--repeats 3`.
- Compare FP16 repeat spread first. If FP16 is stable, compare dense SketchAWQ ratios.
- Then either add GPTQ backend inference or begin the int4 packed SketchAWQ backend/export path.

### 2026-05-19 09:42 UTC - Qwen3-4B A100 Repeat-3 Inference Matrix

Status:
- Done, but not a packed-inference claim.

Artifacts:
- `artifacts/modal/inference/inference/qwen3_4b_4bit_qwen_gptq_fair_balanced_packed_eval2048_fp16_inference_20260519_094259`
- `artifacts/modal/inference/inference/qwen3_4b_4bit_qwen_gptq_fair_balanced_packed_eval2048_sketchawq_dense_inference_20260519_094411`
- `artifacts/modal/inference/inference/qwen3_4b_3bit_qwen_gptq_fair_balanced_packed_eval2048_fp16_inference_20260519_094434`
- `artifacts/modal/inference/inference/qwen3_4b_3bit_qwen_gptq_fair_balanced_packed_eval2048_sketchawq_dense_inference_20260519_094506`
- Summary board: `artifacts/inference_summary/qwen3_4b_a100_repeat3_20260519.csv`
- HTML board: `artifacts/inference_summary/qwen3_4b_a100_repeat3_20260519.html`

Setup:
- Model: `Qwen/Qwen3-4B`
- GPU: `NVIDIA A100-SXM4-40GB`
- Torch/CUDA: `torch 2.12.0+cu130`, CUDA `13.0`
- Prompt lengths: 128, 512, 2048
- Batch sizes: 1, 4
- Generated tokens: 128
- Warmup runs: 1
- Measured repeats per shape: 3
- Runtime formats: `fp16`, `sketchawq_dense_dequantized`

Key result at prompt 2048, batch 4:

| Bits | Variant | Generate tok/s median | Generate min-max | Decode tok/s median | Ratio vs FP16 | Quant time |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 3 | FP16 | 1249.6 | 1228.3-1253.5 | 78.98 | 1.000x | n/a |
| 3 | SketchAWQ dense | 1703.2 | 1692.1-1729.9 | 110.11 | 1.363x | 57.11s |
| 4 | FP16 | 1210.6 | 1163.5-1216.1 | 76.51 | 1.000x | n/a |
| 4 | SketchAWQ dense | 1143.3 | 1138.7-1149.4 | 71.88 | 0.944x | 64.81s |

Median dense-vs-FP16 ratios across all prompt/batch shapes:

| Bits | Prefill ratio | Decode ratio | Generate ratio |
| ---: | ---: | ---: | ---: |
| 3 | 1.322x | 1.459x | 1.456x |
| 4 | 0.999x | 0.970x | 0.969x |

Sanity check:
- The two FP16 config runs are now stable: 3-bit-config FP16 vs 4-bit-config FP16 generate throughput ranges from `0.987x` to `1.032x`, median `1.006x`.
- This is a major improvement over the first one-shot matrix, where FP16-to-FP16 drift was too large for speed claims.

Interpretation:
- The 4-bit dense result behaves as expected: approximately FP16, slightly slower at longer contexts.
- The 3-bit dense result is surprisingly faster than FP16 even though it is still dense dequantized simulation. Treat this as a real measured phenomenon in this run, but not yet an inference-backend result.
- Possible explanations to test: run-order/cache effects, generated-token path differences, kernel/autotune state, or quantized-weight value distributions affecting model execution indirectly.

Decision:
- Do not claim packed speed yet.
- Do not treat 3-bit dense speedup as a core contribution until it survives order-controlled reruns.
- The immediate next inference experiment should isolate whether 3-bit dense speedup is reproducible and causal.

Next action:
- Run order-controlled repeat matrix: FP16 3-bit config again after all dense runs, then 3-bit dense again with the same prompt/batch matrix.
- Add GPTQ inference backend only after the dense-vs-FP16 behavior is understood well enough to make comparison meaningful.

### Inference

- FP16 generate benchmark for Qwen3-4B on A100.
- GPTQ backend generate benchmark for Qwen3-4B on A100.
- SketchAWQ dense simulation generate benchmark for Qwen3-4B.
- SketchAWQ packed/backend generate benchmark after backend exists.

### Approximation Failure

- Qwen3-4B 4-bit block rescue.
- Qwen3-4B 3-bit block rescue.
- Correlate block rescue with layer search scores.
- Compare `20260518_125249` vs `20260518_134520` configs and search-choice plots.
- Repeat the same diagnostics on Qwen3-8B after Qwen3-4B tooling works.

### Calibration

- Batch size 1 vs 4 vs 8 on A100 with identical config otherwise.
- C4 128 vs 256 vs 512 samples.
- Sequence length 1024 vs 2048.
- Alternate calibration seed.
- WikiText calibration as an ablation only, not the main final claim.

## Entry Template

```text
### YYYY-MM-DD HH:MM - Experiment Name

Status:
- planned/running/done/rejected/frozen

Git:
- Commit:
- Branch:
- Dirty worktree:

Command:
- 

Configs:
- SketchAWQ:
- Baseline:

Artifacts:
- 

Setup:
- Model:
- Method:
- Bits:
- Group size:
- Search objective:
- Sketch:
- Outliers:
- Calibration:
- Evaluation:
- GPU:

Results:
- FP16 PPL:
- Quantized PPL:
- PPL delta vs GPTQ:
- Quantization time:
- Stats time:
- Stats forward time:
- Layer time:
- Stats batches:
- Stats tokens:
- Eval throughput:
- Inference throughput:
- Compression:

Plots:
- 

Decision:
- 

Next action:
- 
```
