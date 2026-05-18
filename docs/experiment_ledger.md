# SketchAWQ Experiment Ledger

Last updated: 2026-05-18

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
