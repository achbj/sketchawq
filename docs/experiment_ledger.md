# SketchAWQ Experiment Ledger

Last updated: 2026-07-09

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

### 2026-05-25 18:22:53 - OPT-2.7B GPTQ Concat Baseline Sync

Status:
- Accepted as paper-protocol baseline check for GPTQ.

Artifact:
- `artifacts/modal/opt_2_7b_concat_baselines_20260525_182253`
- `artifacts/paper_eval/opt_2_7b_concat_baselines_20260525_182253`

Purpose:
- Verify the corrected OPT-2.7B WikiText2 concatenated evaluation protocol.

Setup:
- Model: `facebook/opt-2.7b`
- Calibration: C4 train, 128 samples, sequence length 2048, packed stream
- Evaluation: WikiText2 test, sequence length 2048, `packing_strategy: concat`
- GPU: NVIDIA A100-SXM4-40GB

Rows:

| Bits | Method | FP16 PPL | Quantized PPL | FP16 MB | GPTQ MB | Compression | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | GPTQModel-GPTQ | 12.4710 | 14.6416 | 5303.2 | 1292.4 | 4.10x | Accept: FP16 anchor is paper-like and quantized row is strong. |
| 4 | GPTQModel-GPTQ | 12.4710 | 12.6324 | 5303.2 | 1607.0 | 3.30x | Accept: close to FP16 and suitable as the GPTQ W4 reference. |

Decision:
- The corrected concat evaluation fixed the FP16 anchor, from 19.0135 down to 12.4710.
- This confirms the old OPT results were dominated by evaluation protocol mismatch, not GPTQ quality.
- Use these GPTQ rows as the OPT-2.7B sanity check before comparing SketchAWQ or AWQ.
- Size values were backfilled as structure estimates: quantized linear weights plus FP16 non-quantized parameters. Future GPTQ runs record the actual saved GPTQModel folder size during quantization.

Next action:
- Run or debug AWQ 4-bit separately under the corrected concat protocol.
- Do not compare AWQ 3-bit with `awq_torch`; that backend rejected non-4-bit kernels.

### 2026-05-25 19:20 - OPT-2.7B GPTQModel-AWQ Torch Backend Failure

Status:
- Rejected.

Artifact:
- `artifacts/modal/opt_2_7b_concat_baselines_20260525_182253/runs/opt_2_7b_awq_4bit_g128_c4_128_wikitext2_test_concat_eval2048`

Rows:

| Bits | Method | Backend | FP16 PPL | Quantized PPL | Decision |
| ---: | --- | --- | ---: | ---: | --- |
| 4 | GPTQModel-AWQ | `awq_torch` | 12.4710 | 9837.5970 | Reject: quantized runtime/checkpoint is unusable. |

Diagnosis:
- Evaluation protocol is correct because FP16 is 12.4710 and GPTQ 4-bit is 12.6324 on the same bundle.
- The failure is specific to the GPTQModel-AWQ backend path we selected.
- The config forced `backend: awq_torch`; GPTQModel's auto path chooses AWQ GEMM for GEMM-format AWQ on CUDA, while `awq_torch` is the portable torch fallback.

Change made:
- Updated OPT AWQ 4-bit config to `backend: auto`.
- Updated the local GPTQModel AWQ default backend from `awq_torch` to `auto`.

Next action:
- Rerun `scripts/run_opt_baselines.sh 4 --method awq --gpu A100`.
- If AWQ is still bad with `backend: auto`, switch this baseline to official `llm-awq`/AutoAWQ rather than treating GPTQModel-AWQ as the reference.

### 2026-05-25 19:39 - OPT-2.7B Official llm-awq Baseline Check

Status:
- Accepted as the AWQ paper-style reference for OPT-2.7B.

External repo:
- `/Users/achbj/Work/quantization`

Artifact:
- Modal run: `https://modal.com/apps/ach-bj7/main/ap-aitDVJHk0331a9MfNSYnyp`
- Modal run: `https://modal.com/apps/ach-bj7/main/ap-KF5S7nbKhM53rfxzrx0XAF`
- Modal volume reports:
  - `/artifacts/opt_2_7b_awq_4bit_g128_wikitext2/report.json`
  - `/artifacts/opt_2_7b_awq_3bit_g128_wikitext2/report.json`
- Recovered local summary: `/Users/achbj/Work/quantization/artifacts/recovered/opt_2_7b_awq_recovered_20260525.json`

Purpose:
- Check AWQ with the official `mit-han-lab/llm-awq` path instead of GPTQModel-AWQ.
- Establish the AWQ paper-comparable anchor for OPT-2.7B before judging SketchAWQ.

Setup:
- Model: `facebook/opt-2.7b`
- Method: official `llm-awq`
- Quantization: asymmetric AWQ, group size 128
- Calibration: AWQ built-in Pile validation subset, 128 samples
- Evaluation: WikiText2 test, sequence length 2048, concatenated text blocks
- Backend: `q_backend=fake`

Rows:

| Bits | Method | Backend | FP16 Ref PPL | Quantized PPL | Delta | Decision |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 3 | official llm-awq | fake quant | 12.47 | 13.5499 | +1.0799 | Accept: matches AWQ INT3-g128 paper anchor. |
| 4 | official llm-awq | fake quant | 12.47 | 12.7043 | +0.2343 | Accept: close to FP16 and usable as the W4 AWQ reference. |

Diagnosis:
- These numbers show AWQ itself is healthy on OPT-2.7B.
- The earlier `9837.5970` PPL row is a GPTQModel-AWQ backend failure, not an AWQ algorithm failure.
- AWQ 3-bit should be compared through fake-quant evaluation because the original `llm-awq` repository does not provide a 3-bit CUDA inference kernel.
- The external harness report JSON originally recorded run status but not the printed PPL. Patched `/Users/achbj/Work/quantization/infra/modal/baseline_app.py` so future AWQ reports include `awq_eval.json` plus parsed WikiText2 PPL in `report.json`. The current Modal JSONs still need a rerun to include those fields.

Next action:
- Treat official `llm-awq` fake-quant rows as the AWQ reference until GPTQModel-AWQ reproduces them.
- Compare SketchAWQ against AWQ targets: W4 near `12.70`, W3 near `13.55` on OPT-2.7B WikiText2 test.

### 2026-05-25 17:40:45 - OPT-2.7B GPTQ/AWQ Baseline Sync

Status:
- Rejected as paper-comparison evidence.

Artifact:
- `artifacts/modal/opt_2_7b_baselines_20260525_174045`
- `artifacts/paper_eval/opt_2_7b_baselines_20260525_174045`

Purpose:
- Check whether the first OPT-2.7B GPTQ/AWQ baseline configs reproduce paper-like WikiText2 loss.

Setup:
- Model: `facebook/opt-2.7b`
- Calibration: C4 train, 128 samples, sequence length 2048, packed stream
- Evaluation: WikiText2 test, sequence length 2048, packed stream with EOS separators
- GPU: A100 variants

Rows:

| Bits | Method | Status | FP16 PPL | Quantized PPL | Decision |
| ---: | --- | --- | ---: | ---: | --- |
| 3 | GPTQModel-GPTQ | done | 19.0135 | 22.2432 | Reject: FP16 anchor too high for GPTQ paper protocol. |
| 4 | GPTQModel-GPTQ | done | 19.0135 | 19.9237 | Reject: FP16 anchor too high for GPTQ paper protocol. |
| 3 | GPTQModel-AWQ | failed | n/a | n/a | Backend rejected AWQ 3-bit: `AwqTorchLinear` supports only 4-bit. |
| 4 | GPTQModel-AWQ | done | 19.0135 | 10871.2447 | Reject: severe quality failure; investigate backend/config only after eval anchor is fixed. |

Diagnosis:
- The GPTQ paper's OPT-2.7B WikiText2 table reports FP16 around 12.46, GPTQ-4 around 12.87, and GPTQ-3 around 16.88.
- The local FP16 anchor was 19.0135, so the evaluation text packing was not paper-comparable.
- The likely mismatch was the old packed stream path inserting EOS-style separators between WikiText rows; GPTQ's reference loader concatenates WikiText text with blank-line separators and slices fixed-length blocks.

Change made:
- Added `packing_strategy: concat` for paper-style concatenated evaluation.
- Updated OPT configs to use `evaluation.packing_strategy: concat`.
- Updated `scripts/run_opt_baselines.sh` to skip AWQ 3-bit by default because GPTQModel `awq_torch` rejects it.

Next action:
- Rerun supported OPT-2.7B rows: GPTQ 3-bit, GPTQ 4-bit, AWQ 4-bit.
- Treat AWQ 3-bit as requiring a different backend or a fake-quant paper harness, not the current `awq_torch` runtime path.

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
- Guardrail setup commits: commits after `8d05aab`; use `git log --oneline
  8d05aab..HEAD` to inspect the exact local checkpoint range.
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

### 2026-06-13 - Strict-Budget Scale Refinement Implemented

Status:
- Qwen3-4B development test passed; confirmation pending.

Git:
- Branch: `codex/sketch-conditioned-refinement`
- Base commit: `255a6a2`

Change:
- Added opt-in sketch-conditioned group-scale refinement.
- Added strict g128 W3/W4 controls with no protected FP16 channels.
- Added run and sync scripts for the four-row control/refinement matrix.
- Added `scale_refinement` and `scale_refinement_steps` to summary and layer
  metrics.

Decision rule:
- Reject if either treatment loses to its same-bit control or spends more
  effective bits per weight.
- Confirm a positive result with an alternate seed, Qwen3-8B, and one non-Qwen
  family before changing canonical paper configs.

Result:
- W3: 39.9267 control to 27.1889 refinement at unchanged 3.25 effective bpw.
- W4: 19.9803 control to 16.8022 refinement at unchanged 4.25 effective bpw.
- Quantization time rose from about 55-57s to about 76-77s.
- Peak CUDA allocation rose from 15.97 GB to 20.64 GB.
- Artifact:
  `artifacts/strict_budget/qwen3_4b_strict_budget_20260613_234319`

Decision:
- Advance the frozen mechanism to confirmation.
- W4 is promising but not yet SOTA; W3 still needs a stronger mechanism.

Commands:
- `scripts/run_strict_budget_ablation.sh both --variant both --gpu A100`
- `scripts/sync_strict_budget_ablation.sh both`

Protocol:
- `docs/strict_budget_scale_refinement.md`

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

### 2026-07-09 - Spectrum-Aware Curvature Search

Status:
- Done as a development component; not promoted.

Evidence:
- Valid bundle:
  `artifacts/spectrum_curvature/qwen3_4b_spectrum_curvature_20260709_195700`
- W3 PPL: 39.9267 strict control to 39.0111 Nystrom residual.
- W4 PPL: 19.9803 strict control to 18.0479 Nystrom residual.
- Invalidated pre-fix bundle retained at:
  `artifacts/spectrum_curvature/qwen3_4b_spectrum_curvature_20260709_194600`

Decision:
- Keep the spectrum factor as an opt-in component and preserve the invalidated
  run for auditability.
- Do not promote the result without a stronger W3 mechanism and cross-model
  confirmation.

Next action:
- Replace independent low-rank group-scale fitting with a globally coupled
  residual update, then rerun W3 and W4 under the same fixed protocol.

### 2026-07-09 - Globally Coupled Nystrom Scale Refinement

Status:
- Rejected.

Git:
- Implementation: `012eb86`
- Controls: `9fad129`
- Scripts: `a36e0df`
- Branch: `codex/sketch-conditioned-refinement`
- Dirty worktree: no source changes during the Modal run.

Command:

```bash
scripts/run_global_refinement_ablation.sh both --gpu A100
scripts/sync_global_refinement_ablation.sh both
```

Configs:
- `configs/experiments/global_refinement/qwen3_4b_3bit_global_refine.yaml`
- `configs/experiments/global_refinement/qwen3_4b_4bit_global_refine.yaml`

Artifact:
- `artifacts/global_refinement/qwen3_4b_global_refinement_20260709_201325`

Setup:
- Model: `Qwen/Qwen3-4B`, FP16, default revision.
- Method: Nystrom diagonal-plus-factor curvature with global scale refinement.
- Bits/group size: symmetric W3 and W4, g128.
- Calibration: C4 train, 256 packed 2048-token samples, seed 42, batch size 8.
- Evaluation: full packed WikiText2 validation at 2048 tokens.
- GPU: NVIDIA A100-SXM4-40GB.

Results:

| Bits | FP16 PPL | Nystrom control PPL | Global PPL | Quantization time | Layer time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 15.3760 | 39.0111 | 74.7686 | 90.06s | 41.60s |
| 4 | 15.3760 | 18.0479 | 18.8322 | 89.12s | 43.01s |

Decision:
- Reject the global residual update. It is a W3 collapse and a W4 regression,
  while adding substantial quantization time.
- Do not alter the mechanism through final-evaluation tuning.

Next action:
- Add a calibration-only output-aware acceptance guard for post-search
  refinements, then confirm any improvement on a held-out calibration split,
  alternate seed, and cross-model board.

### 2026-07-09 - Held-Out Output Guard for Global Refinement

Status:
- Rejected as an accuracy treatment; retained as evaluation infrastructure.

Git:
- Data holdout: `f06eb22`
- Guard config: `1b37a57`
- Scoring: `e83b9a8`
- Pipeline/reporting: `56cf519`
- W3/W4 controls: `93f18bc`

Command:

```bash
scripts/run_holdout_guard_ablation.sh both --gpu A100
scripts/sync_holdout_guard_ablation.sh both
```

Artifact:
- `artifacts/holdout_guard/qwen3_4b_holdout_guard_20260709_203700`

Setup:
- Model: `Qwen/Qwen3-4B`, FP16, default revision.
- Method: global Nystrom refinement with a holdout-output candidate guard.
- Bits/group size: symmetric W3 and W4, g128.
- Search/sketch calibration: C4 train, 256 packed 2048-token samples, seed 42.
- Selection-only calibration: C4 validation, 32 packed 2048-token samples.
- Final evaluation: packed WikiText2 validation at 2048 tokens.
- GPU: NVIDIA A100-SXM4-40GB.

Results:

| Bits | Nystrom PPL | Unguarded global PPL | Guarded PPL | Accepted layers |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 39.0111 | 74.7686 | 54.4967 | 246 / 253 |
| 4 | 18.0479 | 18.8322 | 18.9229 | 250 / 253 |

Decision:
- The C4 validation holdout path behaves correctly, including sensible tied
  embedding decisions, but layer-local reconstruction does not predict
  full-model PPL well enough to rescue global coupling.
- Keep the infrastructure. Reject global refinement and do not tune it on
  WikiText2.

Next action:
- Select future mechanisms by full-model held-out loss before final evaluation,
  then confirm any winner under a separate seed and model family.
