# SketchAWQ Performance Log

## 2026-05-17 - GPU-resident calibration accumulators

### Context

Qwen3-4B fair runs showed stats collection dominating SketchAWQ runtime. The observer was copying small per-layer accumulator updates from GPU to CPU inside every linear-layer forward hook:

- activation absolute sums
- activation square sums
- Hessian sketch sums
- sampled activation rows for the hybrid output-error objective

These repeated device synchronizations are expensive even though each tensor is small.

### Change

The activation observer now creates `LayerStats` accumulators on the same device as each linear layer weight. During calibration it keeps these tensors device-local:

- `abs_sum`
- `sq_sum`
- `sketch_omega`
- `sketch_sum`
- `activation_sample`

The accumulators use `float32` instead of `float64`, matching the downstream sketch/search precision and avoiding slow FP64 math on A10/A10G-class GPUs. After the calibration pass, `collect_layer_stats` moves every stats tensor back to CPU once, so the rest of the quantization pipeline keeps the same interface.

### Expected impact

This should reduce `collect_stats_seconds` substantially by removing per-layer, per-batch GPU-to-CPU synchronization. It does not change the calibration data, search objective, group size, or GPTQ comparison setup.

The exact speedup still needs a Modal timing run. The target metric is:

- before: Qwen3-4B stats collection around 185-210 seconds in recent fair runs
- after: materially lower `collect_stats_seconds`, ideally without PPL regression

### Risks

- VRAM usage increases because layer accumulators stay on GPU during calibration. For Qwen3-4B/8B this should be modest because sketches are `[in_features, rank]`, but larger models should still be watched.
- `float32` accumulation may introduce tiny numeric differences versus the previous CPU `float64` accumulation. The previous pipeline converted these terms back to `float32` before search, so this is expected to be low risk.
- CUDA generators are used for CUDA-resident row sampling to avoid CPU index creation. CPU behavior remains covered by tests.

### Verification

- Added observer tests for device-aware accumulator creation and final CPU materialization.
- Ran observer and Hessian sketch tests locally.
- Ran the full test suite locally: observer changes passed, with one existing Qwen3-4B paper-config expectation failure caused by the current search-grid edits.

### First Qwen3-4B result

Bundle: `artifacts/paper_eval/qwen3_4b_3_4bit_paper_eval_20260517_192225`

Compared with the earlier balanced bundle `artifacts/paper_eval/OLD/qwen3_4b_3_4bit_paper_eval_20260517_163259`:

| Bits | Run | GPU recorded | Quant time | Stats time | PPL |
| --- | --- | --- | ---: | ---: | ---: |
| 3 | old balanced | NVIDIA A10 | 231.48s | 208.62s | 17.7010 |
| 3 | GPU accumulators + search-grid edit | NVIDIA A10G | 188.70s | 165.42s | 17.7015 |
| 4 | old balanced | NVIDIA A10 | 208.41s | 184.37s | 15.8668 |
| 4 | GPU accumulators + search-grid edit | NVIDIA A10 | 165.70s | 143.34s | 16.0346 |

The accumulator change reduced stats time by about 20-22%, but it did not collapse the phase to the hoped-for range. The remaining stats cost is likely dominated by model forward passes, per-layer row sampling, and the sketch matrix products inside hooks.

The 4-bit PPL regression is probably confounded by the search-grid edit. The earlier 4-bit search used exact `0.1` alpha and allowed higher clip ratios such as `0.9`/`0.93`; the latest grid narrowed alpha and clip candidates around lower values.

## 2026-05-17 - Reverted tensor-score search experiment

### Context

We tried keeping search stats on GPU through layer quantization and batching candidate score selection so each scale/clip search synchronized once per stage. The goal was to reduce `layer_quantization_seconds`.

### Result

The experiment did not improve runtime. Recent Qwen3-4B bundles showed layer time staying flat or increasing:

| Bundle | Bits | GPU recorded | Quant time | Stats time | Layer time | PPL |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `20260517_192225` | 3 | NVIDIA A10G | 188.70s | 165.42s | 22.48s | 17.7015 |
| `20260517_203358` | 3 | NVIDIA A10G | 200.57s | 174.82s | 24.93s | 17.7015 |
| `20260517_192225` | 4 | NVIDIA A10 | 165.70s | 143.34s | 21.87s | 16.0346 |
| `20260517_203358` | 4 | NVIDIA A10G | 187.14s | 165.02s | 21.55s | 16.0291 |

### Decision

Reverted the tensor-score search path and the pipeline GPU-cache management experiment. Keep the earlier GPU-resident calibration accumulator change, because that had a measurable stats-time improvement. Next speed work should target calibration batching rather than layer-search score synchronization.

## 2026-05-17 - Stats-phase timing and low-risk calibration forward cleanup

### Context

The latest Qwen3-4B 4-bit runs still spend almost all quantization time in stats collection:

- `20260517_203358`: 187.14s total, 165.02s stats, 21.55s layer quantization
- `20260517_220520`: 188.67s total, 165.87s stats, 22.22s layer quantization

That means the next speed work should measure and reduce the calibration forward/stat path rather than the AWQ search loop.

### Change

The stats collector now reports additional timing/context fields into the quantization report and `run_summary.csv`:

- `stats_batches`
- `stats_tokens`
- `stats_forward_seconds`
- `stats_tokens_per_second`
- `stats_use_cache_disabled`
- `stats_attention_mask_fast_path_batches`
- `stats_masked_batches`

Calibration stats collection now uses `torch.inference_mode()`, passes `use_cache=False` when the model forward signature supports it, and skips per-layer attention-mask filtering when the calibration batch mask is all ones. Packed C4 calibration should hit the all-ones fast path for every batch.

### Expected impact

This is intentionally low-risk. It may shave some overhead, but its main job is to make the next Modal run diagnosable. If `stats_forward_seconds` is nearly the same as `collect_stats_seconds`, the bottleneck is the model forward plus hooks. If the gap is large, CPU-side finalization or artifact/reporting work is suspect.

### Verification

- Focused stats/pipeline/reporting tests passed.
- Nearby observer, pipeline, reporting, model-save, Hessian sketch, and search tests passed.
- Full suite passed except the existing Qwen3-4B paper-config candidate-grid expectation mismatch.

## 2026-05-18 - A100 run confirms the GPU ceiling

### Context

After the stats-phase instrumentation was added, Qwen3-4B was rerun on an A100. This finally separated algorithm cost from A10/A10G hardware limits. The calibration setup stayed the fair GPTQ-comparison shape:

- model: `Qwen/Qwen3-4B`
- calibration: C4 train, 256 packed samples, sequence length 2048
- evaluation: WikiText2 validation packed eval2048
- group size: 128
- SketchAWQ stats accumulators: `cuda`, `float32`

### Result

The A100 run shows that GPU throughput, especially during stats collection, was the main speed limiter in the earlier Modal runs. On A10/A10G, stats collection was typically 145-175s after the GPU-accumulator fix. On A100, it dropped to 55-66s while preserving the same calibration/evaluation shape.

| Bundle | Bits | Method | GPU recorded | Quant time | Stats forward | Stats total | Layer time | PPL |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `20260517_225427` | 3 | SketchAWQ | NVIDIA A10G | 173.35s | 146.37s | 147.57s | 24.77s | 17.7015 |
| `20260518_122209` | 3 | SketchAWQ | NVIDIA A100-SXM4-40GB | 78.70s | 64.73s | 65.57s | 12.29s | 17.6985 |
| `20260517_225427` | 4 | SketchAWQ | NVIDIA A10G | 167.44s | 144.20s | 144.73s | 22.13s | 16.0291 |
| `20260518_122209` | 4 | SketchAWQ | NVIDIA A100-SXM4-40GB | 65.62s | 54.47s | 54.94s | 10.24s | 16.0282 |

The timing fields also explain where the time goes. For the A100 4-bit run, `collect_stats_seconds=54.94s` and `stats_forward_seconds=54.47s`, so nearly all stats time is real model-forward plus hook work, not final CPU movement or reporting overhead. The all-ones mask fast path was active for packed calibration (`stats_attention_mask_fast_path_batches=256` in the run summaries).

### GPTQ comparison on A100

The same A100 paper-eval bundle also contains GPTQ baselines. SketchAWQ is now much faster to quantize, while still slightly behind GPTQ in PPL:

| Bits | Method | GPU recorded | Quant time | PPL | Speedup vs GPTQ |
| ---: | --- | --- | ---: | ---: | ---: |
| 3 | GPTQ | NVIDIA A100-SXM4-40GB | 904.35s | 17.6109 | 1.00x |
| 3 | SketchAWQ | NVIDIA A100-SXM4-40GB | 78.70s | 17.6985 | 11.49x |
| 4 | GPTQ | NVIDIA A100-SXM4-40GB | 853.17s | 15.8523 | 1.00x |
| 4 | SketchAWQ | NVIDIA A100-SXM4-40GB | 65.62s | 16.0282 | 13.00x |

This is the clearest result so far: with proper GPU throughput, SketchAWQ's quantization path is more than 10x faster than GPTQ for Qwen3-4B under the fair comparison setup. The remaining research target is no longer broad speed, but closing the PPL gap without giving back too much of that speed advantage.

### Decision

Keep recording the actual CUDA-reported `gpu_name` on every experiment record. The run-level summary already carries `gpu_name`; evaluation and layer metric records should also keep it so future CSVs, boards, and plots can be grouped by hardware. For speed conclusions, compare runs only within the same GPU class unless the purpose is explicitly a hardware-scaling comparison.
