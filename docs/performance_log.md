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

| Bits | Run | Quant time | Stats time | PPL |
| --- | --- | ---: | ---: | ---: |
| 3 | old balanced | 231.48s | 208.62s | 17.7010 |
| 3 | GPU accumulators + search-grid edit | 188.70s | 165.42s | 17.7015 |
| 4 | old balanced | 208.41s | 184.37s | 15.8668 |
| 4 | GPU accumulators + search-grid edit | 165.70s | 143.34s | 16.0346 |

The accumulator change reduced stats time by about 20-22%, but it did not collapse the phase to the hoped-for range. The remaining stats cost is likely dominated by model forward passes, per-layer row sampling, and the sketch matrix products inside hooks.

The 4-bit PPL regression is probably confounded by the search-grid edit. The earlier 4-bit search used exact `0.1` alpha and allowed higher clip ratios such as `0.9`/`0.93`; the latest grid narrowed alpha and clip candidates around lower values.

## 2026-05-17 - Reverted tensor-score search experiment

### Context

We tried keeping search stats on GPU through layer quantization and batching candidate score selection so each scale/clip search synchronized once per stage. The goal was to reduce `layer_quantization_seconds`.

### Result

The experiment did not improve runtime. Recent Qwen3-4B bundles showed layer time staying flat or increasing:

| Bundle | Bits | Quant time | Stats time | Layer time | PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20260517_192225` | 3 | 188.70s | 165.42s | 22.48s | 17.7015 |
| `20260517_203358` | 3 | 200.57s | 174.82s | 24.93s | 17.7015 |
| `20260517_192225` | 4 | 165.70s | 143.34s | 21.87s | 16.0346 |
| `20260517_203358` | 4 | 187.14s | 165.02s | 21.55s | 16.0291 |

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
