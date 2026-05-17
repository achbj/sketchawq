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

## 2026-05-17 - GPU-resident search stats and tensor candidate scores

### Context

After GPU-resident collection, stats were still moved back to CPU before sketch preparation and per-layer search. The search code also converted every candidate score to a Python float, forcing a GPU synchronization for each alpha/clip candidate.

### Change

- `collect_layer_stats` now accepts `finalize_to_cpu`; the public default remains `true`, while the quantization pipeline passes `false`.
- The pipeline keeps `LayerStats` and `HessianSketch` tensors on the model device through each layer's quantization, then moves that layer's cache back to CPU after it is no longer needed.
- `weighted_error_score_tensor` and `output_error_score_tensor` keep scalar scores on-device.
- AWQ scale/clip search now stacks candidate score tensors and synchronizes once per search stage to select the best candidate, rather than converting each candidate score to a Python float.

### Expected impact

This should mostly affect `layer_quantization_seconds`, not `collect_stats_seconds`. The previous Qwen3-4B layer search phase was only about 22 seconds, so the absolute win is expected to be modest.

### Verification

- Focused tests passed: observer, search, tiny pipeline, Hessian sketch, and reporting metrics.
- Full suite result: one known Qwen3-4B paper-config expectation failure remains because the current config search grid differs from the checked-in test expectation.
