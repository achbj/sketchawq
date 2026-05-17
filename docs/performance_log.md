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
