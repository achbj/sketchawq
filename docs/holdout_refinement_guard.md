# Held-Out Refinement Guard Experiment

Status: the C4 validation holdout infrastructure is retained, but the
layer-local output guard is rejected as an accuracy mechanism for global scale
refinement.

Date: 2026-07-09

Branch: `codex/sketch-conditioned-refinement`

Implementation commits:

- `f06eb22` added disjoint calibration and holdout loading.
- `1b37a57` added an opt-in holdout guard policy.
- `e83b9a8` added holdout candidate scoring.
- `56cf519` integrated the guard with reporting.
- `93f18bc` added frozen W3/W4 launch and sync controls.

## Protocol

The original 256 C4-train packed 2048-token samples remain the only source for
AWQ search and the Hessian sketch. A disjoint 32-sample packed C4-validation
set supplies activation rows and tied-token frequencies only for choosing
between the unrefined and globally refined candidate of each Linear layer.
WikiText2 remains a final evaluation board and is not used for that choice.

The treatment is otherwise identical to the global Nystrom ablation: Qwen3-4B,
symmetric g128 W3/W4, no protected channels, fixed candidate grids, rank 64,
and seed 42.

## Result

Artifact bundle:

`artifacts/holdout_guard/qwen3_4b_holdout_guard_20260709_203700`

| Bits | Nystrom PPL | Global PPL | Held-out guard PPL | Accepted layers |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 39.0111 | 74.7686 | 54.4967 | 246 / 253 |
| 4 | 18.0479 | 18.8322 | 18.9229 | 250 / 253 |

The guard rejects the W3 `lm_head` candidate, whose held-out combined score
rose from 0.0945 to 0.1209. It accepts the W4 `lm_head`, where the score falls
from 0.0241 to 0.0221. Despite these coherent local decisions, neither width
improves full-model PPL. The holdout activation pass takes about 5.2-5.5
seconds; total timings across separate Modal runs are not suitable for a speed
claim.

## Decision

Keep the disjoint C4 validation data path and the per-layer telemetry. Do not
promote the layer-local guard or global scale refinement. The next candidate
must be selected by a full-model held-out loss, not solely by independent layer
output reconstruction. That selection must happen before a final WikiText2
board, followed by an alternate seed and cross-model confirmation.

## Commands

```bash
scripts/run_holdout_guard_ablation.sh both --gpu A100
scripts/sync_holdout_guard_ablation.sh both
```
