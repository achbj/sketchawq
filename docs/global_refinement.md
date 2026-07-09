# Global Sketch Refinement Experiment

Status: rejected Qwen3-4B development treatment. The code and artifacts are
preserved so that the outcome can be reproduced and compared without reusing
the run names.

Date: 2026-07-09

Branch: `codex/sketch-conditioned-refinement`

Implementation commits:

- `012eb86` added the coupled coordinate update.
- `9fad129` added fixed W3/W4 treatment controls.
- `a36e0df` added the launcher and artifact-sync scripts.

## Hypothesis

Independent group-scale fitting sees a low-rank factor only inside one group.
The treatment keeps the complete factor residual while fitting each group,
allowing scale changes in one group to account for errors in all other groups.

## Fixed Protocol

The treatment changes only `scale_refinement` from `none` to
`global_sketch` relative to the valid Nystrom-residual controls. Both widths
use Qwen3-4B, symmetric g128 weights, no protected channels, C4 train
calibration with 256 packed 2048-token samples, and full packed WikiText2
validation at 2048 tokens. No WikiText2 result was used to alter a candidate
grid or an acceptance threshold.

## Result

Artifact bundle:

`artifacts/global_refinement/qwen3_4b_global_refinement_20260709_201325`

| Bits | Nystrom control PPL | Global refinement PPL | Quantization time | Layer time |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 39.0111 | 74.7686 | 90.06s | 41.60s |
| 4 | 18.0479 | 18.8322 | 89.12s | 43.01s |

FP16 PPL was 15.3760 in both rows. The paired runs completed on an
NVIDIA A100-SXM4-40GB with the same C4 seed, model revision, and evaluation
protocol as their controls.

The treatment was much worse at W3 and regressed at W4 while increasing layer
quantization time by about four times. It is not a viable accuracy mechanism.

## Commands

```bash
scripts/run_global_refinement_ablation.sh both --gpu A100
scripts/sync_global_refinement_ablation.sh both
```

Use `--dry-run` before launching or syncing. The sync script collects frozen
strict controls, Nystrom-residual controls, and the global-treatment rows into
one comparison bundle.

## Decision

Do not tune the global update against WikiText2 or promote it to paper configs.
The failure exposes a design gap: post-search refinement is accepted only by
its reconstruction objective, even though the original search uses a hybrid
curvature/output objective and a tied-embedding term. The next mechanism must
evaluate a refined candidate against a calibration-only output-aware guard,
then confirm promising changes on an independent calibration split and new
model/seed before looking at final paper boards.
