# Spectrum-Aware Curvature Experiment

Status: Qwen3-4B development result retained as a component experiment, not
promoted to a paper recipe.

Date: 2026-07-09

Branch: `codex/sketch-conditioned-refinement`

Implementation commits:

- `6f2ad8c` added the opt-in Nystrom curvature factor.
- `79505dc` corrected the factor-plus-diagonal decomposition.
- `07f0ff3` preserved the invalid first run and added fresh residual controls.

## Hypothesis

The previous sketch used an orthonormal basis from `QR(H Omega)`. That retained
the subspace but discarded eigenvalue magnitudes, then restored a manually tuned
low-rank scalar in the search loss. The experiment instead forms a Nystrom
factor `F` from `H Omega` and scores reconstruction with:

`E diag(residual) E^T + (E F)(E F)^T`

The full diagonal remains available for outlier and sensitivity ranking. The
search objective receives only the diagonal residual, preventing the factor
from being counted twice.

## Fixed Protocol

Both rows use Qwen3-4B, symmetric W3 or W4, group size 128, no protected
channels, C4 train calibration with 256 packed 2048-token samples, and full
packed WikiText2 validation at 2048 tokens. Alpha and clipping candidate lists,
seed, rank, and evaluation protocol are frozen from the strict-budget controls.

The Nystrom path adds no packed model tensors, no extra precision, and no
smaller groups. Historical strict-control effective-bpw values cannot be
compared byte-for-byte with these rows because commit `6129423` corrected the
storage estimator to include AWQ input-scale payloads.

## Invalidated Smoke Run

The first Nystrom implementation accidentally passed the full diagonal together
with the factor. This double-counted diagonal curvature in the candidate loss.
The run proved that Modal execution and artifact reporting worked, but it is not
valid accuracy evidence.

- Modal app: `ap-UJbleaZpR7yIr3ckLD7BkW`
- Local artifact: `artifacts/spectrum_curvature/qwen3_4b_spectrum_curvature_20260709_194600`
- Remote runs: `qwen3_4b_3bit_nystrom_g128_control`,
  `qwen3_4b_4bit_nystrom_g128_control`

These names are intentionally preserved and are not reused by the corrected
implementation.

## Valid Development Result

Artifact bundle:

`artifacts/spectrum_curvature/qwen3_4b_spectrum_curvature_20260709_195700`

| Bits | Strict control PPL | Nystrom residual PPL | Improvement | Quantization time |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 39.9267 | 39.0111 | 0.9156 | 56.13s |
| 4 | 19.9803 | 18.0479 | 1.9323 | 54.81s |

FP16 PPL was 15.3760 for both rows. The valid Nystrom residual result reduces
control PPL by about 2.3% at W3 and 9.7% at W4. It is a real same-protocol
improvement, but it remains far behind the existing strict scale-refinement
result: 27.1889 at W3 and 16.8022 at W4.

## Commands

```bash
scripts/run_spectrum_curvature_ablation.sh both --gpu A100
scripts/sync_spectrum_curvature_ablation.sh both
```

Use `--dry-run` before launching or syncing. The sync command fetches the
matching frozen strict controls as well as the Nystrom residual runs and writes
a comparison CSV bundle.

## Decision

Keep the Nystrom factor as an opt-in component because it improved both widths
without an accuracy budget change. Do not promote it alone and do not tune the
eigenvalue threshold or candidate grid on WikiText2. The next fixed experiment
uses the same factor inside globally coupled scale refinement, where each group
update accounts for the residual from all other groups.
