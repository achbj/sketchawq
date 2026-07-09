# Strict-Budget Scale Refinement Experiment

Status: Qwen3-4B development test passed; confirmation pending

Date: 2026-06-13

Branch: `codex/sketch-conditioned-refinement`

Base commit: `255a6a2`

## Why This Experiment Exists

Nominal bit width is not enough for a SOTA claim. Several earlier SketchAWQ
rows used FP16 protected channels or smaller groups:

- Qwen3-4B W3 g128 with protected channels used about 3.89 effective bits per
  weight. Its reported 4.11 value was compression ratio, not effective bpw.
- Llama-2-7B W4 g32 with protected channels reported about 5.64 effective bits
  per weight.

Those runs remain useful diagnostics, but they are not strict W3/W4 results.
This experiment therefore disables protected channels and fixes group size 128
for both the control and treatment.

## Hypothesis

The current implementation uses the Hessian sketch to rank AWQ scale and clip
candidates, but the group quantizer still derives its scales from the weight
range. The treatment alternates between:

1. rounding weights to the existing integer grid; and
2. refitting each group scale against the diagonal-plus-low-rank sketch loss.

The integer width, group size, scale count, and packed format do not change.
Any quality improvement therefore comes from better scale fitting rather than
hidden precision.

This is an experiment, not yet a novelty claim. Related recent work includes:

- [D2Quant](https://arxiv.org/abs/2602.02546), which uses dual-scale
  quantization and activation-deviation correction.
- [DASH-Q](https://arxiv.org/abs/2604.13806), which uses stable diagonal
  curvature and iterative weighted scale fitting.
- [HeRo-Q](https://arxiv.org/abs/2601.21626), which conditions the Hessian with
  smoothing and rotations.
- [FOEM](https://arxiv.org/abs/2507.11017), which improves sequential error
  compensation with first-order terms.
- [ICQuant](https://openreview.net/forum?id=m6nBgFSMTL), which reduces outlier
  overhead through index coding.

The SketchAWQ-specific question is whether the low-rank sketch adds reliable
value beyond diagonal weighted fitting without requiring full Hessian
compensation or training.

## Controlled Rows

The configs are under `configs/experiments/strict_budget/`.

For each bit width, control and treatment are identical except for
`scale_refinement: none` versus `scale_refinement: sketch`.

All rows use:

- Qwen3-4B
- symmetric W3 or W4
- group size 128
- no FP16 protected channels
- C4 train calibration, 256 samples, length 2048
- WikiText2 validation evaluation, length 2048
- the same seed and candidate grids

## Commands

Dry-run:

```bash
scripts/run_strict_budget_ablation.sh both --variant both --gpu A100 --dry-run
```

Run:

```bash
scripts/run_strict_budget_ablation.sh both --variant both --gpu A100
```

Sync and compare:

```bash
scripts/sync_strict_budget_ablation.sh both
```

## Decision Rule

Reject the mechanism if either treatment row is worse than its same-bit
control, if effective bits per weight differ materially, or if quantization
time loses the practical advantage over GPTQ.

Advance it to confirmation only when:

1. W4 improves over strict W4 control and moves toward FP16.
2. W3 improves over strict W3 control.
3. The result repeats with an alternate calibration seed.
4. The fixed recipe improves Qwen3-8B and one non-Qwen family.

Do not tune the refinement step count or low-rank weight on final WikiText2
results. If the mechanism is promising, choose those values using a development
split and freeze them before confirmation.

## Qwen3-4B Development Result

Artifact:

`artifacts/strict_budget/qwen3_4b_strict_budget_20260613_234319`

| Bits | Variant | Effective bpw | PPL | PPL vs FP16 | Quantization | Peak CUDA |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | control | 3.25 | 39.9267 | +24.5506 | 56.54s | 15.97 GB |
| 3 | refinement | 3.25 | 27.1889 | +11.8129 | 76.99s | 20.64 GB |
| 4 | control | 4.25 | 19.9803 | +4.6042 | 54.99s | 15.97 GB |
| 4 | refinement | 4.25 | 16.8022 | +1.4262 | 76.20s | 20.64 GB |

FP16 PPL was 15.3760 in all four rows.

Observed effect:

- W3 PPL fell by 12.7377, a 31.9% reduction versus the strict control.
- W4 PPL fell by 3.1780, a 15.9% reduction versus the strict control.
- Effective bpw was exactly unchanged within each bit width.
- Quantization time increased by about 36-39%, while remaining around 11x
  faster than the existing Qwen3-4B GPTQ A100 rows.
- Peak CUDA allocation increased by about 4.67 GB.
- The proxy score improved for all 253 layers at both widths. This is expected
  because refinement directly optimizes that score and is not independent
  confirmation.

Interpretation:

- The hypothesis passed the Qwen3-4B development gate.
- W4 is promising: strict 4.25 bpw reaches 16.8022 PPL without protected FP16
  channels, close to the earlier 16.0957 protected-channel row.
- W3 remains far from the target: 27.1889 is a large improvement but is not yet
  near the prior 4-bit quality or competitive GPTQ quality.
- Do not call this SOTA and do not replace canonical configs yet.

Next confirmation:

1. Repeat the frozen four-row matrix with an alternate calibration seed.
2. Run frozen control/refinement rows on Qwen3-8B.
3. Run the same fixed mechanism on OPT-2.7B or Llama/Mistral.

## Rollback

The experiment is isolated on `codex/sketch-conditioned-refinement`. To return
to the pre-experiment code, switch back to `main`. Do not delete result
artifacts; mark the experiment rejected in the accuracy log.
