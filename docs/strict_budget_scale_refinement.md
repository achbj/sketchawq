# Strict-Budget Scale Refinement Experiment

Status: implemented, awaiting GPU results

Date: 2026-06-13

Branch: `codex/sketch-conditioned-refinement`

Base commit: `255a6a2`

## Why This Experiment Exists

Nominal bit width is not enough for a SOTA claim. Several earlier SketchAWQ
rows used FP16 protected channels or smaller groups:

- Qwen3-4B W3 g128 reported about 4.11 effective bits per weight.
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

## Rollback

The experiment is isolated on `codex/sketch-conditioned-refinement`. To return
to the pre-experiment code, switch back to `main`. Do not delete result
artifacts; mark the experiment rejected in the accuracy log.
