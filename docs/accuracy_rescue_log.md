# SketchAWQ Accuracy Rescue Log

Last updated: 2026-05-24

Purpose: improve 4-bit toward near-FP16 loss and move 3-bit toward current
4-bit quality without overfitting one leaderboard row or hiding extra storage
inside protected channels and tiny groups.

## Current Baseline Snapshot

Git state at log creation:

- Branch: `main`
- Base commit: `8d05aab`
- Worktree: dirty; includes GPTQ paper-style setup, 2-bit diagnostic configs,
  2-bit launcher support, and effective-bpw reporting.
- Important pre-existing untracked files: `configs/paper/*_v2.yaml`; these were
  already present before this rescue log and should be treated as separate
  experimental drafts.

Qwen3-4B fair setup after the GPTQ cleanup:

| Config | Bits | Group | Calibration | Evaluation | Run name |
| --- | ---: | ---: | --- | --- | --- |
| `configs/paper/qwen3_4b_2bit.yaml` | 2 | 128 | C4 train, 256 x 2048 packed | WikiText2 validation, full 2048 packed | `qwen3_4b_2bit_qwen_gptq_fair_balanced_packed_eval2048` |
| `configs/paper/qwen3_4b_3bit.yaml` | 3 | 128 | C4 train, 256 x 2048 packed | WikiText2 validation, full 2048 packed | `qwen3_4b_3bit_qwen_gptq_fair_balanced_packed_eval2048` |
| `configs/paper/qwen3_4b_4bit.yaml` | 4 | 128 | C4 train, 256 x 2048 packed | WikiText2 validation, full 2048 packed | `qwen3_4b_4bit_qwen_gptq_fair_balanced_packed_eval2048` |
| `configs/baselines/qwen3_4b_gptq_2bit.yaml` | 2 | 128 | C4 train, 256 x 2048 packed | WikiText2 validation, full 2048 packed | `qwen3_4b_gptq_2bit_g128_eval2048` |
| `configs/baselines/qwen3_4b_gptq_3bit.yaml` | 3 | 128 | C4 train, 256 x 2048 packed | WikiText2 validation, full 2048 packed | `qwen3_4b_gptq_3bit_g128_eval2048` |
| `configs/baselines/qwen3_4b_gptq_4bit.yaml` | 4 | 128 | C4 train, 256 x 2048 packed | WikiText2 validation, full 2048 packed | `qwen3_4b_gptq_4bit_g128_eval2048` |

GPTQ baseline configs now use the stronger paper-style knobs when the installed
GPTQModel version supports them:

- `damp_percent: 0.01`
- `desc_act: true`
- `true_sequential: true`

## Non-Overfit Rules

1. Keep calibration and final evaluation separated.
2. Tune on Qwen3-4B only as the development proxy; confirm on Qwen3-8B and at
   least one non-Qwen family before promoting a recipe.
3. Do not promote a change from a single best run. Use repeat or alternate-seed
   confirmation when PPL gaps are small.
4. Treat C4 validation and alternate calibration seed as confirmation boards,
   not tuning targets.
5. Track `estimated_effective_bits_per_weight` beside PPL. A PPL win that
   spends too much effective bpw is a storage tradeoff, not a method win.
6. Reject model-specific micro-grids unless the same idea helps across at least
   two model sizes or families.

## Promotion Criteria

4-bit rescue can move from experiment to candidate only if:

- WikiText2 PPL is within the agreed near-lossless band from FP16.
- C4 or alternate-seed confirmation does not regress materially.
- Effective bpw stays inside the declared W4 budget band.
- Quantization time remains meaningfully faster than GPTQ.

3-bit rescue can move from experiment to candidate only if:

- 3-bit PPL approaches the current honest 4-bit row on Qwen3-4B.
- The same recipe direction helps Qwen3-8B or Mistral/LLaMA.
- Effective bpw stays inside the declared W3 budget band.

2-bit remains diagnostic until the method includes a real 2-bit mechanism such
as rotation plus vector/codebook/lattice/trellis quantization.

## Experiment Ladder

Run narrow steps in this order:

1. Re-freeze Qwen3-4B fair rows: W2/W3/W4 SketchAWQ and GPTQ, same C4
   calibration and full WikiText2 validation.
2. Build a layer failure map from `layer_metrics.csv`, ranking by search score,
   protected channels, and effective bpw.
3. Test budgeted 4-bit rescue: W4 g128 base plus limited rescue only on the
   worst layers.
4. Test budgeted 3-bit rescue: W3 g128 base plus the same rescue mechanism.
5. Confirm only the best fixed recipe on Qwen3-8B, C4 validation, and one
   non-Qwen model family.

## Commands

Run Qwen3-4B all-bit paper eval:

```bash
scripts/run_paper_eval.sh qwen3_4b 2 --method both --gpu A100
scripts/run_paper_eval.sh qwen3_4b 3 --method both --gpu A100
scripts/run_paper_eval.sh qwen3_4b 4 --method both --gpu A100
```

Sync one combined 2/3/4-bit board:

```bash
scripts/sync_paper_eval_results.sh qwen3_4b all --method both
```

Dry-run before spending GPU time:

```bash
scripts/run_paper_eval.sh qwen3_4b 4 --method both --gpu A100 --dry-run
scripts/sync_paper_eval_results.sh qwen3_4b all --method both --dry-run
```

## Revert Notes

Before starting a risky idea, save:

- exact config paths
- run names
- command lines
- artifact bundle path
- `git diff --stat`
- this log entry updated with the decision rule

If an idea is rejected, do not delete the artifact. Mark it rejected here and in
`docs/experiment_ledger.md`, then either revert the small config/code change or
create a new fixed config with a new run name. Avoid overwriting historical run
names.

