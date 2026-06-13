# SketchAWQ Accuracy Rescue Log

Last updated: 2026-06-13

Purpose: improve 4-bit toward near-FP16 loss and move 3-bit toward current
4-bit quality without overfitting one leaderboard row or hiding extra storage
inside protected channels and tiny groups.

## Current Baseline Snapshot

Git state at log creation:

- Branch: `main`
- Pre-guardrail anchor commit: `8d05aab`
- Guardrail setup commits: commits after `8d05aab`; use `git log --oneline
  8d05aab..HEAD` to inspect the exact local checkpoint range.
- Guardrail setup includes GPTQ paper-style setup, 2-bit diagnostic configs,
  2-bit launcher support, accuracy-rescue docs, and effective-bpw reporting.
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

Run supported OPT-2.7B GPTQ/AWQ baselines against a paper-style recipe:

```bash
scripts/run_opt_baselines.sh both --method both --gpu A100
```

Dry-run the same OPT-2.7B rows before spending GPU time:

```bash
scripts/run_opt_baselines.sh both --method both --gpu A100 --dry-run
```

OPT-2.7B baseline configs:

| Config | Method | Bits | Group | Calibration | Evaluation | Run name |
| --- | --- | ---: | ---: | --- | --- | --- |
| `configs/baselines/opt_2_7b_gptq_3bit.yaml` | GPTQModel-GPTQ | 3 | 128 | C4 train, 128 x 2048 packed | WikiText2 test, full 2048-token concat blocks | `opt_2_7b_gptq_3bit_g128_c4_128_wikitext2_test_concat_eval2048` |
| `configs/baselines/opt_2_7b_gptq_4bit.yaml` | GPTQModel-GPTQ | 4 | 128 | C4 train, 128 x 2048 packed | WikiText2 test, full 2048-token concat blocks | `opt_2_7b_gptq_4bit_g128_c4_128_wikitext2_test_concat_eval2048` |
| `configs/baselines/opt_2_7b_awq_3bit.yaml` | GPTQModel-AWQ | 3 | 128 | C4 train, 128 x 2048 packed | WikiText2 test, full 2048-token concat blocks | `opt_2_7b_awq_3bit_g128_c4_128_wikitext2_test_concat_eval2048` |
| `configs/baselines/opt_2_7b_awq_4bit.yaml` | GPTQModel-AWQ | 4 | 128 | C4 train, 128 x 2048 packed | WikiText2 test, full 2048-token concat blocks | `opt_2_7b_awq_4bit_g128_c4_128_wikitext2_test_concat_eval2048` |

Notes:
- GPTQ rows use `damp_percent: 0.01`, `desc_act: true`, and `true_sequential: true`.
- AWQ rows use GPTQModel-AWQ with `backend: auto` and asymmetric zero-point style quantization through `symmetric: false`. GPTQModel's AWQ auto path should choose the AWQ GEMM backend for normal GEMM-format AWQ instead of the portable torch fallback.
- OPT evaluation uses `packing_strategy: concat` to match the GPTQ reference WikiText2 protocol: concatenate the split text with blank-line separators, then evaluate fixed-length blocks without inserting EOS tokens between dataset rows.
- GPTQModel-AWQ is configured for 3-bit and 4-bit here because the AWQ reference repo documents INT3/INT4 support for OPT. The `awq_torch` runtime backend rejects 3-bit kernels, so `scripts/run_opt_baselines.sh both --method both` skips AWQ 3-bit unless `--include-awq-3bit` is passed explicitly.

External AWQ reference check from `/Users/achbj/Work/quantization`:

| Bits | Method | Backend | Calibration | Evaluation | PPL |
| ---: | --- | --- | --- | --- | ---: |
| 3 | official `mit-han-lab/llm-awq` | fake quant | Pile validation, 128 samples | WikiText2 test, 2048-token concat blocks | 13.5499 |
| 4 | official `mit-han-lab/llm-awq` | fake quant | Pile validation, 128 samples | WikiText2 test, 2048-token concat blocks | 12.7043 |

Interpretation:
- Use these external `llm-awq` rows as the AWQ paper-style OPT-2.7B targets.
- The GPTQModel-AWQ `9837.5970` PPL run is a backend failure, not evidence that AWQ is weak.
- AWQ 3-bit is a fake-quant paper comparison row; do not expect a real 3-bit CUDA kernel from the original AWQ repo.

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

## 2026-06-13 - Strict-Budget Sketch-Conditioned Scale Refinement

Status:
- Implemented, awaiting GPU results.

Git:
- Base commit: `255a6a2`
- Branch: `codex/sketch-conditioned-refinement`

Hypothesis:
- Refitting group scales directly against the diagonal-plus-low-rank sketch
  objective can improve W3/W4 reconstruction without protected channels,
  smaller groups, or extra stored parameters.

Guardrails:
- Qwen3-4B control and treatment use group size 128 and no FP16 protection.
- The only treatment change is `scale_refinement: sketch`.
- Promotion requires wins at both bit widths followed by alternate-seed,
  Qwen3-8B, and non-Qwen confirmation.

Commands:

```bash
scripts/run_strict_budget_ablation.sh both --variant both --gpu A100
scripts/sync_strict_budget_ablation.sh both
```

Detailed protocol:
- `docs/strict_budget_scale_refinement.md`

Decision:
- Do not modify canonical paper configs until the controlled experiment passes
  its promotion rule.
