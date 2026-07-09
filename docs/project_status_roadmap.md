# SketchAWQ Project Status Roadmap

Last updated: 2026-05-15

This document is the working status board for the clean SketchAWQ checkout. It
tracks which result rows are trustworthy, which robustness checks exist, and
what to do next without overfitting the method to one final benchmark board.

## Current Situation

- This `sketchawq-clean` folder is the active development workspace. The old
  `sketchawq` checkout remains useful as an experiment archive.
- The clean repo contains the reusable package, curated paper configs, GPTQ
  baseline configs, Modal entrypoints, paper-eval scripts, and focused tests.
- Canonical SketchAWQ configs live under `configs/paper/`; matching GPTQ
  baseline configs live under `configs/baselines/`.
- The current synced artifacts in this checkout cover Qwen3-4B and Qwen3-8B.
  Qwen3-8B also has a later C4 validation board intended as a robustness check.
- Qwen3-1.7B and Qwen3-32B configs are present for both methods and both bit
  widths, but no synced artifact summaries for those models are present in this
  clean checkout yet.

## Current Outcomes

Lower perplexity is better. Time is `quantization_seconds` from the synced
artifact summary, not end-to-end benchmark runtime. Rows here are taken from
the clean checkout only.

| Model | Eval board | Bits | SketchAWQ PPL | GPTQ PPL | SketchAWQ time | GPTQ time | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-1.7B | WikiText2 validation 2048 | 3 | Not synced | Not synced | Not synced | Not synced | Configured, needs run/sync |
| Qwen3-1.7B | WikiText2 validation 2048 | 4 | Not synced | Not synced | Not synced | Not synced | Configured, needs run/sync |
| Qwen3-4B | WikiText2 validation 2048 | 3 | 16.535 | 16.938 | 222.07s | 1011.99s | Quality win, about 4.6x faster |
| Qwen3-4B | WikiText2 validation 2048 | 4 | 15.303 | 14.847 | 167.63s | 1083.78s | Speed win, quality gap |
| Qwen3-8B | WikiText2 validation 2048 | 3 | 11.367 | 11.807 | 185.53s | 1094.62s | Quality win, about 5.9x faster |
| Qwen3-8B | WikiText2 validation 2048 | 4 | 10.510 | 10.500 | 102.93s | 945.34s | Near-tie, about 9.2x faster |
| Qwen3-8B | C4 validation 2048 | 3 | 17.297 | 17.474 | 178.76s | 979.23s | Robustness quality win, about 5.5x faster |
| Qwen3-8B | C4 validation 2048 | 4 | 15.973 | 16.098 | 102.71s | 937.93s | Robustness quality win, about 9.1x faster |
| Qwen3-32B | WikiText2 validation 2048 | 3 | Not synced | Not synced | Not synced | Not synced | Configured, needs run/sync |
| Qwen3-32B | WikiText2 validation 2048 | 4 | Not synced | Not synced | Not synced | Not synced | Configured, needs run/sync |

Coverage notes:

- Qwen3-8B has the strongest overfitting check currently present here: C4
  calibration with WikiText evaluation, plus a separate C4 validation board.
- Qwen3-32B uses C4 calibration and WikiText evaluation in config, but the
  actual 32B artifact rows are not synced into this clean folder.
- Qwen2.5, Qwen3-14B, and LLaMA-2 7B paper rows are represented by configs in
  this clean checkout, but their synced summary artifacts are not present here.

## Known Caveats

- Evaluation still uses the dense dequantized simulation path. Packed int3/int4
  artifacts are storage/export artifacts until real packed inference kernels
  are added.
- `artifacts/paper_eval/4b_baseline_oe/` mixes Qwen3-4B SketchAWQ eval2048 rows
  with GPTQ eval128 rows. Do not use that board for direct method comparison.
- The current Qwen3-8B C4 board checks non-WikiText evaluation robustness, but
  it does not replace alternate calibration-seed validation.
- No Qwen3-32B synced result artifacts are available in this clean checkout.
  Treat 32B status as configured but not verified here.
- Time ratios are useful directionally, but GPU names still matter when making
  tight speed claims.

## Accuracy Without Overfitting Protocol

Use three levels of evaluation:

| Level | Purpose | Default rule |
| --- | --- | --- |
| Tune/dev | Fast iteration and proxy objective checks | Use calibration-only or small held-out subsets; do not report as final |
| Confirm | Catch dataset-specific wins | Use a different seed and at least one non-WikiText validation set |
| Final | Paper-facing result table | Run rarely after the change passes tune and confirm checks |

Decision rules:

- Keep calibration and evaluation data separate for final claims. The current
  Qwen3 paper configs calibrate on C4 and evaluate on WikiText, which is good.
- Treat the Qwen3-8B C4 validation board as confirmation evidence, not as a new
  tuning target.
- Select new recipes by median or average across seeds, not by the best single
  run.
- Reject a change if it improves only one final WikiText board but fails the
  C4 or alternate-seed confirmation board.
- Reject model-specific micro-search unless the same idea wins across at least
  two confirmation settings or two model families.
- Keep PPL, quantization time, compression ratio, benchmark name, calibration
  source, sequence length, GPU, and seed together in every progress entry.

## Step-by-Step Roadmap

### Step 1: Verify And Clean Current Result Boards

- [x] Identify the later Qwen3-8B robustness artifacts:
      `artifacts/paper_eval/2nd_run/qwen3_8b_3_4bit_paper_eval_summary.csv`
      and `artifacts/paper_eval/8b_c4_eval/qwen3_8b_3_4bit_c4_eval_summary.csv`.
- [x] Mark the mixed Qwen3-4B `4b_baseline_oe` board as non-comparable because
      it mixes eval2048 SketchAWQ rows with eval128 GPTQ rows.
- [ ] Sync or rerun Qwen3-32B 3-bit and 4-bit artifacts for both SketchAWQ and
      GPTQ, then add the rows to this document.
- [ ] Decide whether Qwen3-32B also needs a C4 validation board matching the
      Qwen3-8B robustness check.
- [ ] Bring Qwen2.5, Qwen3-1.7B, Qwen3-14B, and LLaMA-2 7B synced summaries
      into the clean checkout only if they are still part of the active paper
      table.

### Step 2: Freeze A Trustworthy Baseline Table

- [ ] Treat the table above as the clean-checkout baseline, not as the whole
      historical project record.
- [ ] Freeze one table per model family with exact artifact paths and run IDs.
- [ ] Do not tune from rows marked `not synced` or from mixed benchmark boards.
- [ ] Keep Qwen3-4B, Qwen3-8B, and Qwen3-32B rows separated so one model's
      quirks do not silently drive another model's recipe.

### Step 3: Add Robustness Checks Before New Tuning

- [x] Qwen3-8B: compare C4-calibrated WikiText eval against C4 validation eval.
- [x] Add a Qwen3-4B full-model C4-validation selection board that disables
      WikiText2 while candidates are compared.
- [ ] Qwen3-8B: rerun final configs with at least one alternate calibration
      seed.
- [ ] Qwen3-32B: run/sync the configured C4-calibrated WikiText rows.
- [ ] Qwen3-32B: add and run a C4 validation confirmation board if 32B is used
      for overfitting claims.
- [ ] Re-run matching GPTQ rows whenever a board is used for claims, not only
      SketchAWQ rows.

### Step 4: Accuracy/PPL Improvement Track

- [ ] First test safer calibration changes: mixed calibration text, more
      calibration samples, and matched final sequence length.
- [ ] Track every accuracy rescue experiment against effective bpw so smaller
      groups/protected channels cannot quietly turn W3/W4 rows into higher-bit
      storage budgets.
- [ ] Then test limited fixed-recipe objective checks for rows with small gaps,
      especially Qwen3-8B 4-bit and any future Qwen3-32B 4-bit row.
- [ ] Keep each experiment narrow: one reason to run, one variable group, one
      decision rule.
- [ ] Avoid adding new knobs unless a fixed recipe fails across multiple
      families and the failure is visible in layer metrics.
- [ ] Promote a recipe only if it improves or near-ties PPL while preserving the
      quantization-time advantage that makes SketchAWQ useful.

### Step 5: Deployment Realism Track

- [ ] Keep packed artifact export status documented separately from PPL quality.
- [ ] Track packed inference kernels as a separate milestone from dense
      simulation evaluation.
- [ ] Before any deployment-facing claim, verify that saved artifacts include
      `packed_model.safetensors` and `packed_manifest.json` when
      `output.save_model: true`.
- [ ] Do not claim runtime inference speedups until real packed kernels exist
      and are benchmarked.

## Progress Log

Use this format for future updates:

```text
### YYYY-MM-DD - Short Title

Status:
- Pending / Running / Done / Rejected / Frozen

Evidence:
- Artifact paths, commands, board rows, or test output.

Decision:
- What changed in the project status.

Next action:
- The single next concrete step.
```

### 2026-05-15 - Clean Roadmap Created

Status:
- Done.

Evidence:
- Added this roadmap in the clean checkout.
- Confirmed Qwen3-8B has synced WikiText and C4 validation summaries.
- Confirmed Qwen3-32B configs exist but no Qwen3-32B artifact summaries are
  synced in `artifacts/`.

Decision:
- Use this file as the shared status tracker for the clean repo.
- Treat Qwen3-8B as having partial robustness evidence.
- Treat Qwen3-32B as configured but not yet verified in this checkout.

Next action:
- Run or sync Qwen3-32B 3-bit and 4-bit rows for SketchAWQ and GPTQ, then add
  them to the Current Outcomes table.

### 2026-05-24 - Accuracy Rescue Guardrails Started

Status:
- Done.

Evidence:
- Added `docs/accuracy_rescue_log.md` with baseline state, non-overfit rules,
  promotion criteria, experiment ladder, commands, and revert notes.
- Added effective-bpw reporting to run summaries, layer metrics, and benchmark
  boards so quality improvements can be judged against storage budget.
- Added 2-bit paper-eval configs/scripts as diagnostics, not as promoted
  competitive rows.

Decision:
- Use Qwen3-4B as the development proxy, but promote only recipes that survive
  confirmation on another dataset/seed and another model size or family.

Next action:
- Re-freeze Qwen3-4B 2/3/4-bit SketchAWQ and GPTQ rows, then use layer metrics
  to choose the smallest possible 4-bit rescue experiment.
