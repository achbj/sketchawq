# Migration Notes

This clean folder was created from the original `sketchawq` checkout.

## What Moved

- `src/sketchawq/` was copied as the reusable package.
- Selected unit/smoke tests were copied; historical config-matrix tests were left behind.
- `modal_app.py` and `modal_baselines.py` moved to `infra/modal/`.
- The config matrix was reduced to 28 curated config/catalog files:
  - 3 smoke configs
  - 12 canonical SketchAWQ paper configs
  - 12 GPTQ paper-eval baseline configs
  - 1 experiment catalog
- `scripts/run_paper_eval.sh` and the `scripts/paper_eval/*_{3,4}bit.sh`
  wrappers replaced the old all-in-one experiment shell tables.
- Shell experiment dispatchers were replaced by `sketchawq` CLI commands and `configs/experiments.yaml`.

## What Stayed In The Old Repo

- Full ablation sweeps
- Historical paper-eval variants
- Large sync/plot shell dispatch tables
- Rough notes and generated artifacts

Pull individual configs forward only when they become active again.
