# SketchAWQ

This is the cleaned project layout for SketchAWQ. The old checkout is still useful as an experiment archive; this folder is the codebase we should develop from.

For the current result coverage, robustness checks, caveats, and next-step
roadmap, see [docs/project_status_roadmap.md](docs/project_status_roadmap.md).
For the current accuracy-rescue guardrails and rollback notes, see
[docs/accuracy_rescue_log.md](docs/accuracy_rescue_log.md).

SketchAWQ is a post-training quantization pipeline that adds lightweight Hessian sketch signals to AWQ-style scaling. Current packed exports use `sketchawq-packed-v1`; throughput numbers are still dense dequantized simulation until a packed inference kernel is added.

## Layout

```text
src/sketchawq/        Python package: quantization, eval, baselines, reporting
configs/smoke/        Fast local configs
configs/paper/        Canonical SketchAWQ result configs
configs/baselines/    GPTQ paper-eval baseline configs
configs/experiments.yaml
                      Named experiment targets for the CLI
scripts/              Thin paper-eval launchers
infra/modal/          Modal entrypoints for remote GPU runs
docs/                 Architecture and migration notes
tests/                Focused unit and smoke tests
```

For baseline choices beyond GPTQ, see [docs/comparison_methods.md](docs/comparison_methods.md).

## Quick Start

```bash
python -m pip install -e ".[dev]"
sketchawq list-targets
sketchawq run configs/smoke/tiny.yaml
```

Run a named target from the catalog:

```bash
sketchawq run-target smoke
sketchawq run-target paper-qwen2
sketchawq run-target paper-qwen3-1-7b
sketchawq run-target paper-qwen3-4b-best
sketchawq run-target paper-qwen3-14b
sketchawq run-target paper-qwen3-32b
sketchawq run-target paper-mistral7b
```

Run an external baseline config:

```bash
sketchawq baseline configs/baselines/qwen2_5_1_5b_gptq_4bit.yaml
sketchawq run-target baseline-qwen3-1-7b
sketchawq run-target baseline-qwen3-14b
sketchawq run-target baseline-qwen3-32b
sketchawq run-target baseline-mistral7b
```

Use the per-model paper-eval scripts when you want a direct launcher. They run on
Modal by default and can switch to GPTQ or local dry-runs:

```bash
scripts/paper_eval/qwen3_14b_3bit.sh
scripts/paper_eval/qwen3_14b_3bit.sh --gptq
scripts/paper_eval/qwen3_4b_2bit.sh --method both --dry-run
scripts/paper_eval/mistral_7b_4bit.sh --method both --dry-run
scripts/paper_eval/qwen3_32b_4bit.sh --method both --dry-run
scripts/run_paper_eval.sh qwen2_5_1_5b 4 --method gptq --backend local
```

Run on Modal:

```bash
modal run infra/modal/sketchawq_app.py --config configs/smoke/tiny.yaml
modal run infra/modal/sketchawq_app.py --configs configs/paper/qwen2_5_1_5b_3bit.yaml,configs/paper/qwen2_5_1_5b_4bit.yaml
modal run infra/modal/baseline_app.py --config configs/baselines/qwen2_5_1_5b_gptq_4bit.yaml
```

Choose a different Modal GPU with `--gpu` on the paper-eval launcher or by setting
`SKETCHAWQ_MODAL_GPU` for direct Modal runs:

```bash
scripts/run_paper_eval.sh qwen3_8b 3 --method both --gpu L40S
scripts/run_paper_eval.sh mistral_7b 4 --method both --gpu L40S
scripts/run_paper_eval.sh qwen3_32b 4 --method both --gpu H100
SKETCHAWQ_MODAL_GPU=A100 modal run infra/modal/sketchawq_app.py --config configs/paper/qwen3_14b_4bit.yaml
```

After Modal jobs finish, sync their artifacts and build the comparison table/plot:

```bash
scripts/sync_paper_eval_results.sh qwen3_8b both --method both
```

This downloads the expected run folders from the `sketchawq-artifacts` Modal volume into
`artifacts/modal/` and writes CSV/HTML/PNG/PDF outputs into `artifacts/paper_eval/`.

Run the Qwen3-8B cross-eval on C4 validation with:

```bash
scripts/run_paper_eval.sh qwen3_8b 3 --method both --eval c4 --gpu L40S
scripts/run_paper_eval.sh qwen3_8b 4 --method both --eval c4 --gpu L40S
scripts/sync_paper_eval_results.sh qwen3_8b both --method both --eval c4
```

For gated LLaMA runs, export `HF_TOKEN` locally or set `SKETCHAWQ_HF_SECRET` to a Modal secret name.

## Development

```bash
make test
make lint
```

The large historical config matrix was intentionally not copied here. Keep only configs that represent current smoke tests, canonical paper rows, or active baseline comparisons. Use the old repo as the archive when a past ablation needs to be recovered.
