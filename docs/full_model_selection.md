# Full-Model Heldout Selection Board

Status: implemented and validated on the current W3/W4 strict-recipe control
pair. This board is the gate for future SketchAWQ accuracy candidates.

Date: 2026-07-09

Implementation commits:

- `bb667d4` added the config, runner, report fields, and comparison table.
- `792e4aa` added frozen W3/W4 selection-only controls and Modal scripts.

## Protocol

Search uses the fixed 256 packed C4-train samples at sequence length 2048.
Model selection uses a disjoint 32 packed C4-validation sample set at the same
length. The runner evaluates both FP16 and the completed quantized model on
that heldout data. The selection configs set `evaluation.enabled: false`, so
they neither load nor evaluate WikiText2.

The result is reported under `model_selection` in `quantization_report.json`,
as `model_selection_metrics.csv`, and through:

```bash
python -m sketchawq.experiments.compare_methods \
  <run-directories> \
  --table model-selection \
  --format csv
```

The selection board chooses a fixed candidate recipe. Only a selected candidate
may move to a separately launched WikiText2 confirmation run. It must then
survive an alternate seed and another model family before promotion.

## Initial Validation

Artifact bundle:

`artifacts/model_selection/qwen3_4b_model_selection_20260709_210000`

Model: `Qwen/Qwen3-4B`, FP16, symmetric g128, NVIDIA A100-SXM4-40GB.

| Bits | FP16 C4 PPL | Strict control C4 PPL | Refined C4 PPL | Delta | Control time | Refined time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 23.1017 | 54.9869 | 36.3325 | -18.6543 | 56.53s | 76.99s |
| 4 | 23.1017 | 27.9812 | 25.1709 | -2.8103 | 55.11s | 76.31s |

The current local sketch-conditioned scale refinement improves both widths on
the heldout board. Its C4 ordering matches the previously frozen WikiText2
results, where the same W3/W4 refinement was the best current local treatment.
That agreement validates the selection path; it does not make the C4 split a
new tuning target.

## Decision

Keep the local scale-refinement recipe as the development control. Reject a
future candidate before WikiText2 if it fails to improve the C4 heldout loss
against this row at the same bit budget and g128 layout. Do not choose among
multiple implementations using WikiText2 PPL.

## Commands

```bash
scripts/run_model_selection_ablation.sh both --gpu A100
scripts/sync_model_selection_ablation.sh both
```
