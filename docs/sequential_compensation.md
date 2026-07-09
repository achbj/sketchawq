# Sequential Sketch Error Compensation

## Status

Promoted as the fixed Qwen3-4B W3/W4 development candidate on 2026-07-09.
It passed a full-model held-out C4 selection board and a separately launched
WikiText2 confirmation board. It is not yet a paper-facing or SOTA claim:
the next required confirmations are an alternate calibration seed and a second
model family.

## Fixed Mechanism

Sequential compensation runs after the existing local sketch scale refinement.
It splits each input-channel axis into fixed 1024-column blocks aligned to the
quantization groups. The first block retains its existing refined quantization;
each later block receives a low-rank Hessian-informed residual update from the
already quantized blocks, then reuses the established local scale refinement.

The implementation uses the existing diagonal plus low-rank sketch and a
Woodbury solve. It does not construct a dense Hessian or inverse. The fixed
block size, rank, calibration budget, search candidates, and group size were
not changed after the C4 selection result.

## Selection Board

Artifact:

- `artifacts/sequential_compensation/qwen3_4b_sequential_compensation_20260709_221500`

Protocol:

- Model: `Qwen/Qwen3-4B`, FP16, default revision.
- Quantization: symmetric W3/W4, group size 128, rank 64 local sketch scale
  refinement.
- Search calibration: 256 packed C4-train samples, 2048 tokens, seed 42.
- Selection board: disjoint 32 packed C4-validation samples, 2048 tokens.
- WikiText2 evaluation: disabled.
- GPU: NVIDIA A100-SXM4-40GB.

| Bits | FP16 C4 PPL | Refined control C4 PPL | Sequential C4 PPL | Delta |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 23.1017 | 36.3325 | 35.2573 | -1.0753 |
| 4 | 23.1017 | 25.1709 | 24.8391 | -0.3319 |

## Independent WikiText2 Confirmation

Artifact:

- `artifacts/sequential_compensation/confirmation/qwen3_4b_sequential_compensation_confirmation_20260709_222500`

Protocol:

- The quantizer used the same fixed 256 packed C4-train samples and seed 42.
- No C4-validation holdout was loaded for this confirmation run.
- Evaluation used full packed WikiText2 validation, 2048 tokens, batch size 1.
- Control and treatment differ only in `error_compensation`.
- GPU: NVIDIA A100-SXM4-40GB.

| Bits | Control PPL | Sequential PPL | Delta | Control quantization time | Sequential quantization time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 27.1890 | 26.8635 | -0.3255 | 78.20s | 79.66s |
| 4 | 16.7909 | 16.5875 | -0.2034 | 76.44s | 78.86s |

## Commands

```bash
scripts/run_sequential_compensation_ablation.sh both --variant sequential --gpu A100
scripts/sync_sequential_compensation_ablation.sh both

scripts/run_sequential_compensation_confirmation.sh both --variant both --gpu A100
scripts/sync_sequential_compensation_confirmation.sh both
```

## Decision Rules

- Do not tune the compensation block size or add a parameter search based on
  these final-board results.
- Keep the local scale-refined recipe as the control for future comparisons.
- Run the same fixed treatment on an alternate C4 calibration seed, then on a
  second model family before promoting it beyond the development track.
- Treat quantization time separately from inference speed; current evaluation
  remains the dense dequantized simulation path.
