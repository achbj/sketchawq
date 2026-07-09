from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sketchawq.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bits", [3, 4])
def test_nystrom_controls_preserve_the_strict_budget_protocol(bits: int) -> None:
    control = load_config(
        PROJECT_ROOT / "configs" / "experiments" / "strict_budget" / f"qwen3_4b_{bits}bit_control.yaml"
    )
    nystrom = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "spectrum_curvature"
        / f"qwen3_4b_{bits}bit_nystrom.yaml"
    )

    assert nystrom.model.repo_id == control.model.repo_id
    assert nystrom.quantization.bits == control.quantization.bits == bits
    assert nystrom.quantization.group_size == control.quantization.group_size == 128
    assert nystrom.quantization.scale_refinement == "none"
    assert nystrom.calibration == control.calibration
    assert nystrom.evaluation.dataset_name == control.evaluation.dataset_name
    assert nystrom.evaluation.dataset_config == control.evaluation.dataset_config
    assert nystrom.evaluation.split == control.evaluation.split
    assert nystrom.evaluation.sequence_length == control.evaluation.sequence_length == 2048
    assert nystrom.sketch.rank == control.sketch.rank == 64
    assert nystrom.sketch.low_rank_representation == "nystrom"
    assert nystrom.output.run_name == f"qwen3_4b_{bits}bit_nystrom_g128_control"


def test_nystrom_launcher_dry_runs_both_bits() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "run_spectrum_curvature_ablation.sh"),
            "both",
            "--gpu",
            "A100",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "SKETCHAWQ_MODAL_GPU=A100" in result.stdout
    assert "infra/modal/sketchawq_app.py" in result.stdout
    assert "qwen3_4b_3bit_nystrom.yaml" in result.stdout
    assert "qwen3_4b_4bit_nystrom.yaml" in result.stdout


def test_nystrom_sync_dry_run_fetches_frozen_controls_and_new_rows() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "sync_spectrum_curvature_ablation.sh"),
            "both",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.count("volume get") == 4
    assert "qwen3_4b_3bit_strict_g128_control" in result.stdout
    assert "qwen3_4b_3bit_nystrom_g128_control" in result.stdout
    assert "qwen3_4b_4bit_strict_g128_control" in result.stdout
    assert "qwen3_4b_4bit_nystrom_g128_control" in result.stdout
    assert "sketchawq.experiments.compare_methods" in result.stdout
