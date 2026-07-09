from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess

import pytest

from sketchawq.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bits", [3, 4])
def test_sequential_treatment_changes_only_error_compensation(bits: int) -> None:
    control = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "model_selection"
        / f"qwen3_4b_{bits}bit_refine.yaml"
    )
    treatment = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "sequential_compensation"
        / f"qwen3_4b_{bits}bit_refine_sequential.yaml"
    )

    assert treatment.quantization.error_compensation == "sequential_sketch"
    assert treatment.output.run_name == (
        f"qwen3_4b_{bits}bit_strict_g128_refine_sequential_selection"
    )

    control_data = asdict(control)
    treatment_data = asdict(treatment)
    for data in (control_data, treatment_data):
        del data["quantization"]["error_compensation"]
        del data["output"]["run_name"]
    assert treatment_data == control_data


def test_sequential_launcher_dry_runs_both_widths() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "run_sequential_compensation_ablation.sh"),
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
    for bits in (3, 4):
        assert f"qwen3_4b_{bits}bit_refine.yaml" in result.stdout
        assert f"qwen3_4b_{bits}bit_refine_sequential.yaml" in result.stdout


def test_sequential_sync_dry_run_collects_holdout_selection_metrics() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "sync_sequential_compensation_ablation.sh"),
            "both",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.count("volume get") == 4
    for bits in (3, 4):
        assert f"qwen3_4b_{bits}bit_strict_g128_refine_selection" in result.stdout
        assert f"qwen3_4b_{bits}bit_strict_g128_refine_sequential_selection" in result.stdout
    assert "--table model-selection" in result.stdout
