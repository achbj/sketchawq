from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess

import pytest

from sketchawq.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bits", [3, 4])
def test_holdout_guard_changes_only_the_guard_and_holdout_protocol(bits: int) -> None:
    global_treatment = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "global_refinement"
        / f"qwen3_4b_{bits}bit_global_refine.yaml"
    )
    guarded = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "holdout_guard"
        / f"qwen3_4b_{bits}bit_global_holdout_guard.yaml"
    )

    assert guarded.quantization.scale_refinement == "global_sketch"
    assert guarded.quantization.scale_refinement_guard == "holdout_output"
    assert guarded.calibration.holdout_samples == 32
    assert guarded.calibration.holdout_split == "validation"
    assert guarded.output.run_name == f"qwen3_4b_{bits}bit_nystrom_global_holdout_guard_g128"

    treatment_data = asdict(global_treatment)
    guarded_data = asdict(guarded)
    for data in (treatment_data, guarded_data):
        del data["quantization"]["scale_refinement_guard"]
        del data["calibration"]["holdout_samples"]
        del data["calibration"]["holdout_split"]
        del data["output"]["run_name"]
    assert guarded_data == treatment_data


def test_holdout_guard_launcher_dry_runs_both_bits() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "run_holdout_guard_ablation.sh"),
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
    assert "qwen3_4b_3bit_global_holdout_guard.yaml" in result.stdout
    assert "qwen3_4b_4bit_global_holdout_guard.yaml" in result.stdout


def test_holdout_guard_sync_dry_run_fetches_all_comparison_rows() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "sync_holdout_guard_ablation.sh"),
            "both",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.count("volume get") == 6
    for bits in (3, 4):
        assert f"qwen3_4b_{bits}bit_nystrom_residual_g128_control" in result.stdout
        assert f"qwen3_4b_{bits}bit_nystrom_global_refine_g128" in result.stdout
        assert f"qwen3_4b_{bits}bit_nystrom_global_holdout_guard_g128" in result.stdout
