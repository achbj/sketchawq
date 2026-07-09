from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess

import pytest

from sketchawq.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("variant", ["control", "refine"])
def test_selection_configs_change_only_the_board_protocol(bits: int, variant: str) -> None:
    strict = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "strict_budget"
        / f"qwen3_4b_{bits}bit_{variant}.yaml"
    )
    selection = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "model_selection"
        / f"qwen3_4b_{bits}bit_{variant}.yaml"
    )

    assert selection.model_selection.enabled is True
    assert selection.model_selection.benchmark_name == "c4_validation_packed_selection2048"
    assert selection.calibration.holdout_samples == 32
    assert selection.calibration.holdout_split == "validation"
    assert selection.evaluation.enabled is False
    assert selection.output.run_name == f"qwen3_4b_{bits}bit_strict_g128_{variant}_selection"

    strict_data = asdict(strict)
    selection_data = asdict(selection)
    for data in (strict_data, selection_data):
        del data["calibration"]["holdout_samples"]
        del data["calibration"]["holdout_split"]
        del data["model_selection"]
        del data["evaluation"]["enabled"]
        del data["output"]["run_name"]
    assert selection_data == strict_data


def test_model_selection_launcher_dry_runs_all_w3_w4_rows() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "run_model_selection_ablation.sh"),
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
        for variant in ("control", "refine"):
            assert f"qwen3_4b_{bits}bit_{variant}.yaml" in result.stdout


def test_model_selection_sync_dry_run_collects_selection_metrics() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "sync_model_selection_ablation.sh"),
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
        for variant in ("control", "refine"):
            assert f"qwen3_4b_{bits}bit_strict_g128_{variant}_selection" in result.stdout
    assert "--table model-selection" in result.stdout
