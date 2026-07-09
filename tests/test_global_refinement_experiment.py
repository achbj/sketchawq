from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess

import pytest

from sketchawq.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bits", [3, 4])
def test_global_refinement_changes_only_the_refinement_policy(bits: int) -> None:
    control = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "spectrum_curvature"
        / f"qwen3_4b_{bits}bit_nystrom_residual.yaml"
    )
    treatment = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "global_refinement"
        / f"qwen3_4b_{bits}bit_global_refine.yaml"
    )

    assert treatment.quantization.scale_refinement == "global_sketch"
    assert treatment.quantization.scale_refinement_steps == 3
    assert treatment.output.run_name == f"qwen3_4b_{bits}bit_nystrom_global_refine_g128"

    control_data = asdict(control)
    treatment_data = asdict(treatment)
    del control_data["quantization"]["scale_refinement"]
    del treatment_data["quantization"]["scale_refinement"]
    del control_data["output"]["run_name"]
    del treatment_data["output"]["run_name"]
    assert treatment_data == control_data


def test_global_refinement_launcher_dry_runs_both_bits() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "run_global_refinement_ablation.sh"),
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
    assert "qwen3_4b_3bit_global_refine.yaml" in result.stdout
    assert "qwen3_4b_4bit_global_refine.yaml" in result.stdout


def test_global_refinement_sync_dry_run_fetches_controls_and_treatment() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "sync_global_refinement_ablation.sh"),
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
        assert f"qwen3_4b_{bits}bit_strict_g128_control" in result.stdout
        assert f"qwen3_4b_{bits}bit_nystrom_residual_g128_control" in result.stdout
        assert f"qwen3_4b_{bits}bit_nystrom_global_refine_g128" in result.stdout
    assert "sketchawq.experiments.compare_methods" in result.stdout
