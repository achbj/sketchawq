from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path

from sketchawq.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    (3, "control"): "configs/experiments/strict_budget/qwen3_4b_3bit_control.yaml",
    (3, "refine"): "configs/experiments/strict_budget/qwen3_4b_3bit_refine.yaml",
    (4, "control"): "configs/experiments/strict_budget/qwen3_4b_4bit_control.yaml",
    (4, "refine"): "configs/experiments/strict_budget/qwen3_4b_4bit_refine.yaml",
}


def test_strict_budget_configs_are_a_controlled_refinement_ablation() -> None:
    for bits in (3, 4):
        control_path = PROJECT_ROOT / CONFIGS[(bits, "control")]
        refine_path = PROJECT_ROOT / CONFIGS[(bits, "refine")]
        assert "#" not in control_path.read_text(encoding="utf-8")
        assert "#" not in refine_path.read_text(encoding="utf-8")

        control = load_config(control_path)
        refine = load_config(refine_path)

        for config in (control, refine):
            assert config.model.repo_id == "Qwen/Qwen3-4B"
            assert config.quantization.bits == bits
            assert config.quantization.group_size == 128
            assert config.quantization.symmetric is True
            assert config.outliers.active() is False
            assert config.adaptive.enabled is False
            assert config.calibration.dataset_name == "allenai/c4"
            assert config.calibration.num_samples == 256
            assert config.calibration.sequence_length == 2048
            assert config.evaluation.dataset_name == "Salesforce/wikitext"
            assert config.evaluation.split == "validation"
            assert config.evaluation.sequence_length == 2048

        assert control.quantization.scale_refinement == "none"
        assert refine.quantization.scale_refinement == "sketch"
        assert refine.quantization.scale_refinement_steps == 3

        control_data = asdict(control)
        refine_data = asdict(refine)
        del control_data["quantization"]["scale_refinement"]
        del refine_data["quantization"]["scale_refinement"]
        del control_data["output"]["run_name"]
        del refine_data["output"]["run_name"]
        assert control_data == refine_data


def test_strict_budget_launcher_dry_runs_full_ablation() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "run_strict_budget_ablation.sh"),
            "both",
            "--variant",
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
    assert "modal run" in result.stdout
    assert "infra/modal/sketchawq_app.py" in result.stdout
    for config_path in CONFIGS.values():
        assert config_path in result.stdout


def test_strict_budget_launcher_can_select_one_local_row() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "run_strict_budget_ablation.sh"),
            "4",
            "--variant",
            "refine",
            "--local",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "python -m sketchawq.cli run" in result.stdout
    assert CONFIGS[(4, "refine")] in result.stdout
    assert CONFIGS[(4, "control")] not in result.stdout
    assert CONFIGS[(3, "refine")] not in result.stdout


def test_strict_budget_sync_dry_run_downloads_and_compares_all_rows() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "sync_strict_budget_ablation.sh"),
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
    assert "qwen3_4b_3bit_strict_g128_refine" in result.stdout
    assert "qwen3_4b_4bit_strict_g128_control" in result.stdout
    assert "qwen3_4b_4bit_strict_g128_refine" in result.stdout
    assert "sketchawq.experiments.compare_methods" in result.stdout
