from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sketchawq.baselines.config import load_baseline_config
from sketchawq.baselines.runner import _validate_external_baseline_runtime_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OPT_BASELINE_CONFIGS = {
    ("gptq", 3): "configs/baselines/opt_2_7b_gptq_3bit.yaml",
    ("gptq", 4): "configs/baselines/opt_2_7b_gptq_4bit.yaml",
    ("awq", 3): "configs/baselines/opt_2_7b_awq_3bit.yaml",
    ("awq", 4): "configs/baselines/opt_2_7b_awq_4bit.yaml",
}


def test_opt_2_7b_gptq_and_awq_configs_are_paper_like() -> None:
    seen_run_names: set[str] = set()

    for (method, bits), config_path in OPT_BASELINE_CONFIGS.items():
        config = load_baseline_config(PROJECT_ROOT / config_path)

        assert config.method.normalized_name() == method
        assert config.model.repo_id == "facebook/opt-2.7b"
        assert config.model.dtype == "float16"
        assert config.quantization.bits == bits
        assert config.quantization.group_size == 128
        assert config.quantization.symmetric is (method == "gptq")
        assert config.calibration.dataset_name == "allenai/c4"
        assert config.calibration.dataset_config == "en"
        assert config.calibration.split == "train"
        assert config.calibration.num_samples == 128
        assert config.calibration.sequence_length == 2048
        assert config.calibration.packing is True
        assert config.evaluation.benchmark_name == "wikitext2_test_packed_eval2048"
        assert config.evaluation.dataset_name == "Salesforce/wikitext"
        assert config.evaluation.dataset_config == "wikitext-2-raw-v1"
        assert config.evaluation.split == "test"
        assert config.evaluation.num_samples is None
        assert config.evaluation.sequence_length == 2048
        assert config.evaluation.packing is True
        assert config.evaluation.packing_strategy == "concat"
        assert config.evaluation.max_batches is None
        assert config.evaluation.measure_throughput is True
        assert config.output.run_name not in seen_run_names
        seen_run_names.add(config.output.run_name)


def test_gptqmodel_awq_torch_and_llmcompressor_awq_reject_3_bit() -> None:
    awq_3bit = load_baseline_config(PROJECT_ROOT / OPT_BASELINE_CONFIGS[("awq", 3)])
    with pytest.raises(RuntimeError, match="awq_torch backend only supports 4-bit"):
        _validate_external_baseline_runtime_support(awq_3bit)

    llmcompressor_3bit = load_baseline_config(PROJECT_ROOT / OPT_BASELINE_CONFIGS[("awq", 3)])
    llmcompressor_3bit.method.name = "llmcompressor_awq"
    with pytest.raises(RuntimeError, match="only supports 4-bit"):
        _validate_external_baseline_runtime_support(llmcompressor_3bit)


def test_opt_2_7b_awq_4bit_uses_gptqmodel_auto_backend() -> None:
    config = load_baseline_config(PROJECT_ROOT / OPT_BASELINE_CONFIGS[("awq", 4)])

    assert config.method.adapter_kwargs["backend"] == "auto"


def test_opt_baseline_script_dry_runs_supported_modal_rows() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "run_opt_baselines.sh"),
            "both",
            "--method",
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

    assert "SKETCHAWQ_MODAL_GPU=A100 modal run infra/modal/baseline_app.py" in result.stdout
    for config_path in {
        OPT_BASELINE_CONFIGS[("gptq", 3)],
        OPT_BASELINE_CONFIGS[("gptq", 4)],
        OPT_BASELINE_CONFIGS[("awq", 4)],
    }:
        assert f"--config {config_path}" in result.stdout
    assert OPT_BASELINE_CONFIGS[("awq", 3)] not in result.stdout
    assert "skipping AWQ 3-bit" in result.stderr


def test_opt_baseline_script_dry_runs_one_local_row() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "run_opt_baselines.sh"),
            "4",
            "--method",
            "awq",
            "--local",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "python -m sketchawq.cli baseline configs/baselines/opt_2_7b_awq_4bit.yaml"
        in result.stdout
    )
    assert "opt_2_7b_awq_3bit.yaml" not in result.stdout


def test_opt_baseline_script_can_explicitly_dry_run_experimental_awq_3bit() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "run_opt_baselines.sh"),
            "3",
            "--method",
            "awq",
            "--include-awq-3bit",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"--config {OPT_BASELINE_CONFIGS[('awq', 3)]}" in result.stdout


def test_opt_baseline_script_rejects_gpu_override_for_local_backend() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts" / "run_opt_baselines.sh"),
            "4",
            "--method",
            "gptq",
            "--local",
            "--gpu",
            "A100",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--gpu only applies to the modal backend" in result.stderr
