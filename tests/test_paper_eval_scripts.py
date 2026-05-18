from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from sketchawq.baselines.config import load_baseline_config
from sketchawq.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAPER_EVAL_CONFIGS = {
    ("qwen2_5_1_5b", 3): (
        "configs/paper/qwen2_5_1_5b_3bit.yaml",
        "configs/baselines/qwen2_5_1_5b_gptq_3bit.yaml",
    ),
    ("qwen2_5_1_5b", 4): (
        "configs/paper/qwen2_5_1_5b_4bit.yaml",
        "configs/baselines/qwen2_5_1_5b_gptq_4bit.yaml",
    ),
    ("qwen3_1_7b", 3): (
        "configs/paper/qwen3_1_7b_3bit.yaml",
        "configs/baselines/qwen3_1_7b_gptq_3bit.yaml",
    ),
    ("qwen3_1_7b", 4): (
        "configs/paper/qwen3_1_7b_4bit.yaml",
        "configs/baselines/qwen3_1_7b_gptq_4bit.yaml",
    ),
    ("qwen3_4b", 3): (
        "configs/paper/qwen3_4b_3bit.yaml",
        "configs/baselines/qwen3_4b_gptq_3bit.yaml",
    ),
    ("qwen3_4b", 4): (
        "configs/paper/qwen3_4b_4bit.yaml",
        "configs/baselines/qwen3_4b_gptq_4bit.yaml",
    ),
    ("qwen3_8b", 3): (
        "configs/paper/qwen3_8b_3bit.yaml",
        "configs/baselines/qwen3_8b_gptq_3bit.yaml",
    ),
    ("qwen3_8b", 4): (
        "configs/paper/qwen3_8b_4bit.yaml",
        "configs/baselines/qwen3_8b_gptq_4bit.yaml",
    ),
    ("qwen3_14b", 3): (
        "configs/paper/qwen3_14b_3bit.yaml",
        "configs/baselines/qwen3_14b_gptq_3bit.yaml",
    ),
    ("qwen3_14b", 4): (
        "configs/paper/qwen3_14b_4bit.yaml",
        "configs/baselines/qwen3_14b_gptq_4bit.yaml",
    ),
    ("qwen3_32b", 3): (
        "configs/paper/qwen3_32b_3bit.yaml",
        "configs/baselines/qwen3_32b_gptq_3bit.yaml",
    ),
    ("qwen3_32b", 4): (
        "configs/paper/qwen3_32b_4bit.yaml",
        "configs/baselines/qwen3_32b_gptq_4bit.yaml",
    ),
    ("llama_7b", 3): (
        "configs/paper/llama_7b_3bit.yaml",
        "configs/baselines/llama_7b_gptq_3bit.yaml",
    ),
    ("llama_7b", 4): (
        "configs/paper/llama_7b_4bit.yaml",
        "configs/baselines/llama_7b_gptq_4bit.yaml",
    ),
    ("mistral_7b", 3): (
        "configs/paper/mistral_7b_3bit.yaml",
        "configs/baselines/mistral_7b_gptq_3bit.yaml",
    ),
    ("mistral_7b", 4): (
        "configs/paper/mistral_7b_4bit.yaml",
        "configs/baselines/mistral_7b_gptq_4bit.yaml",
    ),
}

QWEN3_8B_C4_EVAL_CONFIGS = {
    3: (
        "configs/paper/qwen3_8b_3bit_c4_eval.yaml",
        "configs/baselines/qwen3_8b_gptq_3bit_c4_eval.yaml",
    ),
    4: (
        "configs/paper/qwen3_8b_4bit_c4_eval.yaml",
        "configs/baselines/qwen3_8b_gptq_4bit_c4_eval.yaml",
    ),
}


def test_every_model_bit_has_valid_sketchawq_and_gptq_configs() -> None:
    seen_run_names: set[str] = set()
    for sketchawq_config, gptq_config in PAPER_EVAL_CONFIGS.values():
        sketchawq = load_config(PROJECT_ROOT / sketchawq_config)
        baseline = load_baseline_config(PROJECT_ROOT / gptq_config)
        assert baseline.method.normalized_name() == "gptq"
        assert sketchawq.model.repo_id == baseline.model.repo_id
        assert sketchawq.quantization.bits == baseline.quantization.bits
        assert sketchawq.calibration.dataset_name == baseline.calibration.dataset_name
        assert sketchawq.calibration.dataset_config == baseline.calibration.dataset_config
        assert sketchawq.calibration.split == baseline.calibration.split
        assert sketchawq.calibration.text_column == baseline.calibration.text_column
        assert sketchawq.calibration.num_samples == baseline.calibration.num_samples
        assert sketchawq.calibration.sequence_length == baseline.calibration.sequence_length
        assert sketchawq.calibration.packing == baseline.calibration.packing
        assert sketchawq.evaluation.benchmark_name == baseline.evaluation.benchmark_name
        assert sketchawq.evaluation.dataset_name == baseline.evaluation.dataset_name
        assert sketchawq.evaluation.dataset_config == baseline.evaluation.dataset_config
        assert sketchawq.evaluation.split == baseline.evaluation.split
        assert sketchawq.evaluation.text_column == baseline.evaluation.text_column
        assert sketchawq.evaluation.num_samples == baseline.evaluation.num_samples
        assert sketchawq.evaluation.sequence_length == baseline.evaluation.sequence_length
        assert sketchawq.evaluation.packing == baseline.evaluation.packing
        assert sketchawq.evaluation.max_batches == baseline.evaluation.max_batches
        assert baseline.evaluation.benchmark_name.startswith("wikitext2_validation")
        if baseline.evaluation.benchmark_name == "wikitext2_validation_packed_eval2048":
            assert baseline.evaluation.num_samples is None
            assert baseline.evaluation.max_batches is None
        else:
            assert baseline.evaluation.max_batches in {64, 128, 256}
        assert baseline.evaluation.measure_throughput
        assert sketchawq.output.run_name not in seen_run_names
        seen_run_names.add(sketchawq.output.run_name)
        assert baseline.output.run_name not in seen_run_names
        seen_run_names.add(baseline.output.run_name)


def test_qwen3_8b_c4_eval_configs_match_across_methods() -> None:
    for bits, (sketchawq_config, gptq_config) in QWEN3_8B_C4_EVAL_CONFIGS.items():
        sketchawq = load_config(PROJECT_ROOT / sketchawq_config)
        baseline = load_baseline_config(PROJECT_ROOT / gptq_config)

        assert baseline.method.normalized_name() == "gptq"
        assert sketchawq.model.repo_id == "Qwen/Qwen3-8B"
        assert sketchawq.model.repo_id == baseline.model.repo_id
        assert sketchawq.quantization.bits == bits
        assert sketchawq.quantization.bits == baseline.quantization.bits
        assert sketchawq.calibration.dataset_name == "allenai/c4"
        assert sketchawq.calibration.dataset_name == baseline.calibration.dataset_name
        assert sketchawq.calibration.sequence_length == baseline.calibration.sequence_length
        assert sketchawq.evaluation.benchmark_name == "c4_validation_packed_eval2048"
        assert sketchawq.evaluation.dataset_name == "allenai/c4"
        assert sketchawq.evaluation.dataset_config == "en"
        assert sketchawq.evaluation.dataset_name == baseline.evaluation.dataset_name
        assert sketchawq.evaluation.dataset_config == baseline.evaluation.dataset_config
        assert sketchawq.evaluation.split == baseline.evaluation.split == "validation"
        assert sketchawq.evaluation.sequence_length == baseline.evaluation.sequence_length == 2048
        assert sketchawq.evaluation.max_batches == baseline.evaluation.max_batches == 64
        assert sketchawq.evaluation.measure_throughput
        assert baseline.evaluation.measure_throughput


def test_qwen3_1_7b_check_configs_use_1024_by_256_budget() -> None:
    for bits, (sketchawq_config, gptq_config) in {
        3: PAPER_EVAL_CONFIGS[("qwen3_1_7b", 3)],
        4: PAPER_EVAL_CONFIGS[("qwen3_1_7b", 4)],
    }.items():
        sketchawq = load_config(PROJECT_ROOT / sketchawq_config)
        baseline = load_baseline_config(PROJECT_ROOT / gptq_config)

        assert sketchawq.model.repo_id == baseline.model.repo_id == "Qwen/Qwen3-1.7B-Base"
        assert sketchawq.quantization.bits == baseline.quantization.bits == bits
        assert sketchawq.calibration.dataset_name == baseline.calibration.dataset_name == "allenai/c4"
        assert sketchawq.calibration.num_samples == baseline.calibration.num_samples == 256
        assert sketchawq.calibration.sequence_length == baseline.calibration.sequence_length == 1024
        assert sketchawq.calibration.packing == baseline.calibration.packing is True
        assert sketchawq.evaluation.benchmark_name == baseline.evaluation.benchmark_name
        assert sketchawq.evaluation.benchmark_name == "wikitext2_validation_packed_eval1024"
        assert sketchawq.evaluation.num_samples == baseline.evaluation.num_samples == 256
        assert sketchawq.evaluation.sequence_length == baseline.evaluation.sequence_length == 1024
        assert sketchawq.evaluation.max_batches == baseline.evaluation.max_batches == 256
        assert "_check_" in sketchawq.output.run_name
        assert "_check_" in baseline.output.run_name
        assert sketchawq.output.run_name.endswith("eval1024")
        assert baseline.output.run_name.endswith("eval1024")


def test_mistral_7b_configs_use_official_base_model_and_full_eval() -> None:
    for bits, (sketchawq_config, gptq_config) in {
        3: PAPER_EVAL_CONFIGS[("mistral_7b", 3)],
        4: PAPER_EVAL_CONFIGS[("mistral_7b", 4)],
    }.items():
        sketchawq = load_config(PROJECT_ROOT / sketchawq_config)
        baseline = load_baseline_config(PROJECT_ROOT / gptq_config)

        assert sketchawq.model.repo_id == baseline.model.repo_id == "mistralai/Mistral-7B-v0.3"
        assert sketchawq.model.dtype == baseline.model.dtype == "bfloat16"
        assert sketchawq.quantization.bits == baseline.quantization.bits == bits
        assert sketchawq.calibration.dataset_name == baseline.calibration.dataset_name == "allenai/c4"
        assert sketchawq.calibration.num_samples == baseline.calibration.num_samples == 256
        assert sketchawq.calibration.sequence_length == baseline.calibration.sequence_length == 2048
        assert sketchawq.calibration.packing == baseline.calibration.packing is True
        assert sketchawq.evaluation.benchmark_name == baseline.evaluation.benchmark_name
        assert sketchawq.evaluation.benchmark_name == "wikitext2_validation_packed_eval2048"
        assert sketchawq.evaluation.num_samples == baseline.evaluation.num_samples is None
        assert sketchawq.evaluation.max_batches == baseline.evaluation.max_batches is None
        assert sketchawq.evaluation.sequence_length == baseline.evaluation.sequence_length == 2048
        assert sketchawq.evaluation.measure_throughput
        assert baseline.evaluation.measure_throughput


def test_every_model_bit_has_wrapper_script() -> None:
    for model, bits in PAPER_EVAL_CONFIGS:
        script = PROJECT_ROOT / "scripts" / "paper_eval" / f"{model}_{bits}bit.sh"
        assert script.exists()
        assert script.stat().st_mode & 0o111


def test_wrapper_can_dry_run_gptq_for_one_model() -> None:
    script = PROJECT_ROOT / "scripts" / "paper_eval" / "qwen3_14b_3bit.sh"

    result = subprocess.run(
        [str(script), "--method", "gptq", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "infra/modal/baseline_app.py" in result.stdout
    assert "configs/baselines/qwen3_14b_gptq_3bit.yaml" in result.stdout


def test_generic_script_can_dry_run_both_methods_locally() -> None:
    script = PROJECT_ROOT / "scripts" / "run_paper_eval.sh"

    result = subprocess.run(
        [str(script), "qwen3_32b", "4", "--method", "both", "--backend", "local", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "python -m sketchawq.cli run configs/paper/qwen3_32b_4bit.yaml" in result.stdout
    assert (
        "python -m sketchawq.cli baseline configs/baselines/qwen3_32b_gptq_4bit.yaml"
        in result.stdout
    )


def test_generic_script_can_dry_run_qwen3_1_7b_both_methods() -> None:
    script = PROJECT_ROOT / "scripts" / "run_paper_eval.sh"

    result = subprocess.run(
        [str(script), "qwen3_1_7b", "3", "--method", "both", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "configs/paper/qwen3_1_7b_3bit.yaml" in result.stdout
    assert "configs/baselines/qwen3_1_7b_gptq_3bit.yaml" in result.stdout


def test_generic_script_can_dry_run_modal_gpu_override() -> None:
    script = PROJECT_ROOT / "scripts" / "run_paper_eval.sh"

    result = subprocess.run(
        [str(script), "qwen3_8b", "3", "--method", "both", "--gpu", "L40S", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "SKETCHAWQ_MODAL_GPU=L40S modal run infra/modal/sketchawq_app.py" in result.stdout
    assert "SKETCHAWQ_MODAL_GPU=L40S modal run infra/modal/baseline_app.py" in result.stdout


def test_generic_script_can_dry_run_c4_eval_for_qwen3_8b() -> None:
    script = PROJECT_ROOT / "scripts" / "run_paper_eval.sh"

    result = subprocess.run(
        [str(script), "qwen3_8b", "3", "--method", "both", "--eval", "c4", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "configs/paper/qwen3_8b_3bit_c4_eval.yaml" in result.stdout
    assert "configs/baselines/qwen3_8b_gptq_3bit_c4_eval.yaml" in result.stdout


def test_generic_script_rejects_gpu_override_for_local_backend() -> None:
    script = PROJECT_ROOT / "scripts" / "run_paper_eval.sh"

    result = subprocess.run(
        [str(script), "qwen3_8b", "3", "--local", "--gpu", "L40S", "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--gpu only applies to the modal backend" in result.stderr


def test_sync_results_script_dry_runs_qwen3_8b_both_bits_and_methods() -> None:
    script = PROJECT_ROOT / "scripts" / "sync_paper_eval_results.sh"

    result = subprocess.run(
        [str(script), "qwen3_8b", "both", "--method", "both", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "modal volume get --force sketchawq-artifacts /qwen3_8b_3bit_qwen_gptq_fair_packed_eval2048" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /qwen3_8b_gptq_3bit_eval2048" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /qwen3_8b_4bit_qwen_final_packed_eval2048" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /qwen3_8b_gptq_4bit_eval2048" in result.stdout
    assert "sketchawq.experiments.compare_methods" in result.stdout
    assert "sketchawq.experiments.plot_benchmark_board" in result.stdout
    assert "sketchawq.experiments.plot_search_choices" in result.stdout
    assert "qwen3_8b_3_4bit_paper_eval_summary.csv" in result.stdout
    assert re.search(
        r"artifacts/paper_eval/qwen3_8b_3_4bit_paper_eval_\d{8}_\d{6}/"
        r"qwen3_8b_3_4bit_paper_eval_summary\.csv",
        result.stdout,
    )


def test_sync_results_script_dry_runs_qwen3_1_7b_both_bits_and_methods() -> None:
    script = PROJECT_ROOT / "scripts" / "sync_paper_eval_results.sh"

    result = subprocess.run(
        [str(script), "qwen3_1_7b", "both", "--method", "both", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "modal volume get --force sketchawq-artifacts /qwen3_1_7b_3bit_qwen_check_packed_eval1024" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /qwen3_1_7b_gptq_3bit_check_eval1024" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /qwen3_1_7b_4bit_qwen_check_packed_eval1024" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /qwen3_1_7b_gptq_4bit_check_eval1024" in result.stdout
    assert "qwen3_1_7b_3_4bit_paper_eval_summary.csv" in result.stdout


def test_run_script_dry_runs_mistral_7b_both_methods() -> None:
    script = PROJECT_ROOT / "scripts" / "run_paper_eval.sh"

    result = subprocess.run(
        [str(script), "mistral_7b", "3", "--method", "both", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "modal run infra/modal/sketchawq_app.py --config configs/paper/mistral_7b_3bit.yaml" in result.stdout
    assert (
        "modal run infra/modal/baseline_app.py --config configs/baselines/mistral_7b_gptq_3bit.yaml"
        in result.stdout
    )


def test_sync_results_script_dry_runs_mistral_7b_both_bits_and_methods() -> None:
    script = PROJECT_ROOT / "scripts" / "sync_paper_eval_results.sh"

    result = subprocess.run(
        [str(script), "mistral_7b", "both", "--method", "both", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "modal volume get --force sketchawq-artifacts /mistral_7b_3bit_mistral_final_packed_eval2048" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /mistral_7b_gptq_3bit_eval2048" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /mistral_7b_4bit_mistral_final_packed_eval2048" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /mistral_7b_gptq_4bit_eval2048" in result.stdout
    assert "mistral_7b_3_4bit_paper_eval_summary.csv" in result.stdout


def test_sync_results_script_can_skip_plot_generation() -> None:
    script = PROJECT_ROOT / "scripts" / "sync_paper_eval_results.sh"

    result = subprocess.run(
        [str(script), "qwen3_8b", "3", "--method", "gptq", "--no-plots", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "modal volume get --force sketchawq-artifacts /qwen3_8b_gptq_3bit_eval2048" in result.stdout
    assert "sketchawq.experiments.plot_benchmark_board" not in result.stdout
    assert "sketchawq.experiments.plot_search_choices" not in result.stdout


def test_sync_results_script_uses_unique_readable_bundle_dirs(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "sync_paper_eval_results.sh"
    modal_root = tmp_path / "modal"
    results_root = tmp_path / "paper_eval"
    timestamp = "20260517_143000"
    existing = results_root / f"qwen3_4b_3_4bit_paper_eval_{timestamp}"
    existing.mkdir(parents=True)

    result = subprocess.run(
        [
            str(script),
            "qwen3_4b",
            "both",
            "--method",
            "both",
            "--output-dir",
            str(modal_root),
            "--results-dir",
            str(results_root),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "SKETCHAWQ_SYNC_TIMESTAMP": timestamp},
    )

    bundle = f"qwen3_4b_3_4bit_paper_eval_{timestamp}_02"
    assert f"{modal_root}/{bundle}/runs/qwen3_4b_3bit_qwen_gptq_fair_balanced_packed_eval2048" in result.stdout
    assert f"{results_root}/{bundle}/qwen3_4b_3_4bit_paper_eval_summary.csv" in result.stdout
    assert "sketchawq.experiments.plot_search_choices" in result.stdout
    assert "--name qwen3_4b_search_choices" in result.stdout


def test_sync_results_script_dry_runs_qwen3_8b_c4_eval() -> None:
    script = PROJECT_ROOT / "scripts" / "sync_paper_eval_results.sh"

    result = subprocess.run(
        [str(script), "qwen3_8b", "both", "--method", "both", "--eval", "c4", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "modal volume get --force sketchawq-artifacts /qwen3_8b_3bit_qwen_gptq_fair_c4_eval2048" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /qwen3_8b_gptq_3bit_c4_eval2048" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /qwen3_8b_4bit_qwen_final_c4_eval2048" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /qwen3_8b_gptq_4bit_c4_eval2048" in result.stdout
    assert "--benchmark c4_validation_packed_eval2048" in result.stdout
    assert "qwen3_8b_3_4bit_c4_eval_summary.csv" in result.stdout


def test_qwen3_8b_paper_configs_match_gptq_fair_comparison_shape() -> None:
    three_bit = load_config(PROJECT_ROOT / "configs/paper/qwen3_8b_3bit.yaml")
    four_bit = load_config(PROJECT_ROOT / "configs/paper/qwen3_8b_4bit.yaml")

    assert three_bit.quantization.group_size == 128
    assert three_bit.quantization.search_objective == "weighted_error"
    assert three_bit.quantization.output_error_rows == 0
    assert three_bit.quantization.awq_alpha_candidates == [0.0, 0.125, 0.25]
    assert three_bit.quantization.clip_candidates == [0.85, 0.82, 0.8]
    assert four_bit.quantization.group_size == 128
    assert four_bit.quantization.search_objective == "weighted_error"
    assert four_bit.quantization.awq_alpha_candidates == [0.0, 0.1, 0.2, 0.3]
    assert four_bit.quantization.clip_candidates == [0.93, 0.9, 0.87, 0.85]


def test_qwen3_4b_paper_configs_use_balanced_fair_comparison_shape() -> None:
    three_bit = load_config(PROJECT_ROOT / "configs/paper/qwen3_4b_3bit.yaml")
    four_bit = load_config(PROJECT_ROOT / "configs/paper/qwen3_4b_4bit.yaml")
    three_bit_baseline = load_baseline_config(
        PROJECT_ROOT / "configs/baselines/qwen3_4b_gptq_3bit.yaml"
    )
    four_bit_baseline = load_baseline_config(
        PROJECT_ROOT / "configs/baselines/qwen3_4b_gptq_4bit.yaml"
    )

    assert three_bit.quantization.group_size == three_bit_baseline.quantization.group_size == 128
    assert three_bit.quantization.scale_clip_search == "greedy"
    assert three_bit.quantization.search_objective == "hybrid"
    assert three_bit.quantization.output_error_rows == 64
    assert three_bit.quantization.output_error_weight == 0.5
    assert three_bit.quantization.awq_alpha_candidates == [0.0, 0.0875, 0.125, 0.175, 0.2, 0.25]
    assert three_bit.quantization.clip_candidates == [0.8, 0.79, 0.78, 0.77]
    assert three_bit.sketch.rank == 64
    assert three_bit.calibration.batch_size == 8
    assert four_bit.quantization.group_size == four_bit_baseline.quantization.group_size == 128
    assert four_bit.quantization.scale_clip_search == "greedy"
    assert four_bit.quantization.search_objective == "hybrid"
    assert four_bit.quantization.output_error_rows == 32
    assert four_bit.quantization.output_error_weight == 0.5
    assert four_bit.quantization.awq_alpha_candidates == [0.0, 0.0875, 0.125, 0.175, 0.2, 0.25]
    assert four_bit.quantization.clip_candidates == [0.88, 0.87, 0.86, 0.85]
    assert four_bit.sketch.rank == 64
    assert four_bit.calibration.batch_size == 8
    assert four_bit.sketch.normalize_error_terms is True


def test_qwen3_4b_best_configs_keep_accuracy_oriented_shape() -> None:
    three_bit = load_config(PROJECT_ROOT / "configs/paper/qwen3_4b_3bit_best.yaml")
    four_bit = load_config(PROJECT_ROOT / "configs/paper/qwen3_4b_4bit_best.yaml")

    assert three_bit.quantization.group_size == 64
    assert three_bit.quantization.search_objective == "hybrid"
    assert three_bit.quantization.output_error_rows == 64
    assert three_bit.output.run_name == "qwen3_4b_3bit_qwen_best_packed_eval2048"
    assert four_bit.quantization.group_size == 64
    assert four_bit.quantization.search_objective == "weighted_error"
    assert four_bit.output.run_name == "qwen3_4b_4bit_qwen_best_packed_eval2048"


def test_sync_results_script_dry_runs_qwen3_4b_fair_configs() -> None:
    script = PROJECT_ROOT / "scripts" / "sync_paper_eval_results.sh"

    result = subprocess.run(
        [str(script), "qwen3_4b", "both", "--method", "both", "--dry-run", "--no-plots"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "modal volume get --force sketchawq-artifacts /qwen3_4b_3bit_qwen_gptq_fair_balanced_packed_eval2048" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /qwen3_4b_gptq_3bit_g128_eval2048" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /qwen3_4b_4bit_qwen_gptq_fair_balanced_packed_eval2048" in result.stdout
    assert "modal volume get --force sketchawq-artifacts /qwen3_4b_gptq_4bit_g128_eval2048" in result.stdout
