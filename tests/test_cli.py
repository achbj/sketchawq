from __future__ import annotations

from pathlib import Path

from sketchawq.cli import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_list_targets_prints_catalog(capsys) -> None:
    exit_code = main(["list-targets", "--catalog", str(PROJECT_ROOT / "configs/experiments.yaml")])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "smoke [sketchawq]" in output
    assert "baseline-qwen2 [baseline]" in output


def test_inference_command_delegates_to_benchmark(monkeypatch, tmp_path, capsys) -> None:
    captured = {}

    def fake_benchmark_from_config(config_path, **kwargs):
        captured["config_path"] = config_path
        captured.update(kwargs)
        return {"output_dir": str(tmp_path / "tiny_inference")}

    monkeypatch.setattr(
        "sketchawq.experiments.benchmark_inference.benchmark_from_config",
        fake_benchmark_from_config,
    )

    exit_code = main(
        [
            "inference",
            "configs/paper/qwen3_4b_4bit.yaml",
            "--variant",
            "fp16",
            "--prompt-lengths",
            "16",
            "32",
            "--batch-sizes",
            "1",
            "2",
            "--generated-tokens",
            "4",
            "--warmup-runs",
            "0",
            "--repeats",
            "2",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured["config_path"] == "configs/paper/qwen3_4b_4bit.yaml"
    assert captured["prompt_lengths"] == [16, 32]
    assert captured["batch_sizes"] == [1, 2]
    assert captured["generated_tokens"] == 4
    assert captured["warmup_runs"] == 0
    assert captured["repeats"] == 2
    assert "completed: tiny_inference" in capsys.readouterr().out


def test_inference_summary_command_delegates(monkeypatch, tmp_path, capsys) -> None:
    captured = {}

    def fake_summarize(inputs, **kwargs):
        captured["inputs"] = inputs
        captured.update(kwargs)
        return {
            "csv": str(tmp_path / "summary.csv"),
            "json": str(tmp_path / "summary.json"),
            "html": str(tmp_path / "summary.html"),
        }

    monkeypatch.setattr(
        "sketchawq.experiments.summarize_inference.summarize_inference_artifacts",
        fake_summarize,
    )

    exit_code = main(
        [
            "inference-summary",
            "artifacts/modal/inference/inference",
            "--output-dir",
            str(tmp_path),
            "--name",
            "qwen3_4b_a100",
            "--model",
            "Qwen/Qwen3-4B",
        ]
    )

    assert exit_code == 0
    assert captured["inputs"] == ["artifacts/modal/inference/inference"]
    assert captured["output_dir"] == str(tmp_path)
    assert captured["name"] == "qwen3_4b_a100"
    assert captured["model"] == "Qwen/Qwen3-4B"
    assert "summary.csv" in capsys.readouterr().out
