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
