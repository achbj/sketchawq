from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sketchawq.experiments.plot_search_choices import (
    build_choice_rows,
    summarize_choices,
    write_choice_csvs,
    write_choice_plots,
)


def _sample_rows() -> list[dict[str, str]]:
    return [
        {
            "run_name": "qwen3_8b_3bit",
            "bits": "3",
            "group_size": "64",
            "layer_name": "model.layers.0.self_attn.q_proj",
            "awq_alpha": "0.0",
            "clip_ratio": "0.8",
            "search_score": "0.1",
        },
        {
            "run_name": "qwen3_8b_3bit",
            "bits": "3",
            "group_size": "64",
            "layer_name": "model.layers.0.mlp.down_proj",
            "awq_alpha": "0.125",
            "clip_ratio": "0.82",
            "search_score": "0.2",
        },
        {
            "run_name": "qwen3_8b_4bit",
            "bits": "4",
            "group_size": "128",
            "layer_name": "model.layers.0.self_attn.v_proj",
            "awq_alpha": "0.1",
            "clip_ratio": "0.87",
            "search_score": "0.01",
        },
        {
            "run_name": "qwen3_8b_4bit",
            "bits": "4",
            "group_size": "128",
            "layer_name": "model.layers.0.mlp.up_proj",
            "awq_alpha": "0.0",
            "clip_ratio": "0.87",
            "search_score": "0.02",
        },
    ]


def test_choice_rows_add_layer_type_and_numeric_strings() -> None:
    rows = build_choice_rows(_sample_rows())
    by_layer = {row["layer_name"]: row for row in rows}

    q_proj = by_layer["model.layers.0.self_attn.q_proj"]
    down_proj = by_layer["model.layers.0.mlp.down_proj"]

    assert q_proj["layer_type"] == "q_proj"
    assert down_proj["layer_type"] == "down_proj"
    assert q_proj["awq_alpha"] == "0"
    assert down_proj["clip_ratio"] == "0.82"


def test_summarize_choices_reports_top_alpha_and_clip() -> None:
    summary = summarize_choices(build_choice_rows(_sample_rows()))

    three_bit = next(row for row in summary if row["bits"] == "3")
    four_bit = next(row for row in summary if row["bits"] == "4")

    assert three_bit["selected_awq_alphas"] == "0:1; 0.125:1"
    assert three_bit["selected_clip_ratios"] == "0.8:1; 0.82:1"
    assert four_bit["top_clip_ratio"] == "0.87"
    assert four_bit["top_clip_ratio_fraction"] == pytest.approx(1.0)


def test_write_choice_csvs(tmp_path: Path) -> None:
    rows = build_choice_rows(_sample_rows())
    summary = summarize_choices(rows)

    paths = write_choice_csvs(rows, summary, tmp_path, "choices")

    assert {path.name for path in paths} == {
        "choices_layer_choices.csv",
        "choices_summary.csv",
    }
    with (tmp_path / "choices_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert "top_awq_alpha" in header
    assert "selected_clip_ratios" in header


def test_write_choice_plots_writes_images_when_matplotlib_is_available(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    rows = build_choice_rows(_sample_rows())

    paths = write_choice_plots(rows, tmp_path, "choices")

    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.stat().st_size > 0 for path in paths)
