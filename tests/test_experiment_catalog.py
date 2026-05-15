from __future__ import annotations

from pathlib import Path

import pytest

from sketchawq.baselines.config import load_baseline_config
from sketchawq.config import load_config
from sketchawq.experiments.catalog import load_catalog


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_catalog_loads_curated_targets() -> None:
    catalog = load_catalog(PROJECT_ROOT / "configs" / "experiments.yaml")

    assert set(catalog) == {
        "baseline-llama7b",
        "baseline-qwen2",
        "baseline-qwen3-1-7b",
        "baseline-qwen3-4b",
        "baseline-qwen3-8b",
        "baseline-qwen3-14b",
        "baseline-qwen3-32b",
        "paper-llama7b",
        "paper-qwen2",
        "paper-qwen3-1-7b",
        "paper-qwen3-14b",
        "paper-qwen3-32b",
        "paper-qwen3-4b",
        "paper-qwen3-8b",
        "smoke",
        "smoke-qwen",
    }
    assert catalog["smoke"].runner == "sketchawq"
    assert len(catalog["smoke"].configs) == 1


def test_catalog_configs_exist_and_validate() -> None:
    catalog = load_catalog(PROJECT_ROOT / "configs" / "experiments.yaml")

    for target in catalog.values():
        for config_path in target.configs:
            assert config_path.exists()
            if target.runner == "baseline":
                load_baseline_config(config_path)
            else:
                load_config(config_path)


def test_catalog_rejects_unknown_runner(tmp_path) -> None:
    catalog_path = tmp_path / "experiments.yaml"
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text("{}", encoding="utf-8")
    catalog_path.write_text(
        """
targets:
  bad:
    runner: magic
    configs:
      - tiny.yaml
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runner"):
        load_catalog(catalog_path)
