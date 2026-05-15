from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_RUNNERS = {"baseline", "sketchawq"}


@dataclass(frozen=True, slots=True)
class ExperimentTarget:
    name: str
    runner: str
    configs: tuple[Path, ...]
    description: str = ""


def default_catalog_path() -> Path:
    return Path("configs") / "experiments.yaml"


def load_catalog(path: str | Path | None = None) -> dict[str, ExperimentTarget]:
    catalog_path = Path(path) if path is not None else default_catalog_path()
    if not catalog_path.is_absolute():
        catalog_path = Path.cwd() / catalog_path
    catalog_path = catalog_path.resolve()
    with catalog_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    project_root = catalog_path.parent.parent
    targets = raw.get("targets", {})
    if not isinstance(targets, dict):
        raise ValueError("catalog must contain a mapping at 'targets'")

    return {
        name: _coerce_target(name, data, project_root=project_root)
        for name, data in sorted(targets.items())
    }


def _coerce_target(
    name: str,
    data: dict[str, Any],
    *,
    project_root: Path,
) -> ExperimentTarget:
    runner = str(data.get("runner", "sketchawq")).strip().lower()
    if runner not in VALID_RUNNERS:
        allowed = ", ".join(sorted(VALID_RUNNERS))
        raise ValueError(f"target {name!r} uses runner {runner!r}; expected one of {allowed}")

    raw_configs = data.get("configs", ())
    if not raw_configs:
        raise ValueError(f"target {name!r} must list at least one config")
    configs = tuple(_resolve_config_path(item, project_root=project_root) for item in raw_configs)
    missing = [str(path) for path in configs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"target {name!r} references missing configs: {missing}")

    return ExperimentTarget(
        name=name,
        runner=runner,
        configs=configs,
        description=str(data.get("description", "")).strip(),
    )


def _resolve_config_path(value: str | Path, *, project_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()
