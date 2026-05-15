from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sketchawq.baselines.runner import run_baseline_from_config


def run_from_config(
    config_path: str | Path,
    *,
    model_cache_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    return run_baseline_from_config(
        config_path,
        model_cache_dir=model_cache_dir,
        output_root=output_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an external quantization baseline.")
    parser.add_argument("--config", required=True, help="Path to an external baseline YAML config.")
    parser.add_argument("--model-cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--output-root", default=None, help="Optional root directory for run artifacts.")
    args = parser.parse_args()
    run_from_config(args.config, model_cache_dir=args.model_cache_dir, output_root=args.output_root)


if __name__ == "__main__":
    main()
