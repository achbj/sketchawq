from __future__ import annotations

import argparse

from sketchawq.experiments.run_quantization import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for evaluation experiments.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
