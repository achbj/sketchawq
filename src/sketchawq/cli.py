from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sketchawq.experiments.catalog import ExperimentTarget, load_catalog


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args) or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sketchawq", description="SketchAWQ project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-targets", help="List catalog experiment targets")
    list_parser.add_argument("--catalog", default=None, help="Path to configs/experiments.yaml")
    list_parser.set_defaults(handler=_handle_list_targets)

    run_parser = subparsers.add_parser("run", help="Run one or more SketchAWQ configs locally")
    run_parser.add_argument("configs", nargs="+", help="SketchAWQ YAML configs")
    _add_runtime_options(run_parser)
    run_parser.set_defaults(handler=_handle_run)

    target_parser = subparsers.add_parser("run-target", help="Run a named catalog target locally")
    target_parser.add_argument("target", help="Target name from configs/experiments.yaml")
    target_parser.add_argument("--catalog", default=None, help="Path to configs/experiments.yaml")
    _add_runtime_options(target_parser)
    target_parser.set_defaults(handler=_handle_run_target)

    baseline_parser = subparsers.add_parser("baseline", help="Run one or more baseline configs locally")
    baseline_parser.add_argument("configs", nargs="+", help="External baseline YAML configs")
    _add_runtime_options(baseline_parser)
    baseline_parser.set_defaults(handler=_handle_baseline)

    inference_parser = subparsers.add_parser(
        "inference",
        help="Benchmark generation inference throughput for a SketchAWQ config",
    )
    inference_parser.add_argument("config", help="SketchAWQ YAML config")
    inference_parser.add_argument(
        "--variant",
        choices=["fp16", "sketchawq-dense", "sketchawq_dense"],
        default="fp16",
        help="Model/runtime variant to benchmark",
    )
    inference_parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512, 2048])
    inference_parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    inference_parser.add_argument("--generated-tokens", type=int, default=128)
    inference_parser.add_argument("--warmup-runs", type=int, default=1)
    inference_parser.add_argument("--seed-text", default="SketchAWQ inference benchmark prompt.")
    _add_runtime_options(inference_parser)
    inference_parser.set_defaults(handler=_handle_inference)

    return parser


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-cache-dir", default=None, help="Optional Hugging Face cache directory")
    parser.add_argument("--output-root", default=None, help="Optional root directory for run artifacts")
    parser.add_argument("--json", action="store_true", help="Print full JSON reports")


def _handle_list_targets(args: argparse.Namespace) -> int:
    for target in load_catalog(args.catalog).values():
        configs = ", ".join(_display_path(path) for path in target.configs)
        detail = f" - {target.description}" if target.description else ""
        print(f"{target.name} [{target.runner}]{detail}")
        print(f"  {configs}")
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    reports = _run_sketchawq_configs(
        [Path(item) for item in args.configs],
        model_cache_dir=args.model_cache_dir,
        output_root=args.output_root,
    )
    _print_reports(reports, as_json=args.json)
    return 0


def _handle_baseline(args: argparse.Namespace) -> int:
    reports = _run_baseline_configs(
        [Path(item) for item in args.configs],
        model_cache_dir=args.model_cache_dir,
        output_root=args.output_root,
    )
    _print_reports(reports, as_json=args.json)
    return 0


def _handle_inference(args: argparse.Namespace) -> int:
    from sketchawq.experiments.benchmark_inference import benchmark_from_config

    report = benchmark_from_config(
        args.config,
        variant=args.variant,
        prompt_lengths=args.prompt_lengths,
        batch_sizes=args.batch_sizes,
        generated_tokens=args.generated_tokens,
        warmup_runs=args.warmup_runs,
        model_cache_dir=args.model_cache_dir,
        output_root=args.output_root or "artifacts/inference",
        seed_text=args.seed_text,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"completed: {Path(str(report['output_dir'])).name}")
    return 0


def _handle_run_target(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    try:
        target = catalog[args.target]
    except KeyError as exc:
        available = ", ".join(catalog)
        raise SystemExit(f"unknown target {args.target!r}; available targets: {available}") from exc

    reports = _run_target(
        target,
        model_cache_dir=args.model_cache_dir,
        output_root=args.output_root,
    )
    _print_reports(reports, as_json=args.json)
    return 0


def _run_target(
    target: ExperimentTarget,
    *,
    model_cache_dir: str | None,
    output_root: str | None,
) -> list[dict[str, Any]]:
    if target.runner == "baseline":
        return _run_baseline_configs(
            target.configs,
            model_cache_dir=model_cache_dir,
            output_root=output_root,
        )
    return _run_sketchawq_configs(
        target.configs,
        model_cache_dir=model_cache_dir,
        output_root=output_root,
    )


def _run_sketchawq_configs(
    configs: Sequence[Path],
    *,
    model_cache_dir: str | None,
    output_root: str | None,
) -> list[dict[str, Any]]:
    from sketchawq.experiments.run_quantization import run_from_config

    return [
        run_from_config(path, model_cache_dir=model_cache_dir, output_root=output_root)
        for path in configs
    ]


def _run_baseline_configs(
    configs: Sequence[Path],
    *,
    model_cache_dir: str | None,
    output_root: str | None,
) -> list[dict[str, Any]]:
    from sketchawq.experiments.run_baseline import run_from_config

    return [
        run_from_config(path, model_cache_dir=model_cache_dir, output_root=output_root)
        for path in configs
    ]


def _print_reports(reports: Sequence[dict[str, Any]], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(list(reports), indent=2))
        return
    for report in reports:
        metadata = report.get("metadata", {})
        run_name = metadata.get("run_name") or Path(str(report.get("output_dir", ""))).name
        print(f"completed: {run_name}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
