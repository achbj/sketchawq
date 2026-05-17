from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


LAYER_CHOICE_FIELDS = [
    "run_name",
    "bits",
    "group_size",
    "layer_type",
    "layer_name",
    "awq_alpha",
    "clip_ratio",
    "search_score",
    "scale_score",
    "clip_score",
]

SUMMARY_FIELDS = [
    "run_name",
    "bits",
    "group_size",
    "layers",
    "top_awq_alpha",
    "top_awq_alpha_layers",
    "top_awq_alpha_fraction",
    "selected_awq_alphas",
    "top_clip_ratio",
    "top_clip_ratio_layers",
    "top_clip_ratio_fraction",
    "selected_clip_ratios",
]

LAYER_TYPES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize selected AWQ alpha and clip ratios from layer_metrics.csv files."
    )
    parser.add_argument("inputs", nargs="+", help="Artifacts roots, run dirs, or layer_metrics.csv files.")
    parser.add_argument("--output-dir", default="artifacts/search_choices")
    parser.add_argument("--name", default="search_choices")
    parser.add_argument("--title", default="SketchAWQ search choices by layer")
    args = parser.parse_args()

    rows = build_choice_rows(load_layer_rows(args.inputs))
    if not rows:
        raise SystemExit("No layer rows with awq_alpha and clip_ratio were found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_choices(rows)
    csv_paths = write_choice_csvs(rows, summary, output_dir, args.name)
    plot_paths = write_choice_plots(rows, output_dir, args.name, title=args.title)

    for path in csv_paths + plot_paths:
        print(f"Wrote {path}")


def load_layer_rows(inputs: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in _layer_metric_paths(inputs):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(dict(row))
    return rows


def build_choice_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    choices: list[dict[str, str]] = []
    for row in rows:
        if not row.get("awq_alpha") or not row.get("clip_ratio"):
            continue
        choice = {field: str(row.get(field, "")) for field in LAYER_CHOICE_FIELDS}
        choice["layer_type"] = _layer_type(row.get("layer_name", ""))
        choice["awq_alpha"] = _format_number(row["awq_alpha"])
        choice["clip_ratio"] = _format_number(row["clip_ratio"])
        choices.append(choice)
    return sorted(choices, key=lambda row: (_int_key(row.get("bits")), row.get("layer_name", "")))


def summarize_choices(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("bits", ""), row.get("run_name", ""))].append(row)

    summary = []
    for (bits, run_name), group in grouped.items():
        alpha_counts = Counter(row["awq_alpha"] for row in group)
        clip_counts = Counter(row["clip_ratio"] for row in group)
        top_alpha, top_alpha_count = _top_count(alpha_counts)
        top_clip, top_clip_count = _top_count(clip_counts)
        layer_count = len(group)
        summary.append(
            {
                "run_name": run_name,
                "bits": bits,
                "group_size": group[0].get("group_size", ""),
                "layers": layer_count,
                "top_awq_alpha": top_alpha,
                "top_awq_alpha_layers": top_alpha_count,
                "top_awq_alpha_fraction": top_alpha_count / layer_count if layer_count else 0.0,
                "selected_awq_alphas": _format_counts(alpha_counts),
                "top_clip_ratio": top_clip,
                "top_clip_ratio_layers": top_clip_count,
                "top_clip_ratio_fraction": top_clip_count / layer_count if layer_count else 0.0,
                "selected_clip_ratios": _format_counts(clip_counts),
            }
        )
    return sorted(summary, key=lambda row: (_int_key(row.get("bits")), str(row.get("run_name", ""))))


def write_choice_csvs(
    rows: list[dict[str, str]],
    summary: list[dict[str, object]],
    output_dir: Path,
    name: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_path = output_dir / f"{name}_layer_choices.csv"
    summary_path = output_dir / f"{name}_summary.csv"
    _write_csv(rows, layer_path, LAYER_CHOICE_FIELDS)
    _write_csv(summary, summary_path, SUMMARY_FIELDS)
    return [layer_path, summary_path]


def write_choice_plots(
    rows: list[dict[str, str]],
    output_dir: Path,
    name: str,
    *,
    title: str = "SketchAWQ search choices by layer",
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("bits", "")].append(row)
    bits_values = sorted(groups, key=_int_key)
    fig, axes = plt.subplots(
        len(bits_values),
        2,
        figsize=(13, max(3.6, 3.0 * len(bits_values))),
        squeeze=False,
    )

    for row_index, bits in enumerate(bits_values):
        group = groups[bits]
        _draw_count_axis(
            axes[row_index][0],
            Counter(row["awq_alpha"] for row in group),
            f"{bits}-bit selected AWQ alpha",
            "#1f7a8c",
        )
        _draw_count_axis(
            axes[row_index][1],
            Counter(row["clip_ratio"] for row in group),
            f"{bits}-bit selected clip ratio",
            "#7a8795",
        )

    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _layer_metric_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            paths.append(path)
            continue
        direct = path / "layer_metrics.csv"
        if direct.exists():
            paths.append(direct)
            continue
        paths.extend(sorted(path.glob("*/layer_metrics.csv")))
    return paths


def _write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _draw_count_axis(ax, counts: Counter[str], title: str, color: str) -> None:
    labels = sorted(counts, key=_float_key)
    values = [counts[label] for label in labels]
    total = sum(values)
    bars = ax.bar(labels, values, color=color, alpha=0.9)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("layers")
    ax.grid(axis="y", color="#d8e0e8", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        pct = 100.0 * value / total if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}\n{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.margins(y=0.18)


def _layer_type(layer_name: str) -> str:
    for layer_type in LAYER_TYPES:
        if layer_name.endswith(layer_type):
            return layer_type
    return layer_name.rsplit(".", maxsplit=1)[-1] if layer_name else ""


def _top_count(counts: Counter[str]) -> tuple[str, int]:
    if not counts:
        return "", 0
    return sorted(counts.items(), key=lambda item: (-item[1], _float_key(item[0])))[0]


def _format_counts(counts: Counter[str]) -> str:
    return "; ".join(f"{label}:{counts[label]}" for label in sorted(counts, key=_float_key))


def _format_number(value: str) -> str:
    number = float(value)
    return f"{number:g}"


def _float_key(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_key(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
