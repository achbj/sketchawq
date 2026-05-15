from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


PLOT_FIELDS = [
    "run_name",
    "method",
    "model_repo_id",
    "benchmark",
    "bits",
    "group_size",
    "sketch_mode",
    "uses_diagonal",
    "uses_low_rank",
    "sketch_rank",
    "low_rank_weight",
    "outlier_strategy",
    "outlier_metric",
    "outlier_fraction",
    "outlier_scale_boost",
    "outlier_channels",
    "protected_channels",
    "calibration_samples",
    "calibration_packing",
    "sequence_length",
    "quantization_seconds",
    "original_loss",
    "loss",
    "original_perplexity",
    "perplexity",
    "perplexity_delta",
    "perplexity_ratio",
    "estimated_quantized_mb",
    "estimated_compression_ratio",
    "original_tokens_per_second",
    "tokens_per_second",
    "throughput_ratio",
    "output_dir",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an AWQ/SketchAWQ ablation set.")
    parser.add_argument("inputs", nargs="+", help="Artifacts roots, run dirs, or run_summary.csv files.")
    parser.add_argument("--model", default=None, help="Only include this model repo id.")
    parser.add_argument("--bits", type=int, default=None, help="Only include this bit width.")
    parser.add_argument("--benchmark", default=None, help="Only include this benchmark name when present.")
    parser.add_argument("--output-dir", default="artifacts/ablation_plots")
    parser.add_argument("--name", default="ablation_quality")
    args = parser.parse_args()

    rows = _load_rows(args.inputs)
    rows = _filter_rows(rows, model=args.model, bits=args.bits, benchmark=args.benchmark)
    if not rows:
        raise SystemExit("No matching run_summary.csv rows found.")

    rows = sorted(rows, key=_sort_key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{args.name}.csv"
    _write_csv(rows, summary_path)
    html_path = output_dir / f"{args.name}.html"
    html_path.write_text(_render_html(rows), encoding="utf-8")

    image_paths = _write_matplotlib_plots(rows, output_dir, args.name)
    print(f"Wrote {summary_path}")
    print(f"Wrote {html_path}")
    for path in image_paths:
        print(f"Wrote {path}")


def _load_rows(inputs: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary_path in _summary_paths(inputs):
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row.setdefault("run_name", _infer_run_name(row, summary_path))
                row.setdefault("benchmark", "")
                rows.append(row)
    return rows


def _summary_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            paths.append(path)
            continue
        direct = path / "run_summary.csv"
        if direct.exists():
            paths.append(direct)
            continue
        paths.extend(sorted(path.glob("*/run_summary.csv")))
    return paths


def _filter_rows(
    rows: list[dict[str, str]],
    *,
    model: str | None,
    bits: int | None,
    benchmark: str | None,
) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        if model is not None and row.get("model_repo_id") != model:
            continue
        if bits is not None and _int_or_none(row.get("bits")) != bits:
            continue
        if benchmark is not None and row.get("benchmark") != benchmark:
            continue
        filtered.append(row)
    return filtered


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    fields = [field for field in PLOT_FIELDS if any(field in row for row in rows)]
    extra_fields = sorted({key for row in rows for key in row if key not in fields})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_html(rows: list[dict[str, str]]) -> str:
    max_ppl = max((_float_or_nan(row.get("perplexity")) for row in rows), default=1.0)
    max_time = max((_float_or_nan(row.get("quantization_seconds")) for row in rows), default=1.0)
    max_ppl = max(max_ppl, 1.0)
    max_time = max(max_time, 1.0)
    cards = "\n".join(_render_card(row, max_ppl=max_ppl, max_time=max_time) for row in rows)
    table_rows = "\n".join(_render_table_row(row) for row in rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SketchAWQ Ablation Quality</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18202a;
      --muted: #5d6979;
      --line: #d9e0e8;
      --panel: #f7f9fc;
      --accent: #1f7a8c;
      --accent-2: #d17a22;
    }}
    body {{
      margin: 0;
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 36px 22px 56px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 44px);
      line-height: 1.08;
    }}
    p {{
      margin: 0 0 22px;
      color: var(--muted);
      max-width: 820px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin: 24px 0 30px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: var(--panel);
    }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .metric {{
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 8px;
      align-items: center;
      margin: 8px 0;
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
    }}
    .bar {{
      height: 10px;
      border-radius: 999px;
      background: #dce7ea;
      overflow: hidden;
    }}
    .bar span {{
      display: block;
      height: 100%;
      background: var(--accent);
    }}
    .time span {{
      background: var(--accent-2);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eef3f7;
      color: #27313f;
    }}
    code {{
      font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
  </style>
</head>
<body>
  <main>
    <h1>SketchAWQ Ablation Quality</h1>
    <p>Lower perplexity and loss are better. The throughput number is dense dequantized
    simulation throughput until packed int3/int4 kernels are added.</p>
    <section class="grid">{cards}</section>
    <table>
      <thead>
        <tr>
          <th>Run</th>
          <th>Method</th>
          <th>Mode</th>
          <th>Rank</th>
          <th>Weight</th>
          <th>Outliers</th>
          <th>Packing</th>
          <th>Loss</th>
          <th>Perplexity</th>
          <th>PPL Ratio</th>
          <th>Compression</th>
          <th>Estimated MB</th>
          <th>Quant Time</th>
          <th>Tokens/s</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </main>
</body>
</html>
"""


def _render_card(row: dict[str, str], *, max_ppl: float, max_time: float) -> str:
    ppl = _float_or_nan(row.get("perplexity"))
    time_s = _float_or_nan(row.get("quantization_seconds"))
    ppl_width = _bar_width(ppl, max_ppl)
    time_width = _bar_width(time_s, max_time)
    return f"""
<article class="card">
  <h2>{html.escape(row.get("run_name") or row.get("method") or "run")}</h2>
  <div class="metric"><div class="label">Method</div><div>{html.escape(row.get("method", ""))}</div></div>
  <div class="metric"><div class="label">Mode</div><div><code>{html.escape(row.get("sketch_mode", ""))}</code></div></div>
  <div class="metric"><div class="label">Rank</div><div>{html.escape(row.get("sketch_rank", ""))}</div></div>
  <div class="metric"><div class="label">Low-rank weight</div><div>{html.escape(row.get("low_rank_weight", ""))}</div></div>
  <div class="metric"><div class="label">Outliers</div><div>{html.escape(_outlier_label(row))}</div></div>
  <div class="metric"><div class="label">Perplexity</div><div>{_fmt(ppl)}</div></div>
  <div class="bar"><span style="width:{ppl_width:.1f}%"></span></div>
  <div class="metric"><div class="label">PPL ratio</div><div>{_fmt(row.get("perplexity_ratio"))}</div></div>
  <div class="metric"><div class="label">Compression</div><div>{_fmt(row.get("estimated_compression_ratio"))}x</div></div>
  <div class="metric"><div class="label">Quant time</div><div>{_fmt(time_s)} s</div></div>
  <div class="bar time"><span style="width:{time_width:.1f}%"></span></div>
</article>"""


def _render_table_row(row: dict[str, str]) -> str:
    cells = [
        row.get("run_name", ""),
        row.get("method", ""),
        row.get("sketch_mode", ""),
        row.get("sketch_rank", ""),
        row.get("low_rank_weight", ""),
        _outlier_label(row),
        _packing_label(row),
        _fmt(row.get("loss")),
        _fmt(row.get("perplexity")),
        _fmt(row.get("perplexity_ratio")),
        f"{_fmt(row.get('estimated_compression_ratio'))}x",
        _fmt(row.get("estimated_quantized_mb")),
        f"{_fmt(row.get('quantization_seconds'))} s",
        _fmt(row.get("tokens_per_second")),
    ]
    return "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in cells) + "</tr>"


def _write_matplotlib_plots(rows: list[dict[str, str]], output_dir: Path, name: str) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    labels = [_short_label(row) for row in rows]
    perplexities = [_float_or_nan(row.get("perplexity")) for row in rows]
    ratios = [_float_or_nan(row.get("perplexity_ratio")) for row in rows]
    losses = [_float_or_nan(row.get("loss")) for row in rows]
    times = [_float_or_nan(row.get("quantization_seconds")) for row in rows]
    compression = [_float_or_nan(row.get("estimated_compression_ratio")) for row in rows]
    quantized_mb = [_float_or_nan(row.get("estimated_quantized_mb")) for row in rows]
    has_storage = any(math.isfinite(value) for value in compression)

    if has_storage:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        _bar(axes[0, 0], labels, perplexities, "Perplexity", "lower is better")
        _bar(axes[0, 1], labels, ratios, "Perplexity Ratio", "quantized / original")
        _bar(axes[0, 2], labels, compression, "Compression Ratio", "higher is better")
        _bar(axes[1, 0], labels, losses, "Cross-Entropy Loss", "lower is better")
        _bar(axes[1, 1], labels, quantized_mb, "Estimated Quantized Size", "MB")
        _bar(axes[1, 2], labels, times, "Quantization Time", "seconds")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        _bar(axes[0, 0], labels, perplexities, "Perplexity", "lower is better")
        _bar(axes[0, 1], labels, ratios, "Perplexity Ratio", "quantized / original")
        _bar(axes[1, 0], labels, losses, "Cross-Entropy Loss", "lower is better")
        _bar(axes[1, 1], labels, times, "Quantization Time", "seconds")
    fig.suptitle("SketchAWQ Quality Comparison", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _bar(ax, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    palette = ["#1f7a8c", "#7a8b35", "#d17a22", "#6f5b95", "#b35c44", "#397367"]
    colors = [palette[index % len(palette)] for index in range(len(labels))]
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=18)
    for index, value in enumerate(values):
        if math.isfinite(value):
            ax.text(index, value, _fmt(value), ha="center", va="bottom", fontsize=9)


def _sort_key(row: dict[str, str]) -> tuple[int, float, float, str]:
    mode = (row.get("sketch_mode") or "").lower()
    method = (row.get("method") or "").lower()
    rank = _int_or_none(row.get("sketch_rank")) or 0
    weight = _sort_number(row.get("low_rank_weight"))
    run_name = row.get("run_name", "")
    if mode == "none" or "awq-only" in method:
        return (0, 0.0, 0.0, run_name)
    if mode == "diagonal":
        return (1, 0.0, 0.0, run_name)
    if mode == "diagonal_low_rank" and rank <= 0:
        return (2, 0.0, weight, run_name)
    if mode == "diagonal_low_rank":
        return (3, float(rank), weight, run_name)
    return (4, 0.0, 0.0, run_name)


def _short_label(row: dict[str, str]) -> str:
    mode = row.get("sketch_mode") or row.get("method") or row.get("run_name", "")
    if mode == "none":
        label = "AWQ-only"
        return _label_with_packing(label, row)
    if mode == "diagonal":
        label = "Diagonal"
        return _label_with_packing(label, row)
    if mode == "diagonal_low_rank":
        rank = row.get("sketch_rank", "")
        weight = row.get("low_rank_weight", "")
        if rank or weight:
            return _label_with_packing(f"r{rank} w{weight}", row)
        return _label_with_packing("Diag+Low-rank", row)
    return _label_with_packing(mode.replace("_", " ").title(), row)


def _label_with_packing(label: str, row: dict[str, str]) -> str:
    outliers = _outlier_label(row)
    if outliers != "none":
        label = f"{label} {outliers}"
    if _is_true(row.get("calibration_packing")):
        return f"{label} packed"
    return label


def _outlier_label(row: dict[str, str]) -> str:
    strategy = (row.get("outlier_strategy") or "none").strip().lower()
    fraction = row.get("outlier_fraction", "")
    if strategy in {"", "none"} or _float_or_nan(fraction) <= 0:
        return "none"
    percent = _float_or_nan(fraction) * 100
    if strategy == "scale":
        boost = row.get("outlier_scale_boost", "")
        return f"scale {percent:.2g}% x{_fmt(boost)}"
    if strategy == "protect":
        return f"protect {percent:.2g}%"
    return f"{strategy} {percent:.2g}%"


def _packing_label(row: dict[str, str]) -> str:
    return "packed" if _is_true(row.get("calibration_packing")) else "padded"


def _infer_run_name(row: dict[str, str], summary_path: Path) -> str:
    output_dir = row.get("output_dir")
    if output_dir:
        return Path(output_dir).name
    return summary_path.parent.name


def _bar_width(value: float, maximum: float) -> float:
    if not math.isfinite(value) or maximum <= 0:
        return 0.0
    return max(2.0, min(100.0, 100.0 * value / maximum))


def _fmt(value: object) -> str:
    number = _float_or_nan(value)
    if not math.isfinite(number):
        return ""
    if abs(number) >= 100:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.3f}"


def _float_or_nan(value: object) -> float:
    try:
        if value in {None, ""}:
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _int_or_none(value: object) -> int | None:
    try:
        if value in {None, ""}:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _sort_number(value: object) -> float:
    number = _float_or_nan(value)
    return number if math.isfinite(number) else 0.0


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
