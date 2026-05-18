from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


BOARD_FIELDS = [
    "best_for_bits",
    "display_method",
    "method_family",
    "method",
    "run_name",
    "model_repo_id",
    "benchmark",
    "bits",
    "group_size",
    "perplexity",
    "ppl_delta_from_best",
    "ppl_percent_from_best",
    "comparison_note",
    "original_perplexity",
    "perplexity_ratio",
    "loss",
    "quantization_seconds",
    "collect_stats_seconds",
    "prepare_sketches_seconds",
    "layer_quantization_seconds",
    "gpu_name",
    "tokens_per_second",
    "fp16_weight_mb",
    "estimated_quantized_mb",
    "estimated_compression_ratio",
    "calibration_source",
    "calibration_dataset",
    "calibration_split",
    "calibration_samples",
    "calibration_sequence_length",
    "calibration_packing",
    "calibration_batch_size",
    "evaluation_source",
    "evaluation_dataset",
    "evaluation_split",
    "evaluation_samples",
    "evaluation_sequence_length",
    "evaluation_packing",
    "evaluation_batch_size",
    "sequence_length",
    "output_dir",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an apples-to-apples benchmark board from run_summary.csv files."
    )
    parser.add_argument("inputs", nargs="+", help="Artifacts roots, run dirs, or run_summary.csv files.")
    parser.add_argument("--model", default=None, help="Only include this model repo id.")
    parser.add_argument("--benchmark", default=None, help="Only include this benchmark name.")
    parser.add_argument("--output-dir", default="artifacts/benchmark_board")
    parser.add_argument("--name", default="competition_board")
    args = parser.parse_args()

    rows = load_summary_rows(args.inputs)
    board_rows = build_board_rows(rows, model=args.model, benchmark=args.benchmark)
    if not board_rows:
        raise SystemExit(describe_empty_board(rows, model=args.model, benchmark=args.benchmark))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.name}.csv"
    html_path = output_dir / f"{args.name}.html"
    _write_csv(board_rows, csv_path)
    html_path.write_text(_render_html(board_rows), encoding="utf-8")
    image_paths = _write_matplotlib_plots(board_rows, output_dir, args.name)
    print(f"Wrote {csv_path}")
    print(f"Wrote {html_path}")
    for path in image_paths:
        print(f"Wrote {path}")


def load_summary_rows(inputs: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary_path in _summary_paths(inputs):
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(dict(row))
    return rows


def build_board_rows(
    rows: list[dict[str, str]],
    *,
    model: str | None,
    benchmark: str | None,
) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        if model is not None and row.get("model_repo_id") != model:
            continue
        if benchmark is not None and row.get("benchmark") != benchmark:
            continue
        perplexity = _float_or_nan(row.get("perplexity"))
        if math.isnan(perplexity) or perplexity <= 0:
            continue
        work = dict(row)
        work.setdefault("run_name", _row_label(row))
        work["method_family"] = _method_family(work.get("method", ""))
        work["display_method"] = _display_method(work)
        filtered.append(work)

    best_ppl_by_bits: dict[int, float] = {}
    for row in filtered:
        bits = _int_or_none(row.get("bits"))
        if bits is None:
            continue
        ppl = _float_or_nan(row.get("perplexity"))
        best_ppl_by_bits[bits] = min(best_ppl_by_bits.get(bits, math.inf), ppl)

    for row in filtered:
        bits = _int_or_none(row.get("bits"))
        ppl = _float_or_nan(row.get("perplexity"))
        best_ppl = best_ppl_by_bits.get(bits, math.inf)
        delta = ppl - best_ppl
        pct_delta = 100.0 * delta / best_ppl if best_ppl > 0 and math.isfinite(best_ppl) else math.nan
        is_best = abs(delta) < 1e-9
        row["best_for_bits"] = "yes" if is_best else ""
        row["ppl_delta_from_best"] = 0.0 if is_best else delta
        row["ppl_percent_from_best"] = 0.0 if is_best else pct_delta
        row["comparison_note"] = "best" if is_best else _delta_label(delta, pct_delta)
        if not row.get("perplexity_ratio"):
            original_ppl = _float_or_nan(row.get("original_perplexity"))
            if original_ppl > 0:
                row["perplexity_ratio"] = f"{ppl / original_ppl:.6g}"

    return sorted(filtered, key=_sort_key)


def describe_empty_board(
    rows: list[dict[str, str]],
    *,
    model: str | None,
    benchmark: str | None,
) -> str:
    matching_rows = [
        row
        for row in rows
        if (model is None or row.get("model_repo_id") == model)
        and (benchmark is None or row.get("benchmark") == benchmark)
    ]
    if not matching_rows:
        filters = []
        if model is not None:
            filters.append(f"model={model}")
        if benchmark is not None:
            filters.append(f"benchmark={benchmark}")
        suffix = f" for {' and '.join(filters)}" if filters else ""
        return f"No matching run_summary.csv rows found{suffix}."

    unusable = [
        row
        for row in matching_rows
        if math.isnan(_float_or_nan(row.get("perplexity")))
        or _float_or_nan(row.get("perplexity")) <= 0
    ]
    if unusable:
        run_names = ", ".join(_row_label(row) for row in unusable[:5])
        hidden = len(unusable) - 5
        if hidden > 0:
            run_names = f"{run_names}, and {hidden} more"
        plural = "" if len(matching_rows) == 1 else "s"
        return (
            "No usable run_summary.csv rows found: "
            f"{len(matching_rows)} matching row{plural}, but "
            f"{len(unusable)} had missing, non-finite perplexity, or perplexity <= 0. "
            f"Affected runs: {run_names}."
        )

    return "No matching run_summary.csv rows found."


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


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    fields = [field for field in BOARD_FIELDS if any(field in row for row in rows)]
    extra_fields = sorted({key for row in rows for key in row if key not in fields})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_html(rows: list[dict[str, str]]) -> str:
    table_rows = "\n".join(_render_table_row(row) for row in rows)
    cards = "\n".join(_render_winner_card(row) for row in rows if row.get("best_for_bits") == "yes")
    context = _board_context(rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SketchAWQ Competition Board</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #16202a;
      --muted: #5f6d7a;
      --line: #d8e0e8;
      --panel: #f7f9fb;
      --win: #1f7a8c;
      --warn: #c57420;
    }}
    body {{
      margin: 0;
      background: #fff;
      color: var(--ink);
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 34px 22px 54px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 44px);
      line-height: 1.08;
    }}
    p {{
      margin: 0 0 20px;
      color: var(--muted);
      max-width: 880px;
    }}
    .winners {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin: 24px 0 28px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 16px;
    }}
    .card strong {{
      display: block;
      font-size: 18px;
      margin-bottom: 8px;
    }}
    .context {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      margin: 18px 0 4px;
    }}
    .context div {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fbfcfd;
    }}
    .context span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .03em;
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
    }}
    tr.best {{
      background: #eef8fa;
    }}
    code {{
      font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .best-badge {{
      color: var(--win);
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Paper-Eval Comparison Board</h1>
    <p>Lower perplexity is better. The delta column shows how far each row is from the best method at the same bit-width. Throughput is dense dequantized simulation throughput until packed int3/int4 kernels are added.</p>
    <section class="context">
      <div><span>Model</span>{html.escape(context["model"])}</div>
      <div><span>Evaluation Data</span>{html.escape(context["evaluation"])}</div>
      <div><span>Calibration Data</span>{html.escape(context["calibration"])}</div>
      <div><span>Model Size</span>{html.escape(context["model_size"])}</div>
    </section>
    <section class="winners">{cards}</section>
    <table>
      <thead>
        <tr>
          <th>Best</th>
          <th>Bits</th>
          <th>Method</th>
          <th>Run</th>
          <th>PPL</th>
          <th>Delta vs Best</th>
          <th>FP16 PPL</th>
          <th>Ratio</th>
          <th>Tokens/s</th>
          <th>Quant Time</th>
          <th>GPU</th>
          <th>FP16 Size</th>
          <th>Quantized Size</th>
          <th>Compression</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </main>
</body>
</html>
"""


def _render_winner_card(row: dict[str, str]) -> str:
    bits = html.escape(str(row.get("bits", "")))
    method = html.escape(str(row.get("display_method") or row.get("method", "")))
    ppl = _format_float(row.get("perplexity"), digits=3)
    ratio = _format_float(row.get("perplexity_ratio"), digits=3)
    note = html.escape(str(row.get("comparison_note", "")))
    return f"""<article class="card">
  <strong>{bits}-bit winner: {method}</strong>
  <div>PPL {ppl} &nbsp;&middot;&nbsp; {ratio}x vs FP16 &nbsp;&middot;&nbsp; {note}</div>
</article>"""


def _render_table_row(row: dict[str, str]) -> str:
    best = row.get("best_for_bits") == "yes"
    cells = [
        '<span class="best-badge">yes</span>' if best else "",
        html.escape(str(row.get("bits", ""))),
        html.escape(str(row.get("display_method") or row.get("method", ""))),
        f"<code>{html.escape(str(row.get('run_name', '')))}</code>",
        _format_float(row.get("perplexity"), digits=3),
        html.escape(str(row.get("comparison_note", ""))),
        _format_float(row.get("original_perplexity"), digits=3),
        _format_float(row.get("perplexity_ratio"), digits=3),
        _format_float(row.get("tokens_per_second"), digits=1),
        _format_float(row.get("quantization_seconds"), digits=2),
        html.escape(str(row.get("gpu_name", ""))),
        _format_size_mb(row.get("fp16_weight_mb")),
        _format_size_mb(row.get("estimated_quantized_mb")),
        _format_compression(row.get("estimated_compression_ratio")),
    ]
    class_name = ' class="best"' if best else ""
    return f"<tr{class_name}>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def _write_matplotlib_plots(rows: list[dict[str, str]], output_dir: Path, name: str) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        bits = _int_or_none(row.get("bits"))
        if bits is not None:
            grouped.setdefault(bits, []).append(row)

    if not grouped:
        return []

    max_rows = max(len(items) for items in grouped.values())
    fig_height = max(4.8, 1.0 * max_rows + 2.6)
    fig, axes = plt.subplots(
        1,
        len(grouped),
        figsize=(8.2 * len(grouped), fig_height),
        squeeze=False,
    )
    for axis, bits in zip(axes[0], sorted(grouped, reverse=True), strict=False):
        bit_rows = sorted(grouped[bits], key=lambda row: _float_or_nan(row.get("perplexity")))
        labels = [_axis_label(row) for row in bit_rows]
        values = [_float_or_nan(row.get("perplexity")) for row in bit_rows]
        y_positions = list(range(len(bit_rows)))
        colors = [_row_color(row) for row in bit_rows]
        axis.barh(
            y_positions,
            values,
            color=colors,
            edgecolor=["#102a33" if row.get("best_for_bits") == "yes" else "#dae2ea" for row in bit_rows],
            linewidth=[1.6 if row.get("best_for_bits") == "yes" else 0.8 for row in bit_rows],
        )
        axis.set_yticks(y_positions, labels)
        axis.invert_yaxis()
        best = bit_rows[0]
        best_value = values[0]
        axis.axvline(best_value, color="#102a33", linewidth=1.0, linestyle="--", alpha=0.6)
        axis.set_title(
            f"{bits}-bit: {best.get('display_method', '')} leads",
            fontsize=13,
            fontweight="bold",
        )
        axis.set_xlabel("WikiText2 perplexity (lower is better)")
        axis.grid(axis="x", color="#d8e0e8", linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)
        finite_values = [value for value in values if math.isfinite(value)]
        if finite_values:
            span = max(finite_values) - min(finite_values)
            margin = max(0.10, span * 0.85)
            axis.set_xlim(max(0.0, min(finite_values) - margin), max(finite_values) + margin)
        for index, value in enumerate(values):
            if not math.isfinite(value):
                continue
            label = _plot_value_label(bit_rows[index], value)
            axis.text(value, index, f"  {label}", va="center", ha="left", fontsize=9)
    context = _board_context(rows)
    fig.suptitle(
        f"{context['model']} | eval {context['evaluation']} | calib {context['calibration']}",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Bars are zoomed around the observed PPL range. Compare only rows with matching model, dataset, context length, and evaluation budget.",
        ha="center",
        fontsize=9,
        color="#52606d",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _delta_label(delta: float, pct_delta: float) -> str:
    if not math.isfinite(delta) or not math.isfinite(pct_delta):
        return ""
    return f"+{delta:.3f} PPL (+{pct_delta:.2f}%) vs best"


def _board_context(rows: list[dict[str, str]]) -> dict[str, str]:
    models = _unique_values(rows, "model_repo_id")
    return {
        "model": ", ".join(models) if models else "mixed models",
        "evaluation": _combined_data_context(rows, "evaluation", fallback_key="benchmark"),
        "calibration": _combined_data_context(rows, "calibration"),
        "model_size": _model_size_context(rows),
    }


def _unique_values(rows: list[dict[str, str]], key: str) -> list[str]:
    values = {str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()}
    return sorted(values)


def _data_context(row: dict[str, str], prefix: str, *, fallback: object | None = None) -> str:
    dataset = str(row.get(f"{prefix}_dataset") or "").strip()
    source = str(row.get(f"{prefix}_source") or "").strip()
    split = str(row.get(f"{prefix}_split") or "").strip()
    samples = str(row.get(f"{prefix}_samples") or "").strip()
    sequence_length = str(row.get(f"{prefix}_sequence_length") or "").strip()
    if prefix == "calibration" and not sequence_length:
        sequence_length = str(row.get("sequence_length") or "").strip()
    packing = str(row.get(f"{prefix}_packing") or "").strip()

    fallback_label = str(fallback or "").strip()
    label = dataset or source
    parts = []
    if fallback_label:
        parts.append(fallback_label)
    if label and label != fallback_label:
        parts.append(label)
    if split:
        parts.append(split)
    if samples:
        parts.append(f"{samples} samples")
    if sequence_length:
        parts.append(f"seq {sequence_length}")
    if packing:
        parts.append(f"packing={packing}")
    return html.escape(", ".join(parts))


def _combined_data_context(
    rows: list[dict[str, str]],
    prefix: str,
    *,
    fallback_key: str | None = None,
) -> str:
    labels = _unique_values(rows, f"{prefix}_dataset")
    fallback_labels = _unique_values(rows, fallback_key) if fallback_key is not None else []
    samples = _unique_values(rows, f"{prefix}_samples")
    sequence_lengths = _unique_values(rows, f"{prefix}_sequence_length")
    if prefix == "calibration" and not sequence_lengths:
        sequence_lengths = _unique_values(rows, "sequence_length")
    packing_values = _unique_values(rows, f"{prefix}_packing")
    batch_sizes = _unique_values(rows, f"{prefix}_batch_size")

    parts = []
    if fallback_labels:
        parts.append(", ".join(fallback_labels))
    if labels:
        parts.append(", ".join(labels))
    if samples:
        parts.append(f"{', '.join(samples)} samples")
    if sequence_lengths:
        parts.append(f"seq {', '.join(sequence_lengths)}")
    if packing_values and any(v.lower() == "true" for v in packing_values):
        parts.append("packed")
    if batch_sizes:
        parts.append(f"batch {', '.join(batch_sizes)}")
    return " · ".join(parts) if parts else "not recorded"


def _model_size_context(rows: list[dict[str, str]]) -> str:
    fp16_sizes = [_float_or_nan(row.get("fp16_weight_mb")) for row in rows]
    quant_sizes = [_float_or_nan(row.get("estimated_quantized_mb")) for row in rows]
    compression = [_float_or_nan(row.get("estimated_compression_ratio")) for row in rows]
    fp16_label = _numeric_range_label(fp16_sizes, formatter=_format_size_number)
    quantized_label = _numeric_range_label(quant_sizes, formatter=_format_size_number)
    compression_label = _numeric_range_label(compression, formatter=lambda value: f"{value:.2f}x")
    return f"FP16 {fp16_label}; quantized {quantized_label}; compression {compression_label}"


def _numeric_range_label(values: list[float], *, formatter) -> str:
    finite = sorted({value for value in values if math.isfinite(value) and value > 0})
    if not finite:
        return "not recorded"
    if len(finite) == 1:
        return formatter(finite[0])
    return f"{formatter(finite[0])}-{formatter(finite[-1])}"


def _format_size_number(value: float) -> str:
    if value >= 1024:
        return f"{value / 1024:.2f} GB"
    return f"{value:.1f} MB"


def _method_family(method: object) -> str:
    text = str(method or "").lower()
    if "sketchawq" in text:
        return "SketchAWQ"
    if "gptq" in text:
        return "GPTQ"
    if "llmcompressor" in text or "llm-compressor" in text:
        return "LLMCompressor-AWQ"
    if "autoawq" in text or "awq" in text:
        return "AWQ"
    if "hqq" in text:
        return "HQQ"
    return str(method or "Unknown")


def _display_method(row: dict[str, str]) -> str:
    family = _method_family(row.get("method", ""))
    if family == "SketchAWQ":
        return "SketchAWQ"
    if family == "GPTQ":
        return "GPTQ"
    return family


def _axis_label(row: dict[str, str]) -> str:
    label = str(row.get("display_method") or row.get("method") or "run")
    group_size = str(row.get("group_size", "")).strip()
    throughput = _float_or_nan(row.get("tokens_per_second"))
    suffixes = []
    if group_size:
        suffixes.append(f"g{group_size}")
    if math.isfinite(throughput) and throughput > 0:
        suffixes.append(f"{throughput:.0f} tok/s")
    if suffixes:
        return f"{label} ({', '.join(suffixes)})"
    return label


def _row_color(row: dict[str, str]) -> str:
    if row.get("best_for_bits") == "yes":
        return "#1f7a8c"
    family = str(row.get("method_family", ""))
    if family == "SketchAWQ":
        return "#6f9fab"
    if family == "GPTQ":
        return "#8794a3"
    if family == "AWQ":
        return "#a68a3f"
    if family == "HQQ":
        return "#7c6aa5"
    return "#9aa6b2"


def _plot_value_label(row: dict[str, str], value: float) -> str:
    if row.get("best_for_bits") == "yes":
        return f"{_format_float(value, digits=2)} (best)"
    delta = _float_or_nan(row.get("ppl_delta_from_best"))
    pct_delta = _float_or_nan(row.get("ppl_percent_from_best"))
    if math.isfinite(delta) and math.isfinite(pct_delta):
        return f"{_format_float(value, digits=2)} (+{delta:.3f}, +{pct_delta:.2f}%)"
    return _format_float(value, digits=2)


def _sort_key(row: dict[str, str]) -> tuple[int, float, str]:
    bits = _int_or_none(row.get("bits"))
    return (-(bits or -1), _float_or_nan(row.get("perplexity")), row.get("method", ""))


def _row_label(row: dict[str, str]) -> str:
    return row.get("run_name") or row.get("output_dir") or row.get("method") or "<unknown>"


def _format_float(value: str | float | None, *, digits: int) -> str:
    number = _float_or_nan(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def _format_size_mb(value: str | float | None) -> str:
    number = _float_or_nan(value)
    if math.isnan(number):
        return ""
    return _format_size_number(number)


def _format_compression(value: str | float | None) -> str:
    number = _float_or_nan(value)
    if math.isnan(number):
        return ""
    return f"{number:.2f}x"


def _float_or_nan(value: str | float | None) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def _int_or_none(value: str | int | None) -> int | None:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
