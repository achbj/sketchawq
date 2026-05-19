from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
from pathlib import Path
from typing import Any


INFERENCE_SUMMARY_FIELDS = [
    "model_repo_id",
    "model_revision",
    "bits_context",
    "model_variant",
    "method",
    "bits",
    "group_size",
    "weight_runtime_format",
    "inference_backend",
    "prompt_length",
    "prompt_tokens",
    "generated_tokens",
    "decode_tokens",
    "total_tokens",
    "batch_size",
    "warmup_runs",
    "sample_count",
    "run_count",
    "run_names",
    "gpu_name",
    "device",
    "prefill_tokens_per_second_min",
    "prefill_tokens_per_second_median",
    "prefill_tokens_per_second_max",
    "decode_tokens_per_second_min",
    "decode_tokens_per_second_median",
    "decode_tokens_per_second_max",
    "generate_tokens_per_second_min",
    "generate_tokens_per_second_median",
    "generate_tokens_per_second_max",
    "prefill_vs_fp16_ratio",
    "decode_vs_fp16_ratio",
    "generate_vs_fp16_ratio",
    "time_to_first_token_ms_min",
    "time_to_first_token_ms_median",
    "time_to_first_token_ms_max",
    "peak_memory_mb_min",
    "peak_memory_mb_median",
    "peak_memory_mb_max",
    "quantization_seconds_median",
    "fp16_run_names",
    "created_at_min",
    "created_at_max",
    "metrics_files",
]

_GROUP_FIELDS = [
    "model_repo_id",
    "model_revision",
    "bits_context",
    "model_variant",
    "method",
    "bits",
    "group_size",
    "weight_runtime_format",
    "inference_backend",
    "prompt_length",
    "generated_tokens",
    "batch_size",
    "gpu_name",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize repeated inference_metrics.csv artifacts into median/min/max tables."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Inference artifact roots, run directories, or inference_metrics.csv files.",
    )
    parser.add_argument("--output-dir", default="artifacts/inference_summary")
    parser.add_argument("--name", default="inference_summary")
    parser.add_argument("--model", default=None, help="Optional model repo id filter.")
    args = parser.parse_args()

    outputs = summarize_inference_artifacts(
        args.inputs,
        output_dir=args.output_dir,
        name=args.name,
        model=args.model,
    )
    print(f"Wrote {outputs['csv']}")
    print(f"Wrote {outputs['json']}")
    print(f"Wrote {outputs['html']}")


def summarize_inference_artifacts(
    inputs: list[str],
    *,
    output_dir: str | Path = "artifacts/inference_summary",
    name: str = "inference_summary",
    model: str | None = None,
) -> dict[str, str]:
    rows = load_inference_rows(inputs)
    summary_rows = build_inference_summary_rows(rows, model=model)
    if not summary_rows:
        raise ValueError("No inference sample rows found for the requested inputs.")
    return write_inference_summary(summary_rows, output_dir, name=name)


def load_inference_rows(inputs: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metrics_path in _metric_paths(inputs):
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                work = dict(row)
                work["metrics_file"] = str(metrics_path)
                rows.append(work)
    return rows


def build_inference_summary_rows(
    rows: list[dict[str, str]],
    *,
    model: str | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        if model is not None and row.get("model_repo_id") != model:
            continue
        if (row.get("measurement_kind") or "sample") != "sample":
            continue
        work = dict(row)
        work["bits_context"] = _bits_context(work)
        key = tuple(work.get(field, "") for field in _GROUP_FIELDS)
        groups.setdefault(key, []).append(work)

    summary_rows = [_summarize_group(group) for group in groups.values()]
    _add_fp16_ratios(summary_rows)
    return sorted(summary_rows, key=_summary_sort_key)


def write_inference_summary(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    name: str = "inference_summary",
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    json_path = output_dir / f"{name}.json"
    html_path = output_dir / f"{name}.html"

    fields = [field for field in INFERENCE_SUMMARY_FIELDS if any(field in row for row in rows)]
    extra_fields = sorted({field for row in rows for field in row if field not in fields})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(rows), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "html": str(html_path)}


def _metric_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            paths.append(path)
            continue
        if not path.exists():
            continue
        for candidate in sorted(path.rglob("inference_metrics.csv")):
            if candidate.with_name("inference_report.json").exists():
                paths.append(candidate)
    return _dedupe_paths(paths)


def _summarize_group(rows: list[dict[str, str]]) -> dict[str, Any]:
    first = rows[0]
    run_names = sorted({row.get("run_name", "") for row in rows if row.get("run_name")})
    metrics_files = sorted({row.get("metrics_file", "") for row in rows if row.get("metrics_file")})
    created_at = sorted({row.get("created_at", "") for row in rows if row.get("created_at")})
    summary: dict[str, Any] = {
        "model_repo_id": first.get("model_repo_id"),
        "model_revision": first.get("model_revision"),
        "bits_context": first.get("bits_context"),
        "model_variant": first.get("model_variant"),
        "method": first.get("method"),
        "bits": first.get("bits"),
        "group_size": first.get("group_size"),
        "weight_runtime_format": first.get("weight_runtime_format"),
        "inference_backend": first.get("inference_backend"),
        "prompt_length": first.get("prompt_length"),
        "prompt_tokens": _same_or_join(rows, "prompt_tokens"),
        "generated_tokens": first.get("generated_tokens"),
        "decode_tokens": _same_or_join(rows, "decode_tokens"),
        "total_tokens": _same_or_join(rows, "total_tokens"),
        "batch_size": first.get("batch_size"),
        "warmup_runs": _same_or_join(rows, "warmup_runs"),
        "sample_count": len(rows),
        "run_count": len(run_names),
        "run_names": "; ".join(run_names),
        "gpu_name": first.get("gpu_name"),
        "device": _same_or_join(rows, "device"),
        "created_at_min": created_at[0] if created_at else "",
        "created_at_max": created_at[-1] if created_at else "",
        "metrics_files": "; ".join(metrics_files),
    }
    for field in (
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "generate_tokens_per_second",
        "time_to_first_token_ms",
        "peak_memory_mb",
    ):
        _add_triplet(summary, rows, field)
    values = _numbers(rows, "quantization_seconds")
    summary["quantization_seconds_median"] = float(statistics.median(values)) if values else ""
    return summary


def _add_fp16_ratios(rows: list[dict[str, Any]]) -> None:
    fp16_rows = [row for row in rows if _is_fp16(row)]
    for row in rows:
        baseline = row if _is_fp16(row) else _find_fp16_baseline(row, fp16_rows)
        if baseline is None:
            row["prefill_vs_fp16_ratio"] = ""
            row["decode_vs_fp16_ratio"] = ""
            row["generate_vs_fp16_ratio"] = ""
            row["fp16_run_names"] = ""
            continue
        row["prefill_vs_fp16_ratio"] = _ratio(
            row.get("prefill_tokens_per_second_median"),
            baseline.get("prefill_tokens_per_second_median"),
        )
        row["decode_vs_fp16_ratio"] = _ratio(
            row.get("decode_tokens_per_second_median"),
            baseline.get("decode_tokens_per_second_median"),
        )
        row["generate_vs_fp16_ratio"] = _ratio(
            row.get("generate_tokens_per_second_median"),
            baseline.get("generate_tokens_per_second_median"),
        )
        row["fp16_run_names"] = baseline.get("run_names", "")


def _find_fp16_baseline(row: dict[str, Any], fp16_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    strict = [
        candidate
        for candidate in fp16_rows
        if candidate.get("model_repo_id") == row.get("model_repo_id")
        and candidate.get("prompt_length") == row.get("prompt_length")
        and candidate.get("generated_tokens") == row.get("generated_tokens")
        and candidate.get("batch_size") == row.get("batch_size")
        and candidate.get("gpu_name") == row.get("gpu_name")
        and candidate.get("bits_context") == row.get("bits_context")
    ]
    if strict:
        return sorted(strict, key=lambda item: str(item.get("run_names", "")))[0]
    fallback = [
        candidate
        for candidate in fp16_rows
        if candidate.get("model_repo_id") == row.get("model_repo_id")
        and candidate.get("prompt_length") == row.get("prompt_length")
        and candidate.get("generated_tokens") == row.get("generated_tokens")
        and candidate.get("batch_size") == row.get("batch_size")
        and candidate.get("gpu_name") == row.get("gpu_name")
    ]
    if fallback:
        return sorted(fallback, key=lambda item: str(item.get("run_names", "")))[0]
    return None


def _render_html(rows: list[dict[str, Any]]) -> str:
    context = _render_context(rows)
    body_rows = "\n".join(_render_table_row(row) for row in rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SketchAWQ Inference Summary</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #16202a;
      --muted: #5f6d7a;
      --line: #d8e0e8;
      --panel: #f7f9fb;
      --accent: #1f7a8c;
    }}
    body {{
      margin: 0;
      background: #fff;
      color: var(--ink);
      font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1240px;
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
      max-width: 900px;
    }}
    .context {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
      margin: 22px 0 28px;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px 16px;
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0;
      text-transform: uppercase;
    }}
    .value {{
      margin-top: 4px;
      font-size: 17px;
      font-weight: 650;
      overflow-wrap: anywhere;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
    }}
    th, td {{
      padding: 10px 9px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #edf3f7;
      font-size: 13px;
    }}
    tbody tr:nth-child(odd) {{
      background: #f9fbfc;
    }}
    .ratio {{
      color: var(--accent);
      font-weight: 700;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
    }}
  </style>
</head>
<body>
<main>
  <h1>Inference Summary Board</h1>
  <p>Rows use sample measurements only and report min/median/max across repeats. Ratios compare median throughput against the matching FP16 row when one is present.</p>
  {context}
  <table>
    <thead>
      <tr>
        <th>Bits</th>
        <th>Variant</th>
        <th>Prompt</th>
        <th>Batch</th>
        <th>Samples</th>
        <th>Prefill tok/s</th>
        <th>Decode tok/s</th>
        <th>Generate tok/s</th>
        <th>Vs FP16</th>
        <th>TTFT ms</th>
        <th>Peak MB</th>
        <th>GPU</th>
        <th>Runs</th>
      </tr>
    </thead>
    <tbody>
      {body_rows}
    </tbody>
  </table>
</main>
</body>
</html>
"""


def _render_context(rows: list[dict[str, Any]]) -> str:
    models = _compact_values(row.get("model_repo_id") for row in rows)
    gpus = _compact_values(row.get("gpu_name") for row in rows)
    variants = _compact_values(row.get("model_variant") for row in rows)
    samples = sum(int(row.get("sample_count") or 0) for row in rows)
    cards = [
        ("Models", models),
        ("GPUs", gpus),
        ("Variants", variants),
        ("Measured Samples", str(samples)),
    ]
    return '<section class="context">' + "".join(_render_panel(label, value) for label, value in cards) + "</section>"


def _render_panel(label: str, value: str) -> str:
    return (
        '<div class="panel">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        "</div>"
    )


def _render_table_row(row: dict[str, Any]) -> str:
    return f"""<tr>
  <td>{html.escape(str(row.get("bits_context") or row.get("bits") or ""))}</td>
  <td><code>{html.escape(str(row.get("model_variant") or ""))}</code><div class="muted">{html.escape(str(row.get("weight_runtime_format") or ""))}</div></td>
  <td>{html.escape(str(row.get("prompt_length") or ""))}</td>
  <td>{html.escape(str(row.get("batch_size") or ""))}</td>
  <td>{html.escape(str(row.get("sample_count") or ""))}</td>
  <td>{_triplet_text(row, "prefill_tokens_per_second")}</td>
  <td>{_triplet_text(row, "decode_tokens_per_second")}</td>
  <td>{_triplet_text(row, "generate_tokens_per_second")}</td>
  <td class="ratio">{_ratio_text(row.get("generate_vs_fp16_ratio"))}</td>
  <td>{_triplet_text(row, "time_to_first_token_ms")}</td>
  <td>{_fmt_number(row.get("peak_memory_mb_max"))}</td>
  <td>{html.escape(str(row.get("gpu_name") or ""))}</td>
  <td><code>{html.escape(str(row.get("run_names") or ""))}</code></td>
</tr>"""


def _add_triplet(summary: dict[str, Any], rows: list[dict[str, str]], field: str) -> None:
    values = _numbers(rows, field)
    if not values:
        summary[f"{field}_min"] = ""
        summary[f"{field}_median"] = ""
        summary[f"{field}_max"] = ""
        return
    summary[f"{field}_min"] = min(values)
    summary[f"{field}_median"] = float(statistics.median(values))
    summary[f"{field}_max"] = max(values)


def _numbers(rows: list[dict[str, str]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def _ratio(numerator: Any, denominator: Any) -> float | str:
    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return ""
    if denominator_value <= 0:
        return ""
    return numerator_value / denominator_value


def _ratio_text(value: Any) -> str:
    try:
        return f"{float(value):.3f}x"
    except (TypeError, ValueError):
        return ""


def _triplet_text(row: dict[str, Any], field: str) -> str:
    median_value = _fmt_number(row.get(f"{field}_median"))
    min_value = _fmt_number(row.get(f"{field}_min"))
    max_value = _fmt_number(row.get(f"{field}_max"))
    if not median_value:
        return ""
    return (
        f"{html.escape(median_value)}"
        f'<div class="muted">{html.escape(min_value)} - {html.escape(max_value)}</div>'
    )


def _fmt_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.3f}"


def _same_or_join(rows: list[dict[str, str]], field: str) -> str:
    values = sorted({row.get(field, "") for row in rows if row.get(field) not in (None, "")})
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return "; ".join(values)


def _bits_context(row: dict[str, str]) -> str:
    bits = str(row.get("bits") or "").strip()
    if bits:
        return bits
    run_name = str(row.get("run_name") or "")
    match = re.search(r"(?:^|_)([34])bit(?:_|$)", run_name)
    return match.group(1) if match else ""


def _is_fp16(row: dict[str, Any]) -> bool:
    return (
        str(row.get("model_variant") or "").lower() == "fp16"
        or str(row.get("weight_runtime_format") or "").lower() == "fp16"
        or str(row.get("method") or "").upper() == "FP16"
    )


def _compact_values(values: Any) -> str:
    unique = sorted({str(value) for value in values if value not in (None, "")})
    if not unique:
        return ""
    if len(unique) <= 3:
        return ", ".join(unique)
    return f"{', '.join(unique[:3])}, +{len(unique) - 3} more"


def _summary_sort_key(row: dict[str, Any]) -> tuple:
    return (
        str(row.get("model_repo_id") or ""),
        _int_key(row.get("bits_context")),
        _int_key(row.get("prompt_length")),
        _int_key(row.get("batch_size")),
        str(row.get("model_variant") or ""),
        str(row.get("run_names") or ""),
    )


def _int_key(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    unique = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


if __name__ == "__main__":
    main()
