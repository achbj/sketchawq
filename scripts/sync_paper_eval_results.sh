#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/sync_paper_eval_results.sh <model> <3|4|both> [options]

Download paper-eval run artifacts from the Modal volume and build comparison tables/plots.

Models:
  qwen2_5_1_5b
  qwen3_1_7b
  qwen3_4b
  qwen3_8b
  qwen3_14b
  qwen3_32b
  llama_7b
  mistral_7b

Options:
  --method sketchawq|gptq|both  Which result folders to sync. Default: both
  --volume NAME                  Modal volume name. Default: sketchawq-artifacts
  --output-dir DIR               Local artifact download root. Default: artifacts/modal
  --results-dir DIR              Local tables/plots output root. Default: artifacts/paper_eval
  --name NAME                    Output file stem. Default: <model>_<bits>_paper_eval
  --eval wikitext|c4             Evaluation suite. Default: wikitext
  --no-plots                     Only download artifacts; skip tables and plots
  --dry-run                      Print commands without running them
  -h, --help                     Show this help

Environment:
  MODAL_BIN                      Optional Modal CLI path. Useful when modal is in a conda env.

Examples:
  scripts/sync_paper_eval_results.sh qwen3_8b both --method both --eval c4
  scripts/sync_paper_eval_results.sh qwen3_1_7b both --method both
  scripts/sync_paper_eval_results.sh mistral_7b both --method both
  scripts/sync_paper_eval_results.sh qwen3_8b both --method both
  scripts/sync_paper_eval_results.sh qwen3_8b 3 --method both --dry-run
  MODAL_BIN=/Users/achbj/miniconda3/envs/modal/bin/modal scripts/sync_paper_eval_results.sh qwen3_8b both
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

normalize_model() {
  case "${1//-/_}" in
    qwen2|qwen25|qwen2_5|qwen2_5_1_5b) echo "qwen2_5_1_5b" ;;
    qwen3_1_7b|qwen3_1_7|qwen3_17b|qwen3_17) echo "qwen3_1_7b" ;;
    qwen3_4b|qwen3_4) echo "qwen3_4b" ;;
    qwen3_8b|qwen3_8) echo "qwen3_8b" ;;
    qwen3_14b|qwen3_14) echo "qwen3_14b" ;;
    qwen3_32b|qwen3_32) echo "qwen3_32b" ;;
    llama|llama7b|llama_7b) echo "llama_7b" ;;
    mistral|mistral7b|mistral_7b) echo "mistral_7b" ;;
    *) die "unknown model '$1'" ;;
  esac
}

normalize_bits_selector() {
  case "${1//-/_}" in
    3|3bit|3_bit) echo "3" ;;
    4|4bit|4_bit) echo "4" ;;
    both|all|3_4|3and4) echo "both" ;;
    *) die "bits must be 3, 4, or both; got '$1'" ;;
  esac
}

bits_to_list() {
  case "$1" in
    3) echo "3" ;;
    4) echo "4" ;;
    both) echo "3 4" ;;
    *) die "unsupported bits selector '$1'" ;;
  esac
}

bits_slug() {
  case "$1" in
    3) echo "3bit" ;;
    4) echo "4bit" ;;
    both) echo "3_4bit" ;;
    *) die "unsupported bits selector '$1'" ;;
  esac
}

sketchawq_config() {
  case "$3:$1:$2" in
    c4:qwen3_8b:3) echo "configs/paper/qwen3_8b_3bit_c4_eval.yaml" ;;
    c4:qwen3_8b:4) echo "configs/paper/qwen3_8b_4bit_c4_eval.yaml" ;;
    c4:*) die "no C4 SketchAWQ eval config for $1 $2-bit" ;;
    wikitext:qwen2_5_1_5b:3) echo "configs/paper/qwen2_5_1_5b_3bit.yaml" ;;
    wikitext:qwen2_5_1_5b:4) echo "configs/paper/qwen2_5_1_5b_4bit.yaml" ;;
    wikitext:qwen3_1_7b:3) echo "configs/paper/qwen3_1_7b_3bit.yaml" ;;
    wikitext:qwen3_1_7b:4) echo "configs/paper/qwen3_1_7b_4bit.yaml" ;;
    wikitext:qwen3_4b:3) echo "configs/paper/qwen3_4b_3bit.yaml" ;;
    wikitext:qwen3_4b:4) echo "configs/paper/qwen3_4b_4bit.yaml" ;;
    wikitext:qwen3_8b:3) echo "configs/paper/qwen3_8b_3bit.yaml" ;;
    wikitext:qwen3_8b:4) echo "configs/paper/qwen3_8b_4bit.yaml" ;;
    wikitext:qwen3_14b:3) echo "configs/paper/qwen3_14b_3bit.yaml" ;;
    wikitext:qwen3_14b:4) echo "configs/paper/qwen3_14b_4bit.yaml" ;;
    wikitext:qwen3_32b:3) echo "configs/paper/qwen3_32b_3bit.yaml" ;;
    wikitext:qwen3_32b:4) echo "configs/paper/qwen3_32b_4bit.yaml" ;;
    wikitext:llama_7b:3) echo "configs/paper/llama_7b_3bit.yaml" ;;
    wikitext:llama_7b:4) echo "configs/paper/llama_7b_4bit.yaml" ;;
    wikitext:mistral_7b:3) echo "configs/paper/mistral_7b_3bit.yaml" ;;
    wikitext:mistral_7b:4) echo "configs/paper/mistral_7b_4bit.yaml" ;;
    *) die "no SketchAWQ paper-eval config for $1 $2-bit with eval '$3'" ;;
  esac
}

gptq_config() {
  case "$3:$1:$2" in
    c4:qwen3_8b:3) echo "configs/baselines/qwen3_8b_gptq_3bit_c4_eval.yaml" ;;
    c4:qwen3_8b:4) echo "configs/baselines/qwen3_8b_gptq_4bit_c4_eval.yaml" ;;
    c4:*) die "no C4 GPTQ eval config for $1 $2-bit" ;;
    wikitext:qwen2_5_1_5b:3) echo "configs/baselines/qwen2_5_1_5b_gptq_3bit.yaml" ;;
    wikitext:qwen2_5_1_5b:4) echo "configs/baselines/qwen2_5_1_5b_gptq_4bit.yaml" ;;
    wikitext:qwen3_1_7b:3) echo "configs/baselines/qwen3_1_7b_gptq_3bit.yaml" ;;
    wikitext:qwen3_1_7b:4) echo "configs/baselines/qwen3_1_7b_gptq_4bit.yaml" ;;
    wikitext:qwen3_4b:3) echo "configs/baselines/qwen3_4b_gptq_3bit.yaml" ;;
    wikitext:qwen3_4b:4) echo "configs/baselines/qwen3_4b_gptq_4bit.yaml" ;;
    wikitext:qwen3_8b:3) echo "configs/baselines/qwen3_8b_gptq_3bit.yaml" ;;
    wikitext:qwen3_8b:4) echo "configs/baselines/qwen3_8b_gptq_4bit.yaml" ;;
    wikitext:qwen3_14b:3) echo "configs/baselines/qwen3_14b_gptq_3bit.yaml" ;;
    wikitext:qwen3_14b:4) echo "configs/baselines/qwen3_14b_gptq_4bit.yaml" ;;
    wikitext:qwen3_32b:3) echo "configs/baselines/qwen3_32b_gptq_3bit.yaml" ;;
    wikitext:qwen3_32b:4) echo "configs/baselines/qwen3_32b_gptq_4bit.yaml" ;;
    wikitext:llama_7b:3) echo "configs/baselines/llama_7b_gptq_3bit.yaml" ;;
    wikitext:llama_7b:4) echo "configs/baselines/llama_7b_gptq_4bit.yaml" ;;
    wikitext:mistral_7b:3) echo "configs/baselines/mistral_7b_gptq_3bit.yaml" ;;
    wikitext:mistral_7b:4) echo "configs/baselines/mistral_7b_gptq_4bit.yaml" ;;
    *) die "no GPTQ paper-eval config for $1 $2-bit with eval '$3'" ;;
  esac
}

normalize_eval_suite() {
  case "${1//-/_}" in
    wikitext|wikitext2|wiki|wiki2) echo "wikitext" ;;
    c4|c4_validation|c4_eval) echo "c4" ;;
    *) die "--eval must be wikitext or c4; got '$1'" ;;
  esac
}

config_value() {
  local config="$1"
  local dotted_key="$2"
  [[ -f "$config" ]] || die "missing config: $config"
  python - "$config" "$dotted_key" <<'PY'
from pathlib import Path
import sys

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
value = config
for key in sys.argv[2].split("."):
    if not isinstance(value, dict) or key not in value:
        raise SystemExit(f"missing config key {sys.argv[2]} in {sys.argv[1]}")
    value = value[key]
print(value)
PY
}

resolve_modal_bin() {
  if [[ -n "${MODAL_BIN:-}" ]]; then
    echo "$MODAL_BIN"
    return
  fi
  if [[ "$dry_run" == "1" ]]; then
    echo "modal"
    return
  fi
  if command -v modal >/dev/null 2>&1; then
    command -v modal
    return
  fi
  for candidate in \
    "/Users/achbj/miniconda3/envs/modal/bin/modal" \
    "/Users/achbj/miniconda3/envs/modal2/bin/modal"
  do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done
  die "modal CLI not found. Activate your Modal env or set MODAL_BIN=/path/to/modal."
}

print_command() {
  printf 'dry-run:'
  printf ' %q' "$@"
  printf '\n'
}

run_or_print() {
  if [[ "$dry_run" == "1" ]]; then
    print_command "$@"
  else
    "$@"
  fi
}

sync_config() {
  local modal_bin="$1"
  local config="$2"
  local run_name
  local destination

  run_name="$(config_value "$config" "output.run_name")"
  destination="${output_dir%/}/${run_name}"
  synced_paths+=("$destination")

  run_or_print mkdir -p "$output_dir"
  run_or_print "$modal_bin" volume get --force "$volume" "/${run_name}" "$output_dir"
}

build_result_artifacts() {
  local first_config="$1"
  local model_repo_id
  local benchmark
  local -a python_env

  model_repo_id="$(config_value "$first_config" "model.repo_id")"
  benchmark="$(config_value "$first_config" "evaluation.benchmark_name")"
  python_env=(env "PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}")

  run_or_print mkdir -p "$results_dir"
  run_or_print "${python_env[@]}" python -m sketchawq.experiments.compare_methods \
    "${synced_paths[@]}" \
    --format csv \
    --output "${results_dir%/}/${name}_summary.csv"
  run_or_print "${python_env[@]}" python -m sketchawq.experiments.compare_methods \
    "${synced_paths[@]}" \
    --table evaluation \
    --format csv \
    --output "${results_dir%/}/${name}_evaluation_metrics.csv"
  run_or_print "${python_env[@]}" python -m sketchawq.experiments.plot_benchmark_board \
    "${synced_paths[@]}" \
    --model "$model_repo_id" \
    --benchmark "$benchmark" \
    --output-dir "$results_dir" \
    --name "$name"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ $# -ge 2 ]] || {
  usage
  exit 1
}

model="$(normalize_model "$1")"
bits_selector="$(normalize_bits_selector "$2")"
shift 2

method="both"
volume="sketchawq-artifacts"
output_dir="artifacts/modal"
results_dir="artifacts/paper_eval"
name="${model}_$(bits_slug "$bits_selector")_paper_eval"
plots="1"
dry_run="0"
eval_suite="wikitext"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      [[ $# -ge 2 ]] || die "--method requires a value"
      method="$2"
      shift 2
      ;;
    --volume)
      [[ $# -ge 2 ]] || die "--volume requires a value"
      volume="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a value"
      output_dir="$2"
      shift 2
      ;;
    --results-dir)
      [[ $# -ge 2 ]] || die "--results-dir requires a value"
      results_dir="$2"
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || die "--name requires a value"
      name="$2"
      shift 2
      ;;
    --eval)
      [[ $# -ge 2 ]] || die "--eval requires a value"
      eval_suite="$(normalize_eval_suite "$2")"
      if [[ "$name" == "${model}_$(bits_slug "$bits_selector")_paper_eval" ]]; then
        name="${model}_$(bits_slug "$bits_selector")_${eval_suite}_eval"
      fi
      shift 2
      ;;
    --no-plots)
      plots="0"
      shift
      ;;
    --dry-run)
      dry_run="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option '$1'"
      ;;
  esac
done

case "$method" in
  sketchawq|gptq|both) ;;
  *) die "--method must be sketchawq, gptq, or both" ;;
esac

modal_bin="$(resolve_modal_bin)"
synced_paths=()
first_config=""

for bits in $(bits_to_list "$bits_selector"); do
  case "$method" in
    sketchawq)
      config="$(sketchawq_config "$model" "$bits" "$eval_suite")"
      [[ -n "$first_config" ]] || first_config="$config"
      sync_config "$modal_bin" "$config"
      ;;
    gptq)
      config="$(gptq_config "$model" "$bits" "$eval_suite")"
      [[ -n "$first_config" ]] || first_config="$config"
      sync_config "$modal_bin" "$config"
      ;;
    both)
      config="$(sketchawq_config "$model" "$bits" "$eval_suite")"
      [[ -n "$first_config" ]] || first_config="$config"
      sync_config "$modal_bin" "$config"
      sync_config "$modal_bin" "$(gptq_config "$model" "$bits" "$eval_suite")"
      ;;
  esac
done

if [[ "$plots" == "1" ]]; then
  build_result_artifacts "$first_config"
fi
