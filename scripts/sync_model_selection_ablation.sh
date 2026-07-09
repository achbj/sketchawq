#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/sync_model_selection_ablation.sh <3|4|both> [options]

Options:
  --variant control|refine|both  Rows to sync. Default: both
  --volume NAME                  Modal volume. Default: sketchawq-artifacts
  --output-dir DIR               Bundle root. Default: artifacts/model_selection
  --dry-run                      Print commands without running them
  -h, --help                     Show this help
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

normalize_bits() {
  case "${1//-/_}" in
    3|3bit|3_bit) echo "3" ;;
    4|4bit|4_bit) echo "4" ;;
    both|all|3_4|34) echo "3 4" ;;
    *) die "bits must be 3, 4, or both; got '$1'" ;;
  esac
}

normalize_variants() {
  case "${1//-/_}" in
    control|base|baseline) echo "control" ;;
    refine|refinement|sketch) echo "refine" ;;
    both|all) echo "control refine" ;;
    *) die "--variant must be control, refine, or both; got '$1'" ;;
  esac
}

run_name_for() {
  echo "qwen3_4b_${1}bit_strict_g128_${2}_selection"
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

resolve_modal_bin() {
  if [[ -n "${MODAL_BIN:-}" ]]; then
    echo "$MODAL_BIN"
  elif [[ "$dry_run" == "1" ]]; then
    echo "modal"
  elif command -v modal >/dev/null 2>&1; then
    command -v modal
  else
    die "modal CLI not found; activate the Modal environment or set MODAL_BIN"
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ $# -ge 1 ]] || {
  usage
  exit 1
}

bits_list="$(normalize_bits "$1")"
shift

variant="both"
volume="sketchawq-artifacts"
output_root="artifacts/model_selection"
dry_run="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      [[ $# -ge 2 ]] || die "--variant requires a value"
      variant="$2"
      shift 2
      ;;
    --volume)
      [[ $# -ge 2 ]] || die "--volume requires a value"
      volume="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a value"
      output_root="$2"
      shift 2
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

variants_list="$(normalize_variants "$variant")"
timestamp="${SKETCHAWQ_SYNC_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
bundle="${output_root%/}/qwen3_4b_model_selection_${timestamp}"
runs_dir="${bundle}/runs"
results_dir="${bundle}/results"
modal_bin="$(resolve_modal_bin)"

read -r -a bits_array <<< "$bits_list"
read -r -a variants_array <<< "$variants_list"
run_paths=()

run_or_print mkdir -p "$runs_dir" "$results_dir"
for bits in "${bits_array[@]}"; do
  for selected_variant in "${variants_array[@]}"; do
    run_name="$(run_name_for "$bits" "$selected_variant")"
    run_or_print "$modal_bin" volume get --force "$volume" "/${run_name}" "$runs_dir"
    run_paths+=("${runs_dir}/${run_name}")
  done
done

run_or_print env "PYTHONPATH=${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m sketchawq.experiments.compare_methods \
  "${run_paths[@]}" \
  --format csv \
  --output "${results_dir}/model_selection_summary.csv"

run_or_print env "PYTHONPATH=${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m sketchawq.experiments.compare_methods \
  "${run_paths[@]}" \
  --table model-selection \
  --format csv \
  --output "${results_dir}/model_selection_metrics.csv"

if [[ "$dry_run" != "1" ]]; then
  echo "Synced model-selection artifacts to ${bundle}"
fi
