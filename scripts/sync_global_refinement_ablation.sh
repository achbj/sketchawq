#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/sync_global_refinement_ablation.sh <3|4|both> [options]

Options:
  --volume NAME      Modal volume. Default: sketchawq-artifacts
  --output-dir DIR   Bundle root. Default: artifacts/global_refinement
  --dry-run          Print commands without running them
  -h, --help         Show this help
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

strict_run_name_for() {
  echo "qwen3_4b_${1}bit_strict_g128_control"
}

nystrom_run_name_for() {
  echo "qwen3_4b_${1}bit_nystrom_residual_g128_control"
}

global_run_name_for() {
  echo "qwen3_4b_${1}bit_nystrom_global_refine_g128"
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

volume="sketchawq-artifacts"
output_root="artifacts/global_refinement"
dry_run="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
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

timestamp="${SKETCHAWQ_SYNC_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
bundle="${output_root%/}/qwen3_4b_global_refinement_${timestamp}"
runs_dir="${bundle}/runs"
results_dir="${bundle}/results"
modal_bin="$(resolve_modal_bin)"

read -r -a bits_array <<< "$bits_list"
run_paths=()

run_or_print mkdir -p "$runs_dir" "$results_dir"
for bits in "${bits_array[@]}"; do
  strict_run="$(strict_run_name_for "$bits")"
  nystrom_run="$(nystrom_run_name_for "$bits")"
  global_run="$(global_run_name_for "$bits")"
  run_or_print "$modal_bin" volume get --force "$volume" "/${strict_run}" "$runs_dir"
  run_or_print "$modal_bin" volume get --force "$volume" "/${nystrom_run}" "$runs_dir"
  run_or_print "$modal_bin" volume get --force "$volume" "/${global_run}" "$runs_dir"
  run_paths+=(
    "${runs_dir}/${strict_run}"
    "${runs_dir}/${nystrom_run}"
    "${runs_dir}/${global_run}"
  )
done

run_or_print env "PYTHONPATH=${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m sketchawq.experiments.compare_methods \
  "${run_paths[@]}" \
  --format csv \
  --output "${results_dir}/global_refinement_summary.csv"

run_or_print env "PYTHONPATH=${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m sketchawq.experiments.compare_methods \
  "${run_paths[@]}" \
  --table evaluation \
  --format csv \
  --output "${results_dir}/global_refinement_evaluation_metrics.csv"

if [[ "$dry_run" != "1" ]]; then
  echo "Synced global-refinement artifacts to ${bundle}"
fi
