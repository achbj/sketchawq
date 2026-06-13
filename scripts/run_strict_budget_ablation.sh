#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/run_strict_budget_ablation.sh <3|4|both> [options]

Options:
  --variant control|refine|both  Rows to run. Default: both
  --backend modal|local          Execution backend. Default: modal
  --modal                        Shortcut for --backend modal
  --local                        Shortcut for --backend local
  --gpu MODAL_GPU                Modal GPU request. Default: A100
  --dry-run                      Print the command without running it
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

config_for() {
  local bits="$1"
  local variant="$2"
  echo "configs/experiments/strict_budget/qwen3_4b_${bits}bit_${variant}.yaml"
}

print_command() {
  printf 'dry-run:'
  printf ' %q' "$@"
  printf '\n'
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
backend="modal"
gpu="A100"
gpu_was_set="0"
dry_run="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      [[ $# -ge 2 ]] || die "--variant requires a value"
      variant="$2"
      shift 2
      ;;
    --backend)
      [[ $# -ge 2 ]] || die "--backend requires a value"
      backend="$2"
      shift 2
      ;;
    --modal)
      backend="modal"
      shift
      ;;
    --local)
      backend="local"
      shift
      ;;
    --gpu)
      [[ $# -ge 2 ]] || die "--gpu requires a value"
      gpu="$2"
      gpu_was_set="1"
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

case "$backend" in
  modal|local) ;;
  *) die "--backend must be modal or local" ;;
esac
if [[ "$backend" == "local" && "$gpu_was_set" == "1" ]]; then
  die "--gpu only applies to the modal backend"
fi

variants_list="$(normalize_variants "$variant")"
read -r -a bits_array <<< "$bits_list"
read -r -a variants_array <<< "$variants_list"
configs=()
for bits in "${bits_array[@]}"; do
  for selected_variant in "${variants_array[@]}"; do
    config="$(config_for "$bits" "$selected_variant")"
    [[ -f "${REPO_ROOT}/${config}" ]] || die "config not found: ${REPO_ROOT}/${config}"
    configs+=("$config")
  done
done

case "$backend" in
  modal)
    config_csv="$(IFS=,; echo "${configs[*]}")"
    command=(
      env "SKETCHAWQ_MODAL_GPU=$gpu"
      modal run "${REPO_ROOT}/infra/modal/sketchawq_app.py"
      --configs "$config_csv"
    )
    ;;
  local)
    command=(
      env "PYTHONPATH=${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
      python -m sketchawq.cli run
    )
    for config in "${configs[@]}"; do
      command+=("${REPO_ROOT}/${config}")
    done
    ;;
esac

if [[ "$dry_run" == "1" ]]; then
  print_command "${command[@]}"
else
  "${command[@]}"
fi
