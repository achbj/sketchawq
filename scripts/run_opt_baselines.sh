#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/run_opt_baselines.sh <3|4|both|all> [options]

Runs facebook/opt-2.7b GPTQModel GPTQ/AWQ baseline configs.

Options:
  --method gptq|awq|both  Baseline method rows to run. Default: both
  --backend modal|local   Execution backend. Default: modal
  --modal                 Shortcut for --backend modal
  --local                 Shortcut for --backend local
  --gpu MODAL_GPU         Modal GPU request, e.g. A10, A100, L40S, H100
  --include-awq-3bit      Also run the experimental AWQ 3-bit row. The default awq_torch backend rejects it.
  --dry-run               Print commands without running them
  -h, --help              Show this help

Examples:
  scripts/run_opt_baselines.sh both --method both --gpu A100
  scripts/run_opt_baselines.sh 4 --method awq --gpu A100
  scripts/run_opt_baselines.sh 3 --method gptq --local --dry-run
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
    both|all|3_4|34|3bit_4bit) echo "3 4" ;;
    *) die "bits must be 3, 4, both, or all; got '$1'" ;;
  esac
}

baseline_config() {
  case "$1:$2" in
    gptq:3) echo "configs/baselines/opt_2_7b_gptq_3bit.yaml" ;;
    gptq:4) echo "configs/baselines/opt_2_7b_gptq_4bit.yaml" ;;
    awq:3) echo "configs/baselines/opt_2_7b_awq_3bit.yaml" ;;
    awq:4) echo "configs/baselines/opt_2_7b_awq_4bit.yaml" ;;
    *) die "no OPT-2.7B baseline config for method '$1' and $2-bit" ;;
  esac
}

ensure_config_exists() {
  [[ -f "$1" ]] || die "missing config: $1"
}

print_command() {
  printf 'dry-run:'
  printf ' %q' "$@"
  printf '\n'
}

run_one() {
  local method="$1"
  local bits="$2"
  local config
  local -a command

  config="$(baseline_config "$method" "$bits")"
  ensure_config_exists "$config"

  case "$backend" in
    modal)
      if [[ -n "$gpu" ]]; then
        command=(env "SKETCHAWQ_MODAL_GPU=$gpu" modal run infra/modal/baseline_app.py --config "$config")
      else
        command=(modal run infra/modal/baseline_app.py --config "$config")
      fi
      ;;
    local)
      command=(env "PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}" python -m sketchawq.cli baseline "$config")
      ;;
    *) die "unsupported backend: $backend" ;;
  esac

  if [[ "$dry_run" == "1" ]]; then
    print_command "${command[@]}"
  else
    "${command[@]}"
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

method="both"
backend="modal"
dry_run="0"
gpu=""
include_awq_3bit="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      [[ $# -ge 2 ]] || die "--method requires a value"
      method="$2"
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
      [[ $# -ge 2 ]] || die "--gpu requires a Modal GPU value"
      gpu="$2"
      shift 2
      ;;
    --include-awq-3bit)
      include_awq_3bit="1"
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
  gptq) methods=(gptq) ;;
  awq) methods=(awq) ;;
  both) methods=(gptq awq) ;;
  *) die "--method must be gptq, awq, or both" ;;
esac

case "$backend" in
  modal|local) ;;
  *) die "--backend must be modal or local" ;;
esac

if [[ "$backend" == "local" && -n "$gpu" ]]; then
  die "--gpu only applies to the modal backend"
fi

read -r -a bits_array <<< "$bits_list"
ran_any="0"
for bits in "${bits_array[@]}"; do
  for baseline_method in "${methods[@]}"; do
    if [[ "$baseline_method" == "awq" && "$bits" == "3" && "$include_awq_3bit" != "1" ]]; then
      echo "warning: skipping AWQ 3-bit; GPTQModel awq_torch rejects non-4-bit kernels. Use --include-awq-3bit to retry explicitly." >&2
      continue
    fi
    run_one "$baseline_method" "$bits"
    ran_any="1"
  done
done

if [[ "$ran_any" != "1" ]]; then
  die "no supported OPT-2.7B baseline rows selected"
fi
