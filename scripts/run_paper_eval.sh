#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/run_paper_eval.sh <model> <3|4> [options]

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
  --method sketchawq|gptq|both  Which paper-eval row to run. Default: sketchawq
  --gptq                        Shortcut for --method gptq
  --both                        Shortcut for --method both
  --backend modal|local         Execution backend. Default: modal
  --modal                       Shortcut for --backend modal
  --local                       Shortcut for --backend local
  --gpu MODAL_GPU               Modal GPU request, e.g. L40S, A100, A100-80GB, H100, H100:2
  --eval wikitext|c4            Evaluation suite. Default: wikitext
  --dry-run                     Print commands without running them
  -h, --help                    Show this help

Examples:
  scripts/run_paper_eval.sh qwen3_8b 3 --method both --eval c4 --gpu L40S
  scripts/run_paper_eval.sh qwen3_1_7b 3 --method both --dry-run
  scripts/run_paper_eval.sh qwen3_8b 3 --method both --gpu L40S
  scripts/run_paper_eval.sh mistral_7b 4 --method both --gpu L40S
  scripts/run_paper_eval.sh qwen3_32b 4 --method both --gpu H100
  scripts/run_paper_eval.sh qwen3_14b 3 --method both --dry-run
  scripts/paper_eval/qwen3_32b_4bit.sh --gptq
  scripts/paper_eval/qwen2_5_1_5b_3bit.sh --local
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

normalize_bits() {
  case "${1//-/_}" in
    3|3bit|3_bit) echo "3" ;;
    4|4bit|4_bit) echo "4" ;;
    *) die "bits must be 3 or 4; got '$1'" ;;
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

ensure_config_exists() {
  [[ -f "$1" ]] || die "missing config: $1"
}

print_command() {
  printf 'dry-run:'
  printf ' %q' "$@"
  printf '\n'
}

run_one() {
  local runner="$1"
  local config="$2"
  local -a command

  ensure_config_exists "$config"

  case "${backend}:${runner}" in
    modal:sketchawq)
      if [[ -n "$gpu" ]]; then
        command=(env "SKETCHAWQ_MODAL_GPU=$gpu" modal run infra/modal/sketchawq_app.py --config "$config")
      else
        command=(modal run infra/modal/sketchawq_app.py --config "$config")
      fi
      ;;
    modal:gptq)
      if [[ -n "$gpu" ]]; then
        command=(env "SKETCHAWQ_MODAL_GPU=$gpu" modal run infra/modal/baseline_app.py --config "$config")
      else
        command=(modal run infra/modal/baseline_app.py --config "$config")
      fi
      ;;
    local:sketchawq)
      command=(env "PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}" python -m sketchawq.cli run "$config")
      ;;
    local:gptq)
      command=(env "PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}" python -m sketchawq.cli baseline "$config")
      ;;
    *) die "unsupported backend/runner: ${backend}/${runner}" ;;
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

[[ $# -ge 2 ]] || {
  usage
  exit 1
}

model="$(normalize_model "$1")"
bits="$(normalize_bits "$2")"
shift 2

method="sketchawq"
backend="modal"
dry_run="0"
gpu=""
eval_suite="wikitext"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      [[ $# -ge 2 ]] || die "--method requires a value"
      method="$2"
      shift 2
      ;;
    --gptq)
      method="gptq"
      shift
      ;;
    --both)
      method="both"
      shift
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
    --eval)
      [[ $# -ge 2 ]] || die "--eval requires a value"
      eval_suite="$(normalize_eval_suite "$2")"
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

case "$method" in
  sketchawq|gptq|both) ;;
  *) die "--method must be sketchawq, gptq, or both" ;;
esac

case "$backend" in
  modal|local) ;;
  *) die "--backend must be modal or local" ;;
esac

if [[ "$backend" == "local" && -n "$gpu" ]]; then
  die "--gpu only applies to the modal backend"
fi

case "$method" in
  sketchawq)
    run_one sketchawq "$(sketchawq_config "$model" "$bits" "$eval_suite")"
    ;;
  gptq)
    run_one gptq "$(gptq_config "$model" "$bits" "$eval_suite")"
    ;;
  both)
    run_one sketchawq "$(sketchawq_config "$model" "$bits" "$eval_suite")"
    run_one gptq "$(gptq_config "$model" "$bits" "$eval_suite")"
    ;;
esac
