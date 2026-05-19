#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/run_inference_eval.sh <model> <3|4> [options]

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
  --variant fp16|sketchawq-dense  Runtime variant. Default: fp16
  --backend modal|local           Execution backend. Default: modal
  --modal                         Shortcut for --backend modal
  --local                         Shortcut for --backend local
  --gpu MODAL_GPU                 Modal GPU request, e.g. A100, A100-80GB, L40S, H100
  --prompt-lengths LIST           Comma/space list. Default: 128,512,2048
  --batch-sizes LIST              Comma/space list. Default: 1,4
  --generated-tokens N            Tokens to decode. Default: 128
  --warmup-runs N                 Warmup runs per shape. Default: 1
  --repeats N                     Measured repeats per shape. Default: 3
  --output-root PATH              Output root. Default: /artifacts/inference on Modal, artifacts/inference locally
  --name NAME                     Optional fixed run directory name
  --dry-run                       Print command without running it
  -h, --help                      Show this help

Examples:
  scripts/run_inference_eval.sh qwen3_4b 4 --variant fp16 --gpu A100
  scripts/run_inference_eval.sh qwen3_4b 4 --variant sketchawq-dense --gpu A100
  scripts/run_inference_eval.sh qwen3_4b 3 --variant fp16 --gpu A100
  scripts/run_inference_eval.sh qwen3_1_7b 4 --variant fp16 --gpu A100 --dry-run
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
  case "$1:$2" in
    qwen2_5_1_5b:3) echo "configs/paper/qwen2_5_1_5b_3bit.yaml" ;;
    qwen2_5_1_5b:4) echo "configs/paper/qwen2_5_1_5b_4bit.yaml" ;;
    qwen3_1_7b:3) echo "configs/paper/qwen3_1_7b_3bit.yaml" ;;
    qwen3_1_7b:4) echo "configs/paper/qwen3_1_7b_4bit.yaml" ;;
    qwen3_4b:3) echo "configs/paper/qwen3_4b_3bit.yaml" ;;
    qwen3_4b:4) echo "configs/paper/qwen3_4b_4bit.yaml" ;;
    qwen3_8b:3) echo "configs/paper/qwen3_8b_3bit.yaml" ;;
    qwen3_8b:4) echo "configs/paper/qwen3_8b_4bit.yaml" ;;
    qwen3_14b:3) echo "configs/paper/qwen3_14b_3bit.yaml" ;;
    qwen3_14b:4) echo "configs/paper/qwen3_14b_4bit.yaml" ;;
    qwen3_32b:3) echo "configs/paper/qwen3_32b_3bit.yaml" ;;
    qwen3_32b:4) echo "configs/paper/qwen3_32b_4bit.yaml" ;;
    llama_7b:3) echo "configs/paper/llama_7b_3bit.yaml" ;;
    llama_7b:4) echo "configs/paper/llama_7b_4bit.yaml" ;;
    mistral_7b:3) echo "configs/paper/mistral_7b_3bit.yaml" ;;
    mistral_7b:4) echo "configs/paper/mistral_7b_4bit.yaml" ;;
    *) die "no SketchAWQ config for $1 $2-bit" ;;
  esac
}

normalize_variant() {
  case "${1//_/-}" in
    fp16) echo "fp16" ;;
    sketchawq-dense) echo "sketchawq-dense" ;;
    *) die "--variant must be fp16 or sketchawq-dense; got '$1'" ;;
  esac
}

normalize_int_list() {
  local value="${1//,/ }"
  local output=()
  local item
  for item in $value; do
    [[ "$item" =~ ^[0-9]+$ ]] || die "expected integer list, got '$1'"
    output+=("$item")
  done
  ((${#output[@]} > 0)) || die "empty integer list"
  local IFS=,
  echo "${output[*]}"
}

ensure_config_exists() {
  [[ -f "$1" ]] || die "missing config: $1"
}

print_command() {
  printf 'dry-run:'
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  local config="$1"
  local -a command
  local output_root_arg="$output_root"

  ensure_config_exists "$config"

  if [[ "$backend" == "modal" ]]; then
    [[ -n "$output_root_arg" ]] || output_root_arg="/artifacts/inference"
    command=(
      modal run infra/modal/inference_app.py
      --config "$config"
      --variant "$variant"
      --prompt-lengths "$prompt_lengths"
      --batch-sizes "$batch_sizes"
      --generated-tokens "$generated_tokens"
      --warmup-runs "$warmup_runs"
      --repeats "$repeats"
      --output-root "$output_root_arg"
    )
    if [[ -n "$name" ]]; then
      command+=(--name "$name")
    fi
    if [[ -n "$gpu" ]]; then
      command=(env "SKETCHAWQ_MODAL_GPU=$gpu" "${command[@]}")
    fi
  elif [[ "$backend" == "local" ]]; then
    [[ -n "$output_root_arg" ]] || output_root_arg="artifacts/inference"
    command=(
      env "PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}"
      python -m sketchawq.cli inference "$config"
      --variant "$variant"
      --prompt-lengths
    )
    IFS=, read -r -a prompt_array <<< "$prompt_lengths"
    command+=("${prompt_array[@]}")
    command+=(--batch-sizes)
    IFS=, read -r -a batch_array <<< "$batch_sizes"
    command+=("${batch_array[@]}")
    command+=(
      --generated-tokens "$generated_tokens"
      --warmup-runs "$warmup_runs"
      --repeats "$repeats"
      --output-root "$output_root_arg"
    )
    if [[ -n "$name" ]]; then
      command+=(--name "$name")
    fi
  else
    die "--backend must be modal or local"
  fi

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

variant="fp16"
backend="modal"
dry_run="0"
gpu=""
prompt_lengths="128,512,2048"
batch_sizes="1,4"
generated_tokens="128"
warmup_runs="1"
repeats="3"
output_root=""
name=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      [[ $# -ge 2 ]] || die "--variant requires a value"
      variant="$(normalize_variant "$2")"
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
    --prompt-lengths)
      [[ $# -ge 2 ]] || die "--prompt-lengths requires a list"
      prompt_lengths="$(normalize_int_list "$2")"
      shift 2
      ;;
    --batch-sizes)
      [[ $# -ge 2 ]] || die "--batch-sizes requires a list"
      batch_sizes="$(normalize_int_list "$2")"
      shift 2
      ;;
    --generated-tokens)
      [[ $# -ge 2 ]] || die "--generated-tokens requires a value"
      [[ "$2" =~ ^[0-9]+$ ]] || die "--generated-tokens must be an integer"
      generated_tokens="$2"
      shift 2
      ;;
    --warmup-runs)
      [[ $# -ge 2 ]] || die "--warmup-runs requires a value"
      [[ "$2" =~ ^[0-9]+$ ]] || die "--warmup-runs must be an integer"
      warmup_runs="$2"
      shift 2
      ;;
    --repeats)
      [[ $# -ge 2 ]] || die "--repeats requires a value"
      [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "--repeats must be a positive integer"
      repeats="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || die "--output-root requires a path"
      output_root="$2"
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || die "--name requires a value"
      name="$2"
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

if [[ "$backend" == "local" && -n "$gpu" ]]; then
  die "--gpu only applies to the modal backend"
fi

run_command "$(sketchawq_config "$model" "$bits")"
