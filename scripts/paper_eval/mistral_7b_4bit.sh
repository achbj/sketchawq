#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${SCRIPT_DIR}/run_paper_eval.sh" mistral_7b 4 "$@"
