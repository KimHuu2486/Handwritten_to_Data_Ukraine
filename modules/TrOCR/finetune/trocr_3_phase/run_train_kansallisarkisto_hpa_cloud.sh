#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional env file. Create it from env_hpa_cloud.example, then set:
#   ENV_FILE=/path/to/.env.hpa ./run_train_kansallisarkisto_hpa_cloud.sh
ENV_FILE="${ENV_FILE:-}"
if [[ -n "$ENV_FILE" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "ENV_FILE does not exist: $ENV_FILE" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "Warning: HF_TOKEN is not set. Public model downloads may still work." >&2
fi

"$PYTHON_BIN" "$SCRIPT_DIR/train_kansallisarkisto_hpa_3_phase.py"
