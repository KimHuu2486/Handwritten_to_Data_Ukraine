#!/usr/bin/env bash
set -euo pipefail

# Run single-phase Kansallisarkisto HPA training on Linux/Thunder Compute.
#
# Expected dataset layout by default:
#   /home/ubuntu/dataset/train/metadata.jsonl
#   /home/ubuntu/dataset/train/images/...
#
# Usage:
#   chmod +x run_train_kansallisarkisto_hpa_single_train_cloud.sh
#   ./run_train_kansallisarkisto_hpa_single_train_cloud.sh
#
# Optional:
#   ENV_FILE=/path/to/.env.hpa ./run_train_kansallisarkisto_hpa_single_train_cloud.sh
#
# Common env overrides:
#   HPA_DATA_ROOT=/home/ubuntu/dataset
#   HPA_TRAIN_DIR=/home/ubuntu/dataset/train
#   HPA_OUTPUT_DIR=/home/ubuntu/outputs_hpa_single
#   HPA_FINAL_DIR=/home/ubuntu/outputs_hpa_single/final_cyrillic_htr_model
#   HF_CACHE_DIR=/home/ubuntu/hf_cache
#   HPA_TRAIN_EPOCHS=10
#   HPA_TRAIN_LR=1e-5
#   HPA_TRAIN_BATCH_SIZE=32
#   HPA_EVAL_BATCH_SIZE=32
#   HPA_VAL_RATIO=0.10
#   HPA_REBUILD_SPLITS=false
#   HPA_TRAIN_MAX_PER_CLASS=0
#
# Venv overrides:
#   VENV_DIR=/home/ubuntu/.venvs/kansallisarkisto_hpa
#   REQUIREMENTS_FILE=/path/to/requirements_kansallisarkisto.txt
#   FORCE_REINSTALL_REQUIREMENTS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional env file. Example .env.hpa:
#   HPA_DATA_ROOT=/home/ubuntu/dataset
#   HPA_TRAIN_DIR=/home/ubuntu/dataset/train
#   HPA_OUTPUT_DIR=/home/ubuntu/outputs_hpa_single
#   HF_CACHE_DIR=/home/ubuntu/hf_cache
#   HF_TOKEN=hf_xxx
ENV_FILE="${ENV_FILE:-}"
if [[ -n "$ENV_FILE" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "[ERROR] ENV_FILE does not exist: $ENV_FILE" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "[INFO] Loaded ENV_FILE: $ENV_FILE"
fi

# ---------------------------------------------------------------------
# 1. Create/reuse Python virtual environment
# ---------------------------------------------------------------------

SYSTEM_PYTHON_BIN="${SYSTEM_PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv_kansallisarkisto_hpa}"

if ! command -v "$SYSTEM_PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] System Python executable not found: $SYSTEM_PYTHON_BIN" >&2
  echo "        Try: SYSTEM_PYTHON_BIN=python ./run_train_kansallisarkisto_hpa_single_train_cloud.sh" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[INFO] Creating virtual environment: $VENV_DIR"
  "$SYSTEM_PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[INFO] Reusing virtual environment: $VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

if [[ ! -x "$PYTHON_BIN" || ! -x "$PIP_BIN" ]]; then
  echo "[ERROR] Invalid venv. Missing python/pip in: $VENV_DIR" >&2
  exit 1
fi

echo "[INFO] Upgrading pip/setuptools/wheel..."
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------
# 2. Install requirements
# ---------------------------------------------------------------------

REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$SCRIPT_DIR/requirements_kansallisarkisto.txt}"

# Allow uploaded/downloaded names too.
if [[ ! -f "$REQUIREMENTS_FILE" && -f "$SCRIPT_DIR/requirements_kansallisarkisto(3).txt" ]]; then
  REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_kansallisarkisto(3).txt"
elif [[ ! -f "$REQUIREMENTS_FILE" && -f "$SCRIPT_DIR/requirements_kansallisarkisto(2).txt" ]]; then
  REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_kansallisarkisto(2).txt"
elif [[ ! -f "$REQUIREMENTS_FILE" && -f "$SCRIPT_DIR/requirements_kansallisarkisto(1).txt" ]]; then
  REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_kansallisarkisto(1).txt"
fi

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "[ERROR] Requirements file not found." >&2
  echo "        Expected: $SCRIPT_DIR/requirements_kansallisarkisto.txt" >&2
  echo "        Or set REQUIREMENTS_FILE=/path/to/requirements_kansallisarkisto.txt" >&2
  exit 1
fi

REQ_STAMP="$VENV_DIR/.requirements_kansallisarkisto_installed"

if [[ ! -f "$REQ_STAMP" || "${FORCE_REINSTALL_REQUIREMENTS:-0}" == "1" ]]; then
  echo "[INFO] Installing requirements from: $REQUIREMENTS_FILE"
  "$PIP_BIN" install -r "$REQUIREMENTS_FILE"
  date > "$REQ_STAMP"
else
  echo "[INFO] Requirements already installed. Set FORCE_REINSTALL_REQUIREMENTS=1 to reinstall."
fi

# ---------------------------------------------------------------------
# 3. Resolve training script
# ---------------------------------------------------------------------

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/train_kansallisarkisto_hpa_single_train.py}"
if [[ ! -f "$TRAIN_SCRIPT" && -f "$SCRIPT_DIR/train_kansallisarkisto_hpa_single_train(1).py" ]]; then
  TRAIN_SCRIPT="$SCRIPT_DIR/train_kansallisarkisto_hpa_single_train(1).py"
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "[ERROR] Training script not found." >&2
  echo "        Expected: $SCRIPT_DIR/train_kansallisarkisto_hpa_single_train.py" >&2
  echo "        Or set TRAIN_SCRIPT=/path/to/train_kansallisarkisto_hpa_single_train.py" >&2
  exit 1
fi

# ---------------------------------------------------------------------
# 4. Hugging Face token handling
# ---------------------------------------------------------------------

# HF_KEY is not the standard Hugging Face variable, but map it for convenience.
# Preferred names are HF_TOKEN or HUGGINGFACE_HUB_TOKEN.
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" && -n "${HF_KEY:-}" ]]; then
  export HF_TOKEN="$HF_KEY"
  echo "[INFO] Mapped HF_KEY -> HF_TOKEN."
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "[WARN] HF_TOKEN/HUGGINGFACE_HUB_TOKEN is not set."
  echo "       This is usually OK for public Hugging Face models."
  echo "       Set HF_TOKEN only if the model/cache access needs authentication or you hit rate limits."
fi

# ---------------------------------------------------------------------
# 5. Runtime defaults
# ---------------------------------------------------------------------

# IMPORTANT:
# If your HPA_DATA_ROOT is /home/ubuntu/dataset, the default train dir is:
#   /home/ubuntu/dataset/train
export HPA_DATA_ROOT="${HPA_DATA_ROOT:-${DATA_ROOT:-/home/ubuntu/dataset}}"
export HPA_TRAIN_DIR="${HPA_TRAIN_DIR:-$HPA_DATA_ROOT/train}"
export HPA_OUTPUT_DIR="${HPA_OUTPUT_DIR:-${OUTPUT_DIR:-/home/ubuntu/outputs_hpa_single}}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-/home/ubuntu/hf_cache}}"

# Avoid tokenizer fork warnings/deadlocks with dataloader workers.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "[INFO] Single-phase Kansallisarkisto HPA training"
echo "============================================================"
echo "[INFO] System Python : $SYSTEM_PYTHON_BIN"
echo "[INFO] Venv          : $VENV_DIR"
echo "[INFO] Python       : $PYTHON_BIN"
echo "[INFO] Pip          : $PIP_BIN"
echo "[INFO] Requirements : $REQUIREMENTS_FILE"
echo "[INFO] Train script : $TRAIN_SCRIPT"
echo "[INFO] HPA_DATA_ROOT: $HPA_DATA_ROOT"
echo "[INFO] HPA_TRAIN_DIR: $HPA_TRAIN_DIR"
echo "[INFO] HPA_OUTPUT_DIR: $HPA_OUTPUT_DIR"
echo "[INFO] HF_CACHE_DIR : $HF_CACHE_DIR"
echo "[INFO] VAL_RATIO    : ${HPA_VAL_RATIO:-0.10}"
echo "[INFO] REBUILD_SPLIT: ${HPA_REBUILD_SPLITS:-false}"
echo "[INFO] TRAIN_EPOCHS : ${HPA_TRAIN_EPOCHS:-10.0}"
echo "[INFO] TRAIN_LR     : ${HPA_TRAIN_LR:-1e-5}"
echo "[INFO] TRAIN_BATCH  : ${HPA_TRAIN_BATCH_SIZE:-32}"
echo "[INFO] EVAL_BATCH   : ${HPA_EVAL_BATCH_SIZE:-32}"
echo "============================================================"

if [[ ! -f "$HPA_TRAIN_DIR/metadata.jsonl" ]]; then
  echo "[ERROR] Missing metadata file: $HPA_TRAIN_DIR/metadata.jsonl" >&2
  echo "        Expected layout:" >&2
  echo "          HPA_DATA_ROOT/train/metadata.jsonl" >&2
  echo "          HPA_DATA_ROOT/train/images/..." >&2
  echo "" >&2
  echo "        Current values:" >&2
  echo "          HPA_DATA_ROOT=$HPA_DATA_ROOT" >&2
  echo "          HPA_TRAIN_DIR=$HPA_TRAIN_DIR" >&2
  echo "" >&2
  echo "        Fix by running, for example:" >&2
  echo "          HPA_DATA_ROOT=/home/ubuntu/dataset ./run_train_kansallisarkisto_hpa_single_train_cloud.sh" >&2
  echo "        or:" >&2
  echo "          HPA_TRAIN_DIR=/home/ubuntu/dataset/train ./run_train_kansallisarkisto_hpa_single_train_cloud.sh" >&2
  exit 1
fi

mkdir -p "$HPA_OUTPUT_DIR" "$HF_CACHE_DIR"

echo "[INFO] Starting training..."
"$PYTHON_BIN" "$TRAIN_SCRIPT"
