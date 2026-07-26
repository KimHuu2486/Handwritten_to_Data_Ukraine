#!/usr/bin/env bash
set -euo pipefail

# Single-phase RUKOPYS HPA branch training on Thunder Compute.
# Branch: handwritten_printed
#
# Default cropped dataset layout:
#   /home/ubuntu/dataset/train/manifest.csv
#   /home/ubuntu/dataset/train/handwritten/*.jpg
#   /home/ubuntu/dataset/train/printed/*.jpg
#
# Metadata layout is still accepted as fallback:
#   /home/ubuntu/dataset/train/metadata.jsonl
#   /home/ubuntu/dataset/train/images/...
#
# Usage:
#   chmod +x run_train_hpa_handwritten_printed_single_phase.sh
#   HF_TOKEN=hf_xxx ./run_train_hpa_handwritten_printed_single_phase.sh
#
# Or with env file:
#   ENV_FILE=/home/ubuntu/.env.hpa ./run_train_hpa_handwritten_printed_single_phase.sh
#
# Example .env.hpa:
#   HF_TOKEN=hf_xxx
#   HPA_DATA_ROOT=/home/ubuntu/dataset
#   HPA_TRAIN_DIR=/home/ubuntu/dataset/train
#   HF_CACHE_DIR=/home/ubuntu/hf_cache
#
# Useful overrides:
#   HPA_TRAIN_EPOCHS=10.0
#   HPA_TRAIN_LR=1e-5
#   HPA_TRAIN_BATCH_SIZE=32
#   HPA_EVAL_BATCH_SIZE=16
#   HPA_VAL_RATIO=0.10
#   HPA_REBUILD_SPLITS=true
#   HPA_SKIP_GENERATION_EVAL=false
#   HPA_STRIP_TEXT_MARKERS_FOR_TRAIN=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

SYSTEM_PYTHON_BIN="${SYSTEM_PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv_hpa_handwritten_printed}"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

if ! command -v "$SYSTEM_PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] System Python executable not found: $SYSTEM_PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[INFO] Creating virtual environment: $VENV_DIR"
  "$SYSTEM_PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[INFO] Reusing virtual environment: $VENV_DIR"
fi

if [[ ! -x "$PYTHON_BIN" || ! -x "$PIP_BIN" ]]; then
  echo "[ERROR] Invalid venv. Missing python/pip in: $VENV_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

REQ_STAMP="$VENV_DIR/.requirements_hpa_hp_installed"
if [[ ! -f "$REQ_STAMP" || "${FORCE_REINSTALL_REQUIREMENTS:-0}" == "1" ]]; then
  echo "[INFO] Installing Python dependencies..."
  if ! "$PYTHON_BIN" - <<'TORCHCHECK' >/dev/null 2>&1
import torch
print(torch.__version__)
TORCHCHECK
  then
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
    echo "[INFO] torch not found. Installing torch from: $TORCH_INDEX_URL"
    "$PIP_BIN" install torch torchvision --index-url "$TORCH_INDEX_URL"
  else
    echo "[INFO] Existing torch detected. Keeping current torch install."
  fi

  "$PIP_BIN" install \
    "transformers>=4.42,<5" \
    "accelerate>=0.31" \
    "pillow>=10" \
    "jiwer>=3.0" \
    "tqdm>=4.66" \
    "numpy<2" \
    "python-dotenv>=1.0" \
    "safetensors>=0.4" \
    "sentencepiece>=0.2"
  date > "$REQ_STAMP"
else
  echo "[INFO] Requirements already installed. Set FORCE_REINSTALL_REQUIREMENTS=1 to reinstall."
fi

# HF_KEY is supported as a convenience alias, but HF_TOKEN is preferred.
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" && -n "${HF_KEY:-}" ]]; then
  export HF_TOKEN="$HF_KEY"
  echo "[INFO] Mapped HF_KEY -> HF_TOKEN."
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "[WARN] HF_TOKEN/HUGGINGFACE_HUB_TOKEN is not set. Public models may still download."
fi

export HPA_DATA_ROOT="${HPA_DATA_ROOT:-${DATA_ROOT:-/home/ubuntu/dataset}}"
export HPA_TRAIN_DIR="${HPA_TRAIN_DIR:-$HPA_DATA_ROOT/train}"
export HPA_OUTPUT_DIR="${HPA_OUTPUT_DIR:-/home/ubuntu/outputs_hpa_handwritten_printed_single}"
export HPA_FINAL_DIR="${HPA_FINAL_DIR:-$HPA_OUTPUT_DIR/final_cyrillic_handwritten_printed_model}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-/home/ubuntu/hf_cache}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

# Branch defaults. You can override any of these from shell or ENV_FILE.
export HPA_TRAIN_EPOCHS="${HPA_TRAIN_EPOCHS:-10.0}"
export HPA_TRAIN_LR="${HPA_TRAIN_LR:-1e-5}"
export HPA_TRAIN_BATCH_SIZE="${HPA_TRAIN_BATCH_SIZE:-32}"
export HPA_EVAL_BATCH_SIZE="${HPA_EVAL_BATCH_SIZE:-16}"
export HPA_VAL_RATIO="${HPA_VAL_RATIO:-0.10}"
export HPA_REBUILD_SPLITS="${HPA_REBUILD_SPLITS:-true}"
export HPA_OPTIM="${HPA_OPTIM:-adamw_torch}"
export HPA_LR_SCHEDULER_TYPE="${HPA_LR_SCHEDULER_TYPE:-cosine}"
export HPA_WARMUP_RATIO="${HPA_WARMUP_RATIO:-0.05}"
export HPA_STRIP_TEXT_MARKERS_FOR_TRAIN="${HPA_STRIP_TEXT_MARKERS_FOR_TRAIN:-${HPA_STRIP_STRIKETHROUGH_MARKERS:-false}}"
export HPA_ADD_SPECIAL_MARKER_TOKENS="${HPA_ADD_SPECIAL_MARKER_TOKENS:-${HPA_ADD_STRIKETHROUGH_TOKENS:-false}}"
export HPA_ADD_UKRAINIAN_TOKENS="${HPA_ADD_UKRAINIAN_TOKENS:-false}"
export HPA_MAX_TARGET_LENGTH="${HPA_MAX_TARGET_LENGTH:-192}"
export HPA_GEN_EVAL_BATCH_SIZE="${HPA_GEN_EVAL_BATCH_SIZE:-${HPA_GENERATION_EVAL_BATCH_SIZE:-4}}"

TRAIN_SCRIPT="$SCRIPT_DIR/train_hpa_handwritten_printed_single_phase.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "[ERROR] Missing training script: $TRAIN_SCRIPT" >&2
  exit 1
fi

DEFAULT_CROP_MANIFEST="$HPA_TRAIN_DIR/manifest.csv"
LEGACY_CROP_MANIFEST="$HPA_TRAIN_DIR/cropped_bboxes/manifest.csv"
ACTIVE_CROP_MANIFEST="${HPA_CROP_MANIFEST:-}"

if [[ -n "$ACTIVE_CROP_MANIFEST" && ! -f "$ACTIVE_CROP_MANIFEST" ]]; then
  echo "[ERROR] HPA_CROP_MANIFEST does not exist: $ACTIVE_CROP_MANIFEST" >&2
  exit 1
fi

if [[ -z "$ACTIVE_CROP_MANIFEST" && -f "$DEFAULT_CROP_MANIFEST" ]]; then
  ACTIVE_CROP_MANIFEST="$DEFAULT_CROP_MANIFEST"
elif [[ -z "$ACTIVE_CROP_MANIFEST" && -f "$LEGACY_CROP_MANIFEST" ]]; then
  ACTIVE_CROP_MANIFEST="$LEGACY_CROP_MANIFEST"
fi

if [[ -n "$ACTIVE_CROP_MANIFEST" ]]; then
  export HPA_CROP_MANIFEST="$ACTIVE_CROP_MANIFEST"
  export HPA_USE_CROPPED_MANIFEST="${HPA_USE_CROPPED_MANIFEST:-true}"
elif [[ ! -f "$HPA_TRAIN_DIR/metadata.jsonl" ]]; then
  echo "[ERROR] Missing dataset labels." >&2
  echo "Expected cropped manifest: $DEFAULT_CROP_MANIFEST" >&2
  echo "Fallback metadata path : $HPA_TRAIN_DIR/metadata.jsonl" >&2
  exit 1
fi

mkdir -p "$HPA_OUTPUT_DIR" "$HPA_FINAL_DIR" "$HF_CACHE_DIR"

echo "============================================================"
echo "[INFO] Branch        : handwritten_printed"
echo "[INFO] Train script  : $TRAIN_SCRIPT"
echo "[INFO] HPA_TRAIN_DIR : $HPA_TRAIN_DIR"
echo "[INFO] Crop manifest : ${HPA_CROP_MANIFEST:-disabled}"
echo "[INFO] HPA_OUTPUT_DIR: $HPA_OUTPUT_DIR"
echo "[INFO] HPA_FINAL_DIR : $HPA_FINAL_DIR"
echo "[INFO] HF_CACHE_DIR  : $HF_CACHE_DIR"
echo "[INFO] Optim/LR      : $HPA_OPTIM / $HPA_TRAIN_LR / $HPA_LR_SCHEDULER_TYPE"
echo "[INFO] Epochs        : $HPA_TRAIN_EPOCHS"
echo "[INFO] Batch         : train=$HPA_TRAIN_BATCH_SIZE eval=$HPA_EVAL_BATCH_SIZE"
echo "[INFO] Preserve ~~   : $([[ "$HPA_STRIP_TEXT_MARKERS_FOR_TRAIN" == "true" ]] && echo no || echo yes)"
echo "============================================================"

"$PYTHON_BIN" "$TRAIN_SCRIPT"
