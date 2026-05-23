#!/usr/bin/env bash
set -euo pipefail

# Create a Python 3.10 environment for UniMERNet training on Linux/Thunder Compute.
#
# Usage:
#   bash setup_unimernet_py310.sh
#
# Optional overrides:
#   PYTHON_BIN=python3.10 VENV_DIR=.venv_unimernet bash setup_unimernet_py310.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash setup_unimernet_py310.sh

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv_unimernet}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python 3.10 was not found as '${PYTHON_BIN}'."
  echo "Ubuntu example:"
  echo "  sudo apt-get update"
  echo "  sudo apt-get install -y python3.10 python3.10-venv python3.10-dev"
  exit 1
fi

PY_VER="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "${PY_VER}" != "3.10" ]]; then
  echo "UniMERNet is expected to run under Python 3.10, but ${PYTHON_BIN} is Python ${PY_VER}."
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install torch first so CUDA wheels are selected explicitly.
python -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision
python -m pip install -r requirements_unimernet_py310.txt

# UniMERNet declares opencv-python as a dependency, which can install the GUI
# OpenCV wheel. Headless GPU servers often do not have libGL.so.1, so force the
# headless wheel after dependency resolution.
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless || true
python -m pip install --no-deps --force-reinstall opencv-python-headless==4.11.0.86

python - <<'PY'
import sys
import torch
import unimernet

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("UniMERNet:", getattr(unimernet, "__version__", "installed"))
PY

echo
echo "Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
