#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-/data/input}"
OUTPUT_DIR="${2:-/data/output}"

exec python /app/inference.py "$INPUT_DIR" "$OUTPUT_DIR"
