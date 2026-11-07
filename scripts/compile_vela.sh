#!/usr/bin/env bash
# Phase 7 — Compile TFLite model with Arm Vela for Ethos-U55

set -euo pipefail

INPUT=${1:-models/tflite/best_int8.tflite}
OUTPUT_DIR=${2:-models/vela}
ACCELERATOR=${ACCELERATOR:-ethos-u55-64}
OPTIMISE=${OPTIMISE:-Performance}

mkdir -p "$OUTPUT_DIR"

if ! command -v vela >/dev/null 2>&1; then
  echo "ERROR: Arm Vela not found. Install with: pip install ethos-u-vela" >&2
  exit 1
fi

vela "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --accelerator-config "$ACCELERATOR" \
  --optimise "$OPTIMISE" \
  --verbose-all

echo "\nVela output written to $OUTPUT_DIR"
