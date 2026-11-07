#!/usr/bin/env bash
# Convert an ONNX model to INT8 TFLite using onnx2tf with custom calibration data.
#
# Usage:
#   bash scripts/export_tflite.sh [onnx] [calib] [output_dir]
#
# Defaults:
#   onnx       = best.onnx
#   calib      = calib_192.npy
#   output_dir = saved_model

set -euo pipefail

ONNX=${1:-best.onnx}
CALIB=${2:-calib_192.npy}
OUTPUT=${3:-saved_model}

if [[ ! -f "$ONNX" ]]; then
  echo "Error: ONNX file '$ONNX' not found."
  echo "Run: bash scripts/export_onnx.sh"
  exit 1
fi

if [[ ! -f "$CALIB" ]]; then
  echo "Error: calibration file '$CALIB' not found."
  echo "Run: python scripts/prepare_calibration.py --images <val_images_dir> --output $CALIB"
  exit 1
fi

echo "==> Converting $ONNX → TFLite (INT8, output: $OUTPUT)"
onnx2tf -i "$ONNX" -o "$OUTPUT" -oiqt \
  -cind images "$CALIB" "[[0,0,0]]" "[[1,1,1]]"

echo "==> Done. TFLite models saved to $OUTPUT/"
