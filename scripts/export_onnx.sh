#!/usr/bin/env bash
# Export a YOLO26 .pt model to ONNX.
#
# Usage:
#   bash scripts/export_onnx.sh [model] [imgsz]
#
# Defaults:
#   model = best.pt
#   imgsz = 192

set -euo pipefail

MODEL=${1:-best.pt}
IMGSZ=${2:-192}

echo "==> Exporting $MODEL → ONNX (imgsz=$IMGSZ)"
yolo export model="$MODEL" format=onnx imgsz="$IMGSZ"

echo "==> Done: ${MODEL%.pt}.onnx"
