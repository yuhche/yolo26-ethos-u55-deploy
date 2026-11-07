#!/usr/bin/env bash
# Train YOLO26 using the yolo CLI directly.
#
# Usage:
#   bash scripts/train_yolo26.sh --data datasets/xxx.yaml [options]
#
# Options:
#   --model   Model to fine-tune (default: yolo26n.pt)
#   --data    Dataset yaml (required)
#   --imgsz   Image size (default: 320)
#   --epochs  Epochs (default: 100)
#   --batch   Batch size (default: auto)
#   --output  Output path for best.pt (default: models/pytorch/best.pt)

set -euo pipefail

yolo detect train model=yolo26n.pt "$@"
