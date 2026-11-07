"""
Phase 4 — Export YOLO26 to ONNX

Export a trained YOLO26 PyTorch model to ONNX format for downstream
TFLite conversion. Static input shapes are used for Ethos-U55 compatibility.

Usage:
    python scripts/export_onnx.py \
        --input models/pytorch/best.pt \
        --output models/onnx/best.onnx \
        --imgsz 320 \
        --opset 13
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def export(
    input_path: str,
    output_path: str,
    imgsz: int = 320,
    opset: int = 13,
    simplify: bool = True,
):
    """Export YOLO26 .pt model to static-shape ONNX."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {input_path} ...")
    model = YOLO(str(input_path))

    print(f"Exporting ONNX: imgsz={imgsz}, opset={opset}, dynamic=False")
    onnx_path = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=False,
            half=False,
            nms=False,  # keep NMS outside graph for MCU firmware
        )
    )

    if onnx_path.resolve() != output_path.resolve():
        shutil.move(str(onnx_path), str(output_path))

    print(f"ONNX model saved to {output_path}")
    return str(output_path)


def validate_onnx(onnx_path: str):
    """Basic ONNX structural validation and dummy inference smoke test."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    print("\nValidating ONNX graph ...")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    dummy = np.zeros(shape, dtype=np.float32)
    outputs = sess.run(None, {inp.name: dummy})

    print(f"  Input:  {inp.name} shape={shape}")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: shape={out.shape} dtype={out.dtype}")
    print("  ONNX validation PASSED")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO26 to ONNX")
    parser.add_argument("--input", required=True, help="Path to .pt model")
    parser.add_argument("--output", required=True, help="Path for output .onnx")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX simplification")
    parser.add_argument("--no-validate", action="store_true", help="Skip ONNX validation")
    args = parser.parse_args()

    onnx_path = export(
        input_path=args.input,
        output_path=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
    )
    if not args.no_validate:
        validate_onnx(onnx_path)


if __name__ == "__main__":
    main()
