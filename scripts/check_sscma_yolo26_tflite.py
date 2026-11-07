"""
Validate a TFLite model against SSCMA-Micro's YOLO26 postprocess expectations.

SSCMA-Micro native YOLO26 detector expects one input and six outputs in either:

Interleaved:
  box8, cls8, box16, cls16, box32, cls32
Grouped:
  box8, box16, box32, cls8, cls16, cls32

Each box tensor must be [1, 4, grid, grid]. Class tensors must be
[1, num_classes, grid, grid].

Usage:
    python scripts/check_sscma_yolo26_tflite.py models/tflite/best_int8.tflite
"""

import argparse
from pathlib import Path


def _shape(tensor):
    return [int(x) for x in tensor["shape"]]


def _dtype(tensor):
    return str(tensor["dtype"])


def _is_box(shape, grid):
    return len(shape) == 4 and shape[0] == 1 and shape[1] == 4 and shape[2] == grid and shape[3] == grid


def _is_cls(shape, grid):
    return len(shape) == 4 and shape[0] == 1 and shape[1] > 0 and shape[2] == grid and shape[3] == grid


def validate(model_path: str) -> int:
    import tensorflow as tf

    model_path = Path(model_path)
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    print(f"Model: {model_path}")
    print("Inputs:")
    for i, t in enumerate(inputs):
        print(f"  {i}: name={t['name']} shape={_shape(t)} dtype={_dtype(t)} quant={t['quantization']}")
    print("Outputs:")
    for i, t in enumerate(outputs):
        print(f"  {i}: name={t['name']} shape={_shape(t)} dtype={_dtype(t)} quant={t['quantization']}")

    errors = []

    if len(inputs) != 1:
        errors.append(f"Expected exactly 1 input, got {len(inputs)}")
    else:
        ishape = _shape(inputs[0])
        if len(ishape) != 4 or ishape[0] != 1:
            errors.append(f"Input must be 4D batch-1, got {ishape}")
        else:
            n, d1, d2, d3 = ishape
            if d3 in (1, 3):
                h, w, c = d1, d2, d3
                layout = "NHWC"
            elif d1 in (1, 3):
                c, h, w = d1, d2, d3
                layout = "NCHW"
            else:
                h = w = c = None
                layout = "unknown"
                errors.append(f"Input channel must be 1 or 3 in NHWC/NCHW, got {ishape}")

            if h is not None:
                print(f"Detected input layout: {layout}, H={h}, W={w}, C={c}")
                if h != w:
                    errors.append(f"SSCMA YOLO26 assumes square input, got H={h}, W={w}")
                if h < 32 or h % 32 != 0:
                    errors.append(f"Input height must be >=32 and divisible by 32, got {h}")

    if len(outputs) < 6:
        errors.append(f"Expected at least 6 outputs, got {len(outputs)}")
    else:
        h = _shape(inputs[0])[1] if _shape(inputs[0])[-1] in (1, 3) else _shape(inputs[0])[2]
        grids = [h // 8, h // 16, h // 32]
        shapes = [_shape(t) for t in outputs[:6]]

        interleaved = all(
            _is_box(shapes[i * 2], grids[i]) and _is_cls(shapes[i * 2 + 1], grids[i])
            for i in range(3)
        )
        grouped = all(_is_box(shapes[i], grids[i]) and _is_cls(shapes[i + 3], grids[i]) for i in range(3))

        if interleaved:
            print("SSCMA YOLO26 layout: PASS interleaved")
        elif grouped:
            print("SSCMA YOLO26 layout: PASS grouped")
        else:
            errors.append("Outputs do not match SSCMA YOLO26 interleaved or grouped six-output layout")

        for t in outputs[:6]:
            if "int8" not in _dtype(t) and "float32" not in _dtype(t):
                errors.append(f"Unsupported output dtype for SSCMA YOLO26: {t['name']} {_dtype(t)}")

    print("\nSummary:")
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return 1

    print("  PASS: model shape is compatible with SSCMA-Micro YOLO26 postprocess")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Validate SSCMA YOLO26 TFLite layout")
    parser.add_argument("model")
    args = parser.parse_args()
    raise SystemExit(validate(args.model))


if __name__ == "__main__":
    main()
