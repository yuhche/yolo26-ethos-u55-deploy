"""
Phase 6 — Check TFLite Operator Compatibility

Inspect a TFLite model and report:
- input/output tensor quantization
- operator list
- likely Ethos-U55/TFLite Micro compatibility issues

Usage:
    python scripts/check_tflite_ops.py models/tflite/best_int8.tflite
"""

import argparse
from pathlib import Path


ETHOS_U_FRIENDLY_OPS = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "FULLY_CONNECTED",
    "AVERAGE_POOL_2D",
    "MAX_POOL_2D",
    "ADD",
    "MUL",
    "SUB",
    "RESHAPE",
    "PAD",
    "CONCATENATION",
    "LOGISTIC",
    "TANH",
    "RELU",
    "RELU6",
    "LEAKY_RELU",
    "QUANTIZE",
    "DEQUANTIZE",
}

PROBLEMATIC_FOR_MCU = {
    "NON_MAX_SUPPRESSION_V4",
    "NON_MAX_SUPPRESSION_V5",
    "COMBINED_NON_MAX_SUPPRESSION",
    "CUSTOM",
    "SELECT_TF_OPS",
    "WHILE",
    "IF",
    "TOPK_V2",
    "EXP",
    "SOFTMAX",
    "RESIZE_BILINEAR",
    "RESIZE_NEAREST_NEIGHBOR",
}


def inspect_model(model_path: str) -> int:
    import tensorflow as tf

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    print(f"Model: {model_path}")
    print(f"Size:  {model_path.stat().st_size / 1024:.1f} KB\n")

    print("Inputs:")
    for t in interpreter.get_input_details():
        print(f"  {t['name']}: shape={t['shape']} dtype={t['dtype']} quant={t['quantization']}")

    print("\nOutputs:")
    for t in interpreter.get_output_details():
        print(f"  {t['name']}: shape={t['shape']} dtype={t['dtype']} quant={t['quantization']}")

    ops = interpreter._get_ops_details()  # pylint: disable=protected-access
    counts = {}
    for op in ops:
        name = op["op_name"]
        counts[name] = counts.get(name, 0) + 1

    print("\nOperators:")
    for name, count in sorted(counts.items()):
        marker = "OK"
        if name in PROBLEMATIC_FOR_MCU:
            marker = "PROBLEM"
        elif name not in ETHOS_U_FRIENDLY_OPS:
            marker = "CHECK"
        print(f"  {marker:7} {name:32} x{count}")

    issues = []

    for t in interpreter.get_input_details() + interpreter.get_output_details():
        dtype = str(t["dtype"])
        if "int8" not in dtype and "uint8" not in dtype:
            issues.append(f"Tensor {t['name']} is not int8/uint8: {dtype}")

    for name in counts:
        if name in PROBLEMATIC_FOR_MCU:
            issues.append(f"Problematic op found: {name}")
        elif name not in ETHOS_U_FRIENDLY_OPS:
            issues.append(f"Needs compatibility check: {name}")

    print("\nCompatibility summary:")
    if not issues:
        print("  PASS: No obvious TFLite Micro / Ethos-U55 issues found.")
        return 0

    for issue in issues:
        print(f"  WARN: {issue}")
    return 1


def main():
    parser = argparse.ArgumentParser(description="Inspect TFLite ops")
    parser.add_argument("model", help="Path to .tflite model")
    args = parser.parse_args()
    raise SystemExit(inspect_model(args.model))


if __name__ == "__main__":
    main()
