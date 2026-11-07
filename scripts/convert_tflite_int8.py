"""
Phase 5 — Convert ONNX to Fully Quantized INT8 TFLite

Convert an ONNX YOLO26 model to a fully quantized INT8 TFLite model
suitable for compilation with Arm Vela and deployment on Ethos-U55.

Pipeline:
    ONNX → TensorFlow SavedModel → TFLite INT8

Usage:
    python scripts/convert_tflite_int8.py \
        --input models/onnx/best.onnx \
        --output models/tflite/best_int8.tflite \
        --calibration-dataset datasets/calibration_images/
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np


def _load_calibration_images(dataset_dir: str, count: int, imgsz: int) -> list[np.ndarray]:
    """Load and preprocess calibration images as float32 RGB 0..1 tensors."""
    import cv2

    dataset_dir = Path(dataset_dir)
    image_files = sorted(
        list(dataset_dir.glob("*.jpg"))
        + list(dataset_dir.glob("*.jpeg"))
        + list(dataset_dir.glob("*.png"))
    )[:count]

    if not image_files:
        raise FileNotFoundError(
            f"No calibration images found in {dataset_dir}. "
            "Place 100+ representative camera images there."
        )

    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (imgsz, imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        images.append(img[np.newaxis, ...])

    if not images:
        raise RuntimeError(f"Could not load any calibration images from {dataset_dir}")

    print(f"  Loaded {len(images)} calibration images")
    return images


def _representative_dataset_gen(images: list[np.ndarray]):
    """Representative dataset generator. Uses float inputs before quantization."""
    for img in images:
        yield [img]


def _convert_onnx_to_saved_model(input_path: Path, saved_model_dir: Path):
    """Run onnx2tf CLI to produce a TensorFlow SavedModel directory."""
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    cmd = [
        "onnx2tf",
        "-i",
        str(input_path),
        "-o",
        str(saved_model_dir),
    ]

    print("  Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not saved_model_dir.exists():
        raise RuntimeError(f"onnx2tf did not create {saved_model_dir}")


def convert(
    input_path: str,
    output_path: str,
    calibration_dataset: str | None = None,
    calib_count: int = 100,
    imgsz: int = 320,
):
    """Convert ONNX to fully quantized INT8 TFLite."""
    import tensorflow as tf

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_model_dir = output_path.with_suffix(".saved_model")

    print(f"\nStep 1: ONNX → TensorFlow SavedModel")
    _convert_onnx_to_saved_model(input_path, saved_model_dir)
    print(f"  SavedModel written to {saved_model_dir}")

    print(f"\nStep 2: SavedModel → fully quantized INT8 TFLite")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    if calibration_dataset:
        images = _load_calibration_images(calibration_dataset, calib_count, imgsz)
        converter.representative_dataset = lambda: _representative_dataset_gen(images)
    else:
        print("  WARNING: No calibration dataset supplied; using random data.")
        print("  Accuracy will likely be poor. Use real camera-like images for calibration.")

        def random_gen():
            for _ in range(calib_count):
                yield [np.random.random_sample((1, imgsz, imgsz, 3)).astype(np.float32)]

        converter.representative_dataset = random_gen

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    print(f"\nINT8 TFLite model saved to {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024:.1f} KB")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to INT8 TFLite")
    parser.add_argument("--input", required=True, help="Path to .onnx model")
    parser.add_argument("--output", required=True, help="Path for output .tflite")
    parser.add_argument("--calibration-dataset", default=None, help="Directory of calibration images")
    parser.add_argument("--calib-count", type=int, default=100, help="Calibration images to use")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size")
    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_path=args.output,
        calibration_dataset=args.calibration_dataset,
        calib_count=args.calib_count,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
