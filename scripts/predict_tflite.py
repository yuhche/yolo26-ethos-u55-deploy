"""
TFLite inference script for YOLO26 full-integer-quantized model.

Usage:
    python scripts/predict_tflite.py --source image.jpg
    python scripts/predict_tflite.py --source images/ --save
    python scripts/predict_tflite.py --source image.jpg --conf 0.3 --show
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def load_interpreter(model_path: str):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0]


def preprocess(img_bgr: np.ndarray, in_detail: dict) -> np.ndarray:
    _, h, w, _ = in_detail["shape"]
    scale, zero = in_detail["quantization"]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    img_f = img.astype(np.float32) / 255.0
    img_q = np.clip(np.round(img_f / scale + zero), -128, 127).astype(np.int8)
    return img_q[None]


def postprocess(raw: np.ndarray, in_detail: dict, out_detail: dict,
                orig_h: int, orig_w: int, conf_thres: float, names: list):
    scale, zero = out_detail["quantization"]
    boxes = (raw[0].astype(np.float32) - zero) * scale  # (300, 6)

    sx, sy = orig_w, orig_h
    detections = []
    for x1, y1, x2, y2, conf, cls in boxes:
        if conf < conf_thres:
            continue
        detections.append({
            "box":   (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)),
            "conf":  float(conf),
            "class": int(cls),
            "label": names[int(cls)] if int(cls) < len(names) else str(int(cls)),
        })
    return detections


def draw(img: np.ndarray, detections: list) -> np.ndarray:
    out = img.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"{d['label']} {d['conf']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def run_image(interp, in_detail, out_detail, img_path: Path,
              conf: float, names: list, save: bool, show: bool, out_dir: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Cannot read {img_path}")
        return

    h, w = img.shape[:2]
    tensor = preprocess(img, in_detail)
    interp.set_tensor(in_detail["index"], tensor)
    interp.invoke()
    raw = interp.get_tensor(out_detail["index"])

    dets = postprocess(raw, in_detail, out_detail, h, w, conf, names)
    print(f"{img_path.name}: {len(dets)} detection(s)")
    for d in dets:
        print(f"  {d['label']}  conf={d['conf']:.3f}  box={d['box']}")

    annotated = draw(img, dets)
    if show:
        cv2.imshow(img_path.name, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)
        print(f"  Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="saved_model/best_full_integer_quant.tflite")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--names",  nargs="+", default=["person"])
    parser.add_argument("--save",   action="store_true")
    parser.add_argument("--show",   action="store_true")
    parser.add_argument("--output", default="runs/tflite/predict")
    args = parser.parse_args()

    interp, in_detail, out_detail = load_interpreter(args.model)
    out_dir = Path(args.output)
    source = Path(args.source)

    if source.is_dir():
        images = sorted(source.glob("*"))
        images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    else:
        images = [source]

    for img_path in images:
        run_image(interp, in_detail, out_detail, img_path,
                  args.conf, args.names, args.save, args.show, out_dir)


if __name__ == "__main__":
    main()
