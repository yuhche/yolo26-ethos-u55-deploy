"""
Prepare a calibration dataset (.npy) for INT8 TFLite quantization.

Usage:
    python scripts/prepare_calibration.py \
        --images datasets/ssperson.v7i.yolo26/valid/images \
        --imgsz 192 \
        --n 100 \
        --output calib_192.npy
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory of calibration images")
    parser.add_argument("--imgsz",  type=int, default=192, help="Model input size")
    parser.add_argument("--n",      type=int, default=100, help="Max number of images to use")
    parser.add_argument("--output", default="calib_192.npy")
    args = parser.parse_args()

    img_dir = Path(args.images)
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    paths = paths[: args.n]

    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    imgs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (args.imgsz, args.imgsz))
        imgs.append(img.astype(np.float32) / 255.0)

    cal = np.stack(imgs)
    np.save(args.output, cal)
    print(f"Saved {cal.shape} {cal.dtype} → {args.output}")


if __name__ == "__main__":
    main()
