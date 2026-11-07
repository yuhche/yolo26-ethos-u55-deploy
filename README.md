# yolo26-ethos-u55-deploy

Deploy YOLO26 on ARM Ethos-U55 NPU (Grove Vision AI V2).

## Pipeline

```
Train (.pt) → Export ONNX → Convert TFLite INT8 → Compile with Vela → Deploy
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 1. Train (Colab)

Training runs on Google Colab (free T4 GPU) via
[`notebooks/train_yolo26_colab.ipynb`](notebooks/train_yolo26_colab.ipynb):

1. Open the notebook in Colab (File → Upload notebook, or open from GitHub) and select a GPU runtime.
2. Add your Roboflow API key as the Colab secret `ROBOFLOW_API_KEY`
   (the notebook downloads the `ssperson` v7 dataset from Roboflow).
3. Run all cells. `best.pt` is saved to Google Drive and downloaded.
4. Place it at `models/pytorch/best.pt` (or pass its path to the steps below).

To use the dataset locally for calibration/prediction, download it from Roboflow into
`datasets/ssperson.v7i.yolo26/`.

## 2. Predict (PyTorch)

```bash
bash scripts/predict_yolo26.sh
```

Or directly:

```bash
yolo predict model=best.pt source=image.jpg imgsz=192 save=True
```

## 3. Export

### 3a. Prepare calibration data (once per dataset)

Required for accurate INT8 quantization. Uses real validation images instead of
generic noise.

```bash
python scripts/prepare_calibration.py \
    --images datasets/ssperson.v7i.yolo26/valid/images \
    --imgsz 192 \
    --n 100 \
    --output calib_192.npy
```

### 3b. Export to ONNX

```bash
bash scripts/export_onnx.sh best.pt 192
# output: best.onnx
```

### 3c. Convert to INT8 TFLite

```bash
bash scripts/export_tflite.sh best.onnx calib_192.npy saved_model
# output: saved_model/best_full_integer_quant.tflite
```

> **Note:** `yolo export format=tflite` is not used because it does not support
> custom calibration data, which causes incorrect INT8 quantization.

## 4. Predict (TFLite)

```bash
python scripts/predict_tflite.py \
    --model saved_model/best_full_integer_quant.tflite \
    --source image.jpg \
    --save
```

Options:

| Arg | Default | Description |
|---|---|---|
| `--model` | `saved_model/best_full_integer_quant.tflite` | TFLite model |
| `--source` | — | Image file or directory |
| `--conf` | `0.25` | Confidence threshold |
| `--names` | `person` | Class names |
| `--save` | off | Save annotated output |
| `--show` | off | Display result window |
| `--output` | `runs/tflite/predict` | Output directory |

## 5. Compile with Vela (Ethos-U55)

```bash
bash scripts/compile_vela.sh saved_model/best_full_integer_quant.tflite output_vela
# output: output_vela/best_full_integer_quant_vela.tflite
```

Key flags used internally:

| Flag | Value | Reason |
|---|---|---|
| `--accelerator-config` | `ethos-u55-128` | Grove Vision AI V2 has U55-128 (128 MACs) |
| `--system-config` | `Ethos_U55_High_End_Embedded` | WE2 timing model (500 MHz, SRAM AXI0) |
| `--memory-mode` | `Shared_Sram` | Weights in flash (AXI1), activations in SRAM (AXI0) |

Expected output for this model (192×192, 1 class):

```
NPU operators = 972  (97.1%)   backbone + neck + head convolutions
CPU operators =  29  ( 2.9%)   baked-in NMS (TopK, GatherND, Cast, FloorMod)

Total SRAM used        321 KiB
Total Off-chip Flash  2196 KiB
MACs                   236 M MACs/frame
```

The 2.9% CPU fallback is the end-to-end NMS head — unavoidable with the default
`end2end=True` export. Flash `output_vela/best_full_integer_quant_vela.tflite`
to the device.

## Scripts

Training lives in the Colab notebook above; the scripts cover everything after it.

| Script | Description |
|---|---|
| `scripts/predict_yolo26.sh` | Quick PyTorch inference |
| `scripts/prepare_calibration.py` | Build INT8 calibration dataset |
| `scripts/export_onnx.sh` | Export `.pt` → ONNX |
| `scripts/export_tflite.sh` | Convert ONNX → INT8 TFLite |
| `scripts/compile_vela.sh` | Compile INT8 TFLite with Vela for Ethos-U55 |
| `scripts/predict_tflite.py` | TFLite inference with bounding box output |
