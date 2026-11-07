# yolo26-ethos-u55-deploy

Deployment scaffold for **Ultralytics YOLO26 → INT8 TFLite → Arm Vela → Ethos-U55** targeting **Seeed Studio Grove Vision AI V2**.

## Status

Implemented project structure, conversion/inspection scripts, Vela compile helper, host benchmark, and firmware-side preprocessing/postprocessing skeletons.

## Repository Layout

```text
models/
  pytorch/     # trained .pt models
  onnx/        # exported ONNX models
  tflite/      # INT8 TFLite models
  vela/        # Vela-compiled TFLite models
notebooks/
  train_yolo26_colab.ipynb  # Colab training notebook
datasets/
  calibration_images/  # representative images for INT8 calibration
scripts/
  export_onnx.py
  convert_tflite_int8.py
  check_tflite_ops.py
  check_sscma_yolo26_tflite.py
  compile_vela.sh
  benchmark_tflite.py
  sscma_probe.py
firmware/
  model/
  src/
docs/
  hardware_notes.md
  benchmark_results.md
PLAN.md
```

## Setup

```bash
uv sync
```

## End-to-End Model Pipeline
### 0. Download person dataset from roboflow

[Download from Roboflow](https://universe.roboflow.com/hanzhou-7mktt/ssperson/dataset/7)

### 1. Train or fine-tune YOLO26

Training runs on Colab (free T4 GPU): open
[`notebooks/train_yolo26_colab.ipynb`](notebooks/train_yolo26_colab.ipynb) in Colab
(File → Upload notebook, or open it from GitHub), add your Roboflow API key as the Colab secret
`ROBOFLOW_API_KEY` (the notebook downloads the dataset from Roboflow), and run all cells.
Copy the resulting `best.pt` to `models/pytorch/best.pt`.

### 2. Export ONNX

```bash
python scripts/export_onnx.py \
  --input models/pytorch/best.pt \
  --output models/onnx/best.onnx \
  --imgsz 320 \
  --opset 13
```

### 3. Convert to fully quantized INT8 TFLite

Place 100–500 representative camera images in `datasets/calibration_images/`, then:

```bash
python scripts/convert_tflite_int8.py \
  --input models/onnx/best.onnx \
  --output models/tflite/best_int8.tflite \
  --calibration-dataset datasets/calibration_images \
  --imgsz 320
```

### 4. Inspect TFLite ops and SSCMA YOLO26 layout

```bash
python scripts/check_tflite_ops.py models/tflite/best_int8.tflite
python scripts/check_sscma_yolo26_tflite.py models/tflite/best_int8.tflite
```

Important: SSCMA-Micro's native YOLO26 postprocess expects six outputs: three box maps and three class maps. See `docs/sscma_integration.md`.

### 5. Compile with Arm Vela

Grove Vision AI V2 uses Ethos-U55 64 MACs, so default accelerator config is `ethos-u55-64`:

```bash
scripts/compile_vela.sh models/tflite/best_int8.tflite models/vela
```

Override if needed:

```bash
ACCELERATOR=ethos-u55-64 OPTIMISE=Performance \
  scripts/compile_vela.sh models/tflite/best_int8.tflite models/vela
```

### 6. Host smoke benchmark

```bash
python scripts/benchmark_tflite.py models/tflite/best_int8.tflite --runs 100
```

Real performance must be measured on Grove Vision AI V2.

## Firmware Integration

Implemented skeleton files:

```text
firmware/src/camera_preprocess.cc  # YUV422 → RGB int8 NHWC
firmware/src/yolo_decode.cc        # raw YOLO output decode
firmware/src/nms.cc                # per-class NMS
firmware/src/inference.cc          # placeholder TFLite Micro/Ethos-U integration
firmware/src/main.cc               # integration skeleton
```

After Vela compilation, generate a C array:

```bash
cp models/vela/*_vela.tflite firmware/model/model_int8_vela.tflite
cd firmware/model
xxd -i model_int8_vela.tflite > model_data.cc
```

Then wire/register the model through SSCMA-Micro where possible instead of custom firmware. Register it as model type `MA_MODEL_TYPE_YOLO26` / ID `14`.

## Host Test with python-sscma

After flashing SSCMA firmware/model to Grove Vision AI V2:

```bash
python scripts/sscma_probe.py --port /dev/ttyACM0 --invoke 1 --tscore 60 --tiou 50
```

Continuous invoke:

```bash
python scripts/sscma_probe.py --port /dev/ttyACM0 --invoke -1 --seconds 60
```

## Notes

See:

- `PLAN.md` for full deployment plan
- `docs/hardware_notes.md` for target assumptions
- `docs/sscma_integration.md` for SSCMA-Micro/python-sscma findings
- `docs/benchmark_results.md` for benchmark tracking
