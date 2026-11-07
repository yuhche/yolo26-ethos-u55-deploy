# YOLO26 to Seeed Studio Grove Vision AI V2 / Ethos-U55 Deployment Plan

## Goal

Deploy a quantized Ultralytics YOLO26 object detection model on Seeed Studio Grove Vision AI V2 using the Arm Ethos-U55 NPU.

```text
Ultralytics YOLO26
        ↓
PyTorch .pt
        ↓
ONNX
        ↓
TensorFlow / TFLite INT8
        ↓
Arm Vela compile
        ↓
Ethos-U55 deployable model
        ↓
Grove Vision V2 firmware/app integration
```

---

## Phase 1 — Confirm Target Hardware and Runtime

### Tasks

1. Confirm exact Grove Vision V2 chip/runtime:
   - Ethos-U55 NPU present
   - Supported firmware SDK
   - Supported model format, usually `.tflite` compiled with Vela
2. Identify camera input format:
   - RGB / BGR / YUV
   - Resolution
   - Preprocessing available on-device
3. Confirm memory limits:
   - SRAM
   - Flash
   - External PSRAM, if available
4. Confirm supported inference runtime:
   - TensorFlow Lite Micro
   - Ethos-U delegate
   - Seeed deployment tooling

### Output

A hardware/runtime compatibility note.

---

## Phase 2 — Choose YOLO26 Model Variant

For microcontroller + Ethos-U55 deployment, start with the smallest available model.

Recommended starting point:

```text
YOLO26 nano/smallest available variant
Input size: 160x160, 192x192, 224x224, or 320x320 maximum
Classes: keep as few as possible
Batch size: 1
```

### Important Constraints

Ethos-U55 has limited operator support. Avoid or move to CPU:

- Dynamic shapes
- Complex postprocessing
- NMS inside the model
- Unsupported activations
- Large feature maps
- Large detection heads

Recommended model export style:

```text
Model outputs raw boxes/classes
NMS runs on CPU firmware side
```

---

## Phase 3 — Train or Fine-Tune YOLO26

### Dataset Layout

Use Ultralytics dataset format:

```text
datasets/
  my_dataset/
    images/
      train/
      val/
    labels/
      train/
      val/
    data.yaml
```

### Train

```bash
yolo detect train model=yolo26n.pt data=data.yaml imgsz=320 epochs=100
```

### Validate

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=320
```

### Output

```text
best.pt
metrics.json
example validation images
```

---

## Phase 4 — Export to ONNX

### Recommended Export

```bash
yolo export model=best.pt format=onnx imgsz=320 opset=13 simplify=True dynamic=False
```

Expected output:

```text
best.onnx
```

### Validation

Run ONNX inference and compare with PyTorch output.

---

## Phase 5 — Convert ONNX to TFLite INT8

Ethos-U55 needs a fully quantized INT8 TFLite model.

### Tasks

1. Convert ONNX to TensorFlow SavedModel.
2. Convert SavedModel to TFLite INT8.
3. Provide a representative calibration dataset.
4. Ensure input/output tensors are quantized.

Target properties:

```text
Input: int8 or uint8
Weights: int8
Activations: int8
No float fallback if possible
```

### Representative Dataset

Use 100–500 real camera-like images for calibration.

---

## Phase 6 — Check TFLite Operator Compatibility

Run TFLite model inspection:

```bash
python scripts/check_tflite_ops.py model_int8.tflite
```

Look for unsupported operators.

Common issues:

| Issue | Fix |
|---|---|
| Unsupported op | Modify export/model graph |
| Float tensors remain | Improve quantization path |
| NMS inside graph | Remove NMS and run it in C/C++ |
| Resize/reshape problems | Use static input shapes |

---

## Phase 7 — Compile with Arm Vela

Install Arm Vela:

```bash
pip install ethos-u-vela
```

Compile:

```bash
vela model_int8.tflite \
  --accelerator-config ethos-u55-64 \
  --optimise Performance
```

Possible accelerator configs:

```text
ethos-u55-32
ethos-u55-64
ethos-u55-128
ethos-u55-256
```

Use the one matching Grove Vision V2.

Expected output:

```text
model_int8_vela.tflite
```

This is the NPU-optimized deployable model.

---

## Phase 8 — Validate Vela Output

Check the Vela report for:

```text
NPU delegated operators
CPU fallback operators
SRAM usage
Flash usage
Estimated cycles
```

Acceptance criteria:

```text
Most conv/depthwise ops run on NPU
Postprocessing can run on CPU
Model fits SRAM/flash
Latency is acceptable
```

If the model is too large:

- Reduce input size
- Use smaller YOLO26 variant
- Reduce classes
- Prune channels
- Simplify detection head

---

## Phase 9 — Implement Preprocessing

On Grove Vision V2 firmware:

1. Capture camera frame.
2. Resize to model size.
3. Convert color format.
4. Normalize/quantize to model input type.

Example logic:

```text
camera frame
  ↓
resize to 320x320
  ↓
RGB
  ↓
uint8/int8 quantized tensor
  ↓
TFLite Micro input tensor
```

Important: use the same preprocessing as training/export.

---

## Phase 10 — Implement YOLO Postprocessing on CPU

Recommended pipeline:

```text
raw model output
  ↓
dequantize
  ↓
decode boxes
  ↓
apply confidence threshold
  ↓
apply class scores
  ↓
NMS
  ↓
final detections
```

Keep this outside the model if possible.

Example output type:

```c
typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
} detection_t;
```

---

## Phase 11 — Firmware Integration

Preferred route: integrate through Seeed `SSCMA-Micro`, which already includes native YOLO26 support:

```text
MA_MODEL_TYPE_YOLO26 = 14
MA_MODEL_TYPE_YOLO26_POSE = 15
MA_MODEL_TYPE_YOLO26_SEG = 16
```

Register the Vela-compiled TFLite model as an SSCMA model entry with type `MA_MODEL_TYPE_YOLO26`. Then use the built-in SSCMA YOLO26 postprocess, NMS, AT protocol, and `python-sscma` host client.

The TFLite model must expose the SSCMA YOLO26 six-output layout:

```text
Interleaved: box8, cls8, box16, cls16, box32, cls32
Grouped:     box8, box16, box32, cls8, cls16, cls32
```

Validate this with:

```bash
python scripts/check_sscma_yolo26_tflite.py models/tflite/best_int8.tflite
```

Fallback/custom firmware modules remain useful for reference:

```text
firmware/
  model/
    model_int8_vela.tflite
    model_data.cc
  src/
    camera_preprocess.cc
    inference.cc
    yolo_decode.cc
    nms.cc
    main.cc
```

---

## Phase 12 — Benchmark on Device

Measure:

```text
Preprocess time
NPU inference time
Postprocess time
Total FPS
RAM usage
Flash usage
Detection accuracy
Thermal stability
```

Benchmark table:

| Model | Input | Size | NPU % | Latency | FPS | mAP |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26-n | 320 | TBD | TBD | TBD | TBD | TBD |

---

## Phase 13 — Optimization Loop

If performance is poor:

1. Lower input size: `320 → 224 → 192 → 160`
2. Reduce class count
3. Export without embedded NMS
4. Ensure full INT8 quantization
5. Try another Vela accelerator config
6. Simplify the YOLO head
7. Replace unsupported operators
8. Use per-channel quantization where possible

---

## Recommended Repository Structure

```text
yolo26-ethos-u55-deploy/
  README.md
  PLAN.md
  models/
    pytorch/
    onnx/
    tflite/
    vela/
  datasets/
  scripts/
    export_onnx.py
    convert_tflite_int8.py
    check_tflite_ops.py
    benchmark_tflite.py
  firmware/
    model/
    src/
  docs/
    hardware_notes.md
    benchmark_results.md
```

---

## Main Risks

| Risk | Mitigation |
|---|---|
| YOLO26 ops unsupported by TFLite Micro/Ethos-U | Export raw outputs and simplify graph |
| Model too large | Use nano model and lower input size |
| Poor INT8 accuracy | Use better calibration dataset |
| NMS too expensive | Limit candidates before NMS |
| Memory overflow | Reduce resolution/classes/channels |
| Camera preprocessing mismatch | Match training preprocessing exactly |

---

## First Implementation Milestone

The first useful milestone is:

```text
Train/fine-tune YOLO26 small model
Export to ONNX
Convert to fully INT8 TFLite
Compile with Vela
Generate Vela report
```

After that, proceed with Grove Vision V2 firmware integration.
