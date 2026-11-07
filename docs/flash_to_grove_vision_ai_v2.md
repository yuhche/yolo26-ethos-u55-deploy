# Flashing a Vela Model to Grove Vision AI V2

## Overview

The Grove Vision AI V2 (Himax WiseEye2 / HX6538) requires two things to be flashed together:

1. **Firmware image** (`output.img`) — SSCMA-Micro runtime (boots the device, loads the model, runs inference)
2. **Vela TFLite model** (`*_vela.tflite`) — your compiled model, placed at a fixed flash address

Both are sent in a single `xmodem_send.py` command over serial (921600 baud).

---

## Prerequisites

### Hardware
- Grove Vision AI V2 board
- USB-C cable connected to host PC

### Software

Clone the sscma-example-we2 repo — it contains the flash tool, pre-built firmware, and example models:

```bash
git clone https://github.com/Seeed-Studio/sscma-example-we2
cd sscma-example-we2
pip install -r xmodem/requirements.txt
```

### Serial port access (Linux only)

```bash
sudo setfacl -m u:$USER:rw /dev/ttyACM0
```

Find your port with `ls /dev/ttyACM*` or `ls /dev/ttyUSB*`.

---

## Flash command

### Linux

```bash
python3 xmodem/xmodem_send.py \
  --port=/dev/ttyACM0 \
  --baudrate=921600 \
  --protocol=xmodem \
  --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img \
  --model="/path/to/output_vela/best_full_integer_quant_vela.tflite 0xB7B000 0x00000"
```

### Windows

```cmd
python xmodem\xmodem_send.py ^
  --port=COM3 ^
  --baudrate=921600 ^
  --protocol=xmodem ^
  --file=we2_image_gen_local\output_case1_sec_wlcsp\output.img ^
  --model="path\to\output_vela\best_full_integer_quant_vela.tflite 0xB7B000 0x00000"
```

Then **press the reset button** on the board.

### Parameter reference

| Parameter | Value | Notes |
|---|---|---|
| `--port` | `/dev/ttyACM0` or `COMx` | Your serial port |
| `--baudrate` | `921600` | Fixed, must match |
| `--file` | `output.img` | SSCMA-Micro firmware, max 1 MB |
| `--model` | `<tflite> <addr> <offset>` | Space-separated triple |
| flash address | `0xB7B000` | Where SSCMA-Micro looks for the model |
| offset | `0x00000` | Start of the model file |

---

## What gets flashed where

```
Flash layout (WE2 / HX6538)
─────────────────────────────────────────────────
0x00000000  Bootloader + SSCMA-Micro firmware
            (from output.img, max ~1 MB)

0x00B7B000  TFLite model
            (your *_vela.tflite)
─────────────────────────────────────────────────
```

SSCMA-Micro reads the model from `0xB7B000` at boot, runs `isValid()` against each known model type, and starts inference on the camera stream once a match is found.

---

## YOLO26-specific notes

### Model format requirement

SSCMA-Micro's `ma_model_yolo26.cpp` expects **6 raw output tensors** (pre-NMS multi-scale), not the default end2end NMS output. You must export with `end2end=False`:

```bash
yolo export model=best.pt format=onnx imgsz=192 end2end=False
```

Then re-run `export_tflite.sh` and `compile_vela.sh` on that ONNX before flashing.

### No pre-built scenario app

`sscma-example-we2` has `tflm_yolov8_od` and `tflm_yolo11_od` but **no `tflm_yolo26_od`** yet. Two options:

**Option A — Use the `sscma` scenario firmware**

The generic `sscma` scenario compiles in the full SSCMA-Micro library, which includes `ma_model_yolo26.cpp`. Model type is auto-detected at runtime. Build the `sscma` scenario and use its `output.img`.

**Option B — Clone `tflm_yolo11_od`**

```bash
cp -r EPII_CM55M_APP_S/app/scenario_app/tflm_yolo11_od \
      EPII_CM55M_APP_S/app/scenario_app/tflm_yolo26_od
```

Edit the makefile to set `APP_TYPE = tflm_yolo26_od` and swap the model type reference to YOLO26, then rebuild.

---

## Reading inference output

After flashing, SSCMA-Micro communicates over the same serial port using AT commands (921600 baud). Connect with any serial terminal (minicom, TeraTerm, screen):

```bash
minicom -D /dev/ttyACM0 -b 921600
```

Detection results are streamed as JSON over the AT protocol.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `xmodem_send.py` hangs | Wrong port or baud rate; check `--port` |
| Board not detected | Try pressing reset before running the command |
| Model not loaded after boot | Wrong model format (end2end vs raw tensors) or wrong flash address |
| `isValid()` fails silently | Model output count/shape doesn't match expected (check export flags) |
| Permission denied on `/dev/ttyACM0` | Run `sudo setfacl -m u:$USER:rw /dev/ttyACM0` |

---

## References

- [sscma-example-we2](https://github.com/Seeed-Studio/sscma-example-we2)
- [HimaxWiseEyePlus tflm_yolo11_od README](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2/blob/main/EPII_CM55M_APP_S/app/scenario_app/tflm_yolo11_od/README.md)
- [SSCMA-Micro model types](https://github.com/Seeed-Studio/SSCMA-Micro/tree/main/sscma/core/model)
