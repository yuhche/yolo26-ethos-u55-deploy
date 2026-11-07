# SSCMA Integration Notes

This project should target Seeed's SSCMA stack rather than a fully custom firmware runtime where possible.

Checked upstream libraries:

- `Seeed-Studio/SSCMA-Micro`: embedded C++ inference/runtime/AT server SDK
- `Seeed-Studio/python-sscma`: Python host client for Grove Vision/SenseCraft devices over serial or MQTT

## Key Finding

`SSCMA-Micro` already contains native YOLO26 support:

```text
sscma/core/model/ma_model_yolo26.cpp
sscma/core/model/ma_model_yolo26.h
MA_MODEL_TYPE_YOLO26 = 14
MA_MODEL_TYPE_YOLO26_POSE = 15
MA_MODEL_TYPE_YOLO26_SEG = 16
```

Therefore the preferred firmware route is:

```text
YOLO26 .pt
  → ONNX
  → INT8 TFLite
  → Vela-compiled TFLite with Ethos-U custom op
  → SSCMA-Micro model entry with type MA_MODEL_TYPE_YOLO26
  → SSCMA AT protocol / python-sscma host client
```

This means our earlier custom `firmware/src/yolo_decode.cc` and `nms.cc` are useful as fallback/reference only. For Grove Vision AI V2, use SSCMA-Micro's built-in `Yolo26` postprocess if the model output layout matches.

## Required YOLO26 Output Layout for SSCMA-Micro

`ma_model_yolo26.cpp` expects one input and at least six outputs.

Input:

```text
[1, H, W, 3] or [1, 3, H, W]
H >= 32
H % 32 == 0
C is 3 or 1
```

Outputs must be either interleaved or grouped.

### Interleaved layout

```text
output0: boxes stride 8   [1, 4, H/8,  W/8]
output1: cls   stride 8   [1, C, H/8,  W/8]
output2: boxes stride 16  [1, 4, H/16, W/16]
output3: cls   stride 16  [1, C, H/16, W/16]
output4: boxes stride 32  [1, 4, H/32, W/32]
output5: cls   stride 32  [1, C, H/32, W/32]
```

### Grouped layout

```text
output0: boxes stride 8   [1, 4, H/8,  W/8]
output1: boxes stride 16  [1, 4, H/16, W/16]
output2: boxes stride 32  [1, 4, H/32, W/32]
output3: cls   stride 8   [1, C, H/8,  W/8]
output4: cls   stride 16  [1, C, H/16, W/16]
output5: cls   stride 32  [1, C, H/32, W/32]
```

Box output values are interpreted as left/top/right/bottom distances from grid cell centers. Class logits are sigmoid-scored. SSCMA performs NMS internally.

## Thresholds and Host Commands

The SSCMA AT protocol supports:

```text
AT+ID?
AT+NAME?
AT+VER?
AT+MODELS?
AT+MODEL?
AT+MODEL=<MODEL_ID>
AT+TSCORE?
AT+TSCORE=<0-100>
AT+TIOU?
AT+TIOU=<0-100>
AT+INVOKE=<N_TIMES,DIFFERED,RESULT_ONLY>
AT+SAMPLE=<N_TIMES>
```

Object detection events return boxes as:

```json
"boxes": [[x_center, y_center, width, height, score, class_id]]
```

`python-sscma` wraps this with:

```python
from sscma.micro.client import SerialClient
from sscma.micro.device import Device

client = SerialClient("/dev/ttyACM0", 921600)
device = Device(client)
device.loop_start()
device.tscore = 60
device.tiou = 50
device.Invoke(-1, filter=False, show=True)
```

## Firmware Direction

Use SSCMA-Micro porting APIs:

- `ma_config_board.h`: enable TFLite + Ethos-U op
- `ma_device.cpp`: register model entry with `type = MA_MODEL_TYPE_YOLO26`
- `ma_camera.*`: Grove Vision AI V2 camera capture
- `ma_transport_*`: serial transport for AT protocol
- `ma_storage_*`: model/config storage

Important TFLite operator flag:

```c
#define MA_TFLITE_OP_ETHOS_U 1
```

Also enable only the TFLite ops used by the model, or use `MA_TFLITE_OP_ALL` during bring-up.

## Recommended Change to This Repo

1. Keep the conversion pipeline.
2. Add SSCMA layout validation before Vela.
3. Compile with Vela for `ethos-u55-64`.
4. Package/register the compiled model as SSCMA model type `14`.
5. Use `python-sscma` for host-side testing and result collection.

## Risk

The default Ultralytics export may not produce the six-output layout SSCMA-Micro expects. If it exports a single concatenated detection tensor, SSCMA's built-in YOLO26 postprocess will reject it. In that case, export/modify the YOLO26 head to emit the three box maps and three class maps directly before conversion.
