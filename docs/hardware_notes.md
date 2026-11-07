# Hardware & Runtime Notes — Seeed Studio Grove Vision AI V2

## Chip

- **SoC**: Himax HX6537-A
- **CPU**: Dual Arm Cortex-M55 @ 400 MHz
- **NPU**: Arm Ethos-U55 (64 MACs)
- **Camera**: Himax HX6381 (330×240, YUV422)
- **Display**: 240×240 SPI LCD

## Memory

| Resource | Size |
|---|---|
| Internal SRAM | 512 KB |
| External PSRAM | 8 MB |
| Flash | 16 MB |

## Camera

- Format: YUV422 (YCbYCr)
- Resolution: 330×240 (default)
- FPS: up to 30

## Model Format

- **TFLite Micro** with **Ethos-U NPU delegate**
- Compiled with **Arm Vela** using `ethos-u55-64` config
- Fully quantized INT8 preferred

## Constraints

- Model + buffers must fit in available SRAM/PSRAM
- No dynamic shapes
- No float ops on NPU (INT8 only)
- Postprocessing (NMS, box decode) runs on Cortex-M55 CPU
- Input preprocessing must match training pipeline exactly
