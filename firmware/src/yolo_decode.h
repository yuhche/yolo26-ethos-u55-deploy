#pragma once

#include <stdint.h>
#include "detection.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decode a simple YOLO-like raw output tensor.
// Expected per candidate layout: [cx, cy, w, h, objectness, class0, class1, ...]
// Values may be int8-quantized. Output boxes are normalized 0..1.
void yolo_decode_int8(
    const int8_t* output,
    int num_candidates,
    int num_classes,
    float output_scale,
    int output_zero_point,
    float confidence_threshold,
    detection_list_t* detections
);

#ifdef __cplusplus
}
#endif
