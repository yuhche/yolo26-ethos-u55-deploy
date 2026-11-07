#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Convert YUV422 camera frame to quantized RGB model input.
// Input is expected as YCbYCr / YUYV bytes: Y0 U Y1 V.
void preprocess_yuv422_to_int8_rgb(
    const uint8_t* yuv422,
    int src_w,
    int src_h,
    int8_t* dst,
    int dst_w,
    int dst_h,
    float input_scale,
    int input_zero_point
);

#ifdef __cplusplus
}
#endif
