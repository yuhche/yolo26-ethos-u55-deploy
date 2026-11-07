#include "camera_preprocess.h"

static inline uint8_t clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

static inline int8_t quantize_u8_to_i8(uint8_t v, float scale, int zero_point) {
    // Real value is v / 255.0 because training/export uses 0..1 RGB.
    const float real = (float)v / 255.0f;
    int q = (int)(real / scale + (float)zero_point + 0.5f);
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return (int8_t)q;
}

static inline void yuv_to_rgb(int y, int u, int v, uint8_t* r, uint8_t* g, uint8_t* b) {
    // BT.601 integer approximation.
    int c = y - 16;
    int d = u - 128;
    int e = v - 128;

    int rr = (298 * c + 409 * e + 128) >> 8;
    int gg = (298 * c - 100 * d - 208 * e + 128) >> 8;
    int bb = (298 * c + 516 * d + 128) >> 8;

    *r = clamp_u8(rr);
    *g = clamp_u8(gg);
    *b = clamp_u8(bb);
}

void preprocess_yuv422_to_int8_rgb(
    const uint8_t* yuv422,
    int src_w,
    int src_h,
    int8_t* dst,
    int dst_w,
    int dst_h,
    float input_scale,
    int input_zero_point
) {
    // Nearest-neighbor resize + YUV422→RGB + int8 quantization.
    // Output layout: NHWC RGB, batch dimension omitted.
    for (int y = 0; y < dst_h; ++y) {
        int sy = (y * src_h) / dst_h;
        for (int x = 0; x < dst_w; ++x) {
            int sx = (x * src_w) / dst_w;
            int pair_x = sx & ~1;
            int idx = (sy * src_w + pair_x) * 2;

            int y0 = yuv422[idx + 0];
            int u  = yuv422[idx + 1];
            int y1 = yuv422[idx + 2];
            int v  = yuv422[idx + 3];
            int yy = (sx & 1) ? y1 : y0;

            uint8_t r, g, b;
            yuv_to_rgb(yy, u, v, &r, &g, &b);

            int out = (y * dst_w + x) * 3;
            dst[out + 0] = quantize_u8_to_i8(r, input_scale, input_zero_point);
            dst[out + 1] = quantize_u8_to_i8(g, input_scale, input_zero_point);
            dst[out + 2] = quantize_u8_to_i8(b, input_scale, input_zero_point);
        }
    }
}
