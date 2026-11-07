#include <stdint.h>
#include "camera_preprocess.h"
#include "inference.h"
#include "detection.h"

// Model symbol generated from model_int8_vela.tflite by xxd:
//   xxd -i model_int8_vela.tflite > firmware/model/model_data.cc
extern const unsigned char model_int8_vela_tflite[];
extern const unsigned int model_int8_vela_tflite_len;

#define CAMERA_W 330
#define CAMERA_H 240
#define MODEL_W 320
#define MODEL_H 320

static int8_t g_model_input[MODEL_W * MODEL_H * 3];

int main(void) {
    if (inference_init(model_int8_vela_tflite, model_int8_vela_tflite_len) != 0) {
        // TODO: board-specific error output
        while (1) {}
    }

    while (1) {
        // TODO: Replace with Seeed camera capture pointer.
        const uint8_t* camera_yuv422 = 0;

        // TODO: Read actual quantization params from input tensor.
        const float input_scale = 1.0f / 255.0f;
        const int input_zero_point = -128;

        if (camera_yuv422) {
            preprocess_yuv422_to_int8_rgb(
                camera_yuv422,
                CAMERA_W,
                CAMERA_H,
                g_model_input,
                MODEL_W,
                MODEL_H,
                input_scale,
                input_zero_point
            );

            detection_list_t detections;
            inference_run(g_model_input, &detections);

            // TODO: Display detections on LCD or send over UART/I2C.
        }
    }

    return 0;
}
