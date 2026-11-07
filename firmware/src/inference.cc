#include "inference.h"
#include "yolo_decode.h"
#include "nms.h"

// TODO: Replace placeholders with board SDK/TFLite Micro integration.
// Expected integration points:
// - tflite::MicroInterpreter
// - Ethos-U delegate / NPU initialization
// - model input tensor quantization params
// - model output tensor pointer and quantization params

static const unsigned char* g_model_data = 0;
static unsigned int g_model_size = 0;

int inference_init(const unsigned char* model_data, unsigned int model_size) {
    g_model_data = model_data;
    g_model_size = model_size;

    // TODO: Create MicroInterpreter, allocate tensors, check input/output tensors.
    return (g_model_data && g_model_size > 0) ? 0 : -1;
}

int inference_run(const int8_t* input, detection_list_t* detections) {
    (void)input;

    // TODO: Copy input to TFLite tensor, invoke interpreter, then decode real output.
    // Temporary empty result keeps application linkable until SDK is selected.
    detections->count = 0;

    // Example for real implementation:
    // const int8_t* out = interpreter->output(0)->data.int8;
    // yolo_decode_int8(out, num_candidates, num_classes, out_scale, out_zero_point, 0.25f, detections);
    // nms_inplace(detections, 0.45f);

    return 0;
}
