#pragma once

#include <stdint.h>
#include "detection.h"

#ifdef __cplusplus
extern "C" {
#endif

// Placeholder interface. Wire this to Seeed/TFLite Micro runtime in the board SDK.
int inference_init(const unsigned char* model_data, unsigned int model_size);
int inference_run(const int8_t* input, detection_list_t* detections);

#ifdef __cplusplus
}
#endif
