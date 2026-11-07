#pragma once

#include "detection.h"

#ifdef __cplusplus
extern "C" {
#endif

void nms_inplace(detection_list_t* detections, float iou_threshold);

#ifdef __cplusplus
}
#endif
