#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YOLO_MAX_DETECTIONS 64

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
} detection_t;

typedef struct {
    detection_t items[YOLO_MAX_DETECTIONS];
    int count;
} detection_list_t;

#ifdef __cplusplus
}
#endif
