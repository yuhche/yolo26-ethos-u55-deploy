#include "nms.h"

static float maxf(float a, float b) { return a > b ? a : b; }
static float minf(float a, float b) { return a < b ? a : b; }

static float box_area(const detection_t* d) {
    float w = maxf(0.0f, d->x2 - d->x1);
    float h = maxf(0.0f, d->y2 - d->y1);
    return w * h;
}

static float box_iou(const detection_t* a, const detection_t* b) {
    float x1 = maxf(a->x1, b->x1);
    float y1 = maxf(a->y1, b->y1);
    float x2 = minf(a->x2, b->x2);
    float y2 = minf(a->y2, b->y2);

    float inter_w = maxf(0.0f, x2 - x1);
    float inter_h = maxf(0.0f, y2 - y1);
    float inter = inter_w * inter_h;
    float uni = box_area(a) + box_area(b) - inter;
    return uni <= 0.0f ? 0.0f : inter / uni;
}

static void swap_detection(detection_t* a, detection_t* b) {
    detection_t tmp = *a;
    *a = *b;
    *b = tmp;
}

void nms_inplace(detection_list_t* detections, float iou_threshold) {
    // Sort by descending score. Small YOLO_MAX_DETECTIONS keeps O(n^2) acceptable.
    for (int i = 0; i < detections->count; ++i) {
        for (int j = i + 1; j < detections->count; ++j) {
            if (detections->items[j].score > detections->items[i].score) {
                swap_detection(&detections->items[i], &detections->items[j]);
            }
        }
    }

    int keep[YOLO_MAX_DETECTIONS] = {0};
    int kept = 0;

    for (int i = 0; i < detections->count; ++i) {
        int suppress = 0;
        for (int k = 0; k < kept; ++k) {
            detection_t* prev = &detections->items[keep[k]];
            detection_t* curr = &detections->items[i];
            if (prev->class_id == curr->class_id && box_iou(prev, curr) > iou_threshold) {
                suppress = 1;
                break;
            }
        }
        if (!suppress) {
            keep[kept++] = i;
        }
    }

    for (int i = 0; i < kept; ++i) {
        detections->items[i] = detections->items[keep[i]];
    }
    detections->count = kept;
}
