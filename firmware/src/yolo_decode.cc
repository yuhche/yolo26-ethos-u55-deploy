#include "yolo_decode.h"

static float dequant_i8(int8_t q, float scale, int zero_point) {
    return ((int)q - zero_point) * scale;
}

static float sigmoid_fast(float x) {
    // Cheap bounded approximation acceptable for postprocess on MCU.
    if (x < -8.0f) return 0.0003f;
    if (x > 8.0f) return 0.9997f;
    return 0.5f + x / (2.0f * (1.0f + (x < 0 ? -x : x)));
}

void yolo_decode_int8(
    const int8_t* output,
    int num_candidates,
    int num_classes,
    float output_scale,
    int output_zero_point,
    float confidence_threshold,
    detection_list_t* detections
) {
    detections->count = 0;
    const int stride = 5 + num_classes;

    for (int i = 0; i < num_candidates; ++i) {
        const int8_t* row = output + i * stride;

        float cx = dequant_i8(row[0], output_scale, output_zero_point);
        float cy = dequant_i8(row[1], output_scale, output_zero_point);
        float w  = dequant_i8(row[2], output_scale, output_zero_point);
        float h  = dequant_i8(row[3], output_scale, output_zero_point);
        float obj = sigmoid_fast(dequant_i8(row[4], output_scale, output_zero_point));

        int best_cls = -1;
        float best_cls_score = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            float s = sigmoid_fast(dequant_i8(row[5 + c], output_scale, output_zero_point));
            if (s > best_cls_score) {
                best_cls_score = s;
                best_cls = c;
            }
        }

        float score = obj * best_cls_score;
        if (score < confidence_threshold) continue;
        if (detections->count >= YOLO_MAX_DETECTIONS) break;

        detection_t* det = &detections->items[detections->count++];
        det->x1 = cx - w * 0.5f;
        det->y1 = cy - h * 0.5f;
        det->x2 = cx + w * 0.5f;
        det->y2 = cy + h * 0.5f;
        det->score = score;
        det->class_id = best_cls;
    }
}
