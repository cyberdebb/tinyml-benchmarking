#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <esp_timer.h>

#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

// ---------------------------------------------------------------------------
// Model imports
// ---------------------------------------------------------------------------
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "mlp_classifier.h"

namespace {

constexpr char TAG[] = "tinyml";
constexpr int kInputSamples = 256;
constexpr int kClassCount = 5;
constexpr int kLineSize = 4096;
constexpr char kModelName[] = "MLP";

constexpr float kSos[4][6] = {
    {0.00983804f, 0.01967608f, 0.00983804f, 1.0f, -0.87180078f, 0.21744963f},
    {1.0f, 2.0f, 1.0f, 1.0f, -1.12015830f, 0.58081390f},
    {1.0f, -2.0f, 1.0f, 1.0f, -1.98368737f, 0.98376571f},
    {1.0f, -2.0f, 1.0f, 1.0f, -1.99336671f, 0.99344307f},
};

void filter_sos(float *signal)
{
    for (int section = 0; section < 4; ++section) {
        float state_1 = 0.0f;
        float state_2 = 0.0f;
        for (int i = 0; i < kInputSamples; ++i) {
            const float input = signal[i];
            const float output = kSos[section][0] * input + state_1;
            state_1 = kSos[section][1] * input - kSos[section][4] * output + state_2;
            state_2 = kSos[section][2] * input - kSos[section][5] * output;
            signal[i] = output;
        }
    }
}

constexpr int kFeatureCount = 12;
constexpr int kFeatureWindow = 90;

void extract_features(const float *signal, float *features)
{
    const int start = (kInputSamples - kFeatureWindow) / 2;
    float values[kFeatureWindow];
    for (int i = 0; i < kFeatureWindow; ++i) {
        values[i] = signal[start + i];
    }

    float sorted[kFeatureWindow];
    std::memcpy(sorted, values, sizeof(sorted));
    std::sort(sorted, sorted + kFeatureWindow);

    float sum = 0.0f;
    float energy = 0.0f;
    for (float value : values) {
        sum += value;
        energy += value * value;
    }
    const float mean = sum / kFeatureWindow;

    float absolute_centered = 0.0f;
    float variance = 0.0f;
    for (float value : values) {
        const float delta = value - mean;
        absolute_centered += std::fabs(delta);
        variance += delta * delta;
    }
    variance /= kFeatureWindow;

    float maximum_derivative = 0.0f;
    for (int i = 1; i < kFeatureWindow; ++i) {
        maximum_derivative = std::max(maximum_derivative,
                                      std::fabs(values[i] - values[i - 1]));
    }

    const auto percentile = [&sorted](float fraction) {
        const float position = fraction * (kFeatureWindow - 1);
        const int lower = static_cast<int>(position);
        const int upper = std::min(lower + 1, kFeatureWindow - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    };

    int maximum_index = 0;
    for (int i = 1; i < kFeatureWindow; ++i) {
        if (values[i] > values[maximum_index]) {
            maximum_index = i;
        }
    }

    features[0] = mean;
    features[1] = std::sqrt(variance);
    features[2] = sorted[0];
    features[3] = sorted[kFeatureWindow - 1];
    features[4] = features[3] - features[2];
    features[5] = percentile(0.5f);
    features[6] = energy / kFeatureWindow;
    features[7] = absolute_centered / kFeatureWindow;
    features[8] = maximum_derivative;
    features[9] = percentile(0.1f);
    features[10] = percentile(0.9f);
    features[11] = static_cast<float>(maximum_index) / kFeatureWindow;
}

constexpr size_t kArenaSize = 128 * 1024;

uint8_t *tensor_arena = nullptr;
size_t arena_size = 0;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_tensor = nullptr;

bool allocate_arena()
{
    tensor_arena = static_cast<uint8_t *>(
        heap_caps_malloc(kArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Unable to allocate TFLite tensor arena");
        return false;
    }
    arena_size = kArenaSize;
    ESP_LOGI(TAG, "Tensor arena: %u bytes in internal RAM",
             static_cast<unsigned>(arena_size));
    return true;
}

bool init_model()
{
    if (!allocate_arena()) {
        return false;
    }

    const tflite::Model *model = tflite::GetModel(mlp_classifier_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Unsupported TFLite schema version: %lu",
                 static_cast<unsigned long>(model->version()));
        return false;
    }

    static tflite::MicroMutableOpResolver<16> resolver;
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddAveragePool2D() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddExpandDims() != kTfLiteOk ||
        resolver.AddSqueeze() != kTfLiteOk ||
        resolver.AddAdd() != kTfLiteOk ||
        resolver.AddMul() != kTfLiteOk ||
        resolver.AddRelu() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk ||
        resolver.AddQuantize() != kTfLiteOk ||
        resolver.AddMean() != kTfLiteOk ||
        resolver.AddDequantize() != kTfLiteOk) {
        ESP_LOGE(TAG, "Unable to register TFLite operators");
        return false;
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, arena_size);
    if (static_interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return false;
    }

    interpreter = &static_interpreter;
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    ESP_LOGI(TAG, "Model size: %u bytes, arena used: %u / %u bytes",
             static_cast<unsigned>(sizeof(mlp_classifier_tflite)),
             static_cast<unsigned>(interpreter->arena_used_bytes()),
             static_cast<unsigned>(arena_size));
    return true;
}

int classify(const float *beat)
{
    if (interpreter == nullptr) {
        ESP_LOGE(TAG, "TFLite model is not initialized");
        return -1;
    }

    float features[kFeatureCount];
    extract_features(beat, features);

    if (input_tensor->type == kTfLiteFloat32) {
        const int count = static_cast<int>(input_tensor->bytes / sizeof(float));
        std::memcpy(input_tensor->data.f, features,
                    std::min(count, kFeatureCount) * sizeof(float));
    } else if (input_tensor->type == kTfLiteInt8) {
        const int count = static_cast<int>(
            std::min(input_tensor->bytes, static_cast<size_t>(kFeatureCount)));
        const float scale = input_tensor->params.scale;
        const int zero_point = input_tensor->params.zero_point;
        for (int i = 0; i < count; ++i) {
            const long q = std::lround(features[i] / scale) + zero_point;
            input_tensor->data.int8[i] = static_cast<int8_t>(std::clamp(q, -128L, 127L));
        }
    } else {
        ESP_LOGE(TAG, "Unsupported TFLite input type: %d", input_tensor->type);
        return -1;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "TFLite Invoke failed");
        return -1;
    }

    int classes = kClassCount;
    if (output_tensor->type == kTfLiteFloat32) {
        classes = std::min(classes, static_cast<int>(output_tensor->bytes / sizeof(float)));
    } else if (output_tensor->type == kTfLiteInt8) {
        classes = std::min(classes, static_cast<int>(output_tensor->bytes));
    } else {
        ESP_LOGE(TAG, "Unsupported TFLite output type: %d", output_tensor->type);
        return -1;
    }

    int best = 0;
    float best_score = -INFINITY;
    for (int i = 0; i < classes; ++i) {
        const float score = (output_tensor->type == kTfLiteFloat32)
            ? output_tensor->data.f[i]
            : (output_tensor->data.int8[i] - output_tensor->params.zero_point) *
                  output_tensor->params.scale;
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

void print_model_info()
{
    std::printf("INFO,%s,%u,%u\n", kModelName,
                static_cast<unsigned>(sizeof(mlp_classifier_tflite)),
                interpreter != nullptr
                    ? static_cast<unsigned>(interpreter->arena_used_bytes())
                    : 0u);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Serial entry
// ---------------------------------------------------------------------------
bool parse_beat(char *line, float *beat)
{
    char *token = std::strtok(line, ", \r\n");
    for (int i = 0; i < kInputSamples; ++i) {
        if (token == nullptr || std::sscanf(token, "%f", &beat[i]) != 1) {
            return false;
        }
        token = std::strtok(nullptr, ", \r\n");
    }
    return true;
}

bool is_blank(const char *line)
{
    for (; *line != '\0'; ++line) {
        if (*line != ' ' && *line != '\t' && *line != '\r' && *line != '\n') {
            return false;
        }
    }
    return true;
}

void init_console()
{
    setvbuf(stdin, nullptr, _IONBF, 0);
    uart_driver_install(UART_NUM_0, 2 * kLineSize, 0, 0, nullptr, 0);
    uart_vfs_dev_use_driver(UART_NUM_0);
}

}

extern "C" void tinyml_app_main(void)
{
    init_console();

    ESP_LOGI(TAG, "Model: %s", kModelName);
    if (!init_model()) {
        ESP_LOGE(TAG, "Model initialization failed");
        return;
    }

    print_model_info();
    ESP_LOGI(TAG, "Ready. Send one 256-sample CSV beat per line.");

    // static so the main task stack is not blown
    static char line[kLineSize];
    static float beat[kInputSamples];

    while (std::fgets(line, sizeof(line), stdin) != nullptr) {
        if (is_blank(line)) {
            continue;
        }
        if (!parse_beat(line, beat)) {
            ESP_LOGW(TAG, "Expected 256 comma-separated samples");
            continue;
        }

        // Measured on the device: preprocessing + inference only, with the
        // serial transfer left out.
        const int64_t filter_start_us = esp_timer_get_time();
        filter_sos(beat);
        const int64_t inference_start_us = esp_timer_get_time();
        const int prediction = classify(beat);
        const int64_t end_us = esp_timer_get_time();

        const long long filter_us = inference_start_us - filter_start_us;
        const long long inference_us = end_us - inference_start_us;

        // Machine-readable result line for the host script.
        std::printf("RESULT,%d,%lld,%lld\n", prediction, inference_us, filter_us);
        std::fflush(stdout);
    }

    ESP_LOGE(TAG, "stdin closed, leaving tinyml_app_main");
}