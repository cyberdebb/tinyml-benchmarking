#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "esp_cpu.h"
#include "esp_private/esp_clk.h"

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

#include "cnn_classifier.h"

namespace {

constexpr char TAG[] = "tinyml";
constexpr int kInputSamples = 256;
// RR features sent after the samples on each line, in the order of
// RR_FEATURES in ml/src/load_data.py: pre_rr_local, post_rr_local,
// post_pre_rr, pre_rr_long (all ratios between RR intervals).
constexpr int kRrFeatures = 4;
constexpr int kClassCount = 5;
constexpr int kLineSize = 4096;
constexpr char kModelName[] = "CNN";

constexpr float kSos[4][6] = {
    {0.00983804f, 0.01967608f, 0.00983804f, 1.0f, -0.87180078f, 0.21744963f},
    {1.0f, 2.0f, 1.0f, 1.0f, -1.12015830f, 0.58081390f},
    {1.0f, -2.0f, 1.0f, 1.0f, -1.98368737f, 0.98376571f},
    {1.0f, -2.0f, 1.0f, 1.0f, -1.99336671f, 0.99344307f},
};

void filter_sos(float *signal)
{
    // Subtract the window's starting level first, like filter_beats() in
    // ml/src/load_data.py: the filter starts from zero state, and without
    // this the window's DC offset turns into a transient across the window.
    const float offset = signal[0];
    for (int i = 0; i < kInputSamples; ++i) {
        signal[i] -= offset;
    }

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

constexpr size_t kArenaSize = 128 * 1024;

uint8_t *tensor_arena = nullptr;
size_t arena_size = 0;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *beat_tensor = nullptr;
TfLiteTensor *rr_tensor = nullptr;
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

int element_count(const TfLiteTensor *tensor)
{
    int count = 1;
    for (int i = 0; i < tensor->dims->size; ++i) {
        count *= tensor->dims->data[i];
    }
    return count;
}

bool init_model()
{
    if (!allocate_arena()) {
        return false;
    }

    const tflite::Model *model = tflite::GetModel(cnn_classifier_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Unsupported TFLite schema version: %lu",
                 static_cast<unsigned long>(model->version()));
        return false;
    }

    static tflite::MicroMutableOpResolver<20> resolver;
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddAveragePool2D() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddExpandDims() != kTfLiteOk ||
        resolver.AddSqueeze() != kTfLiteOk ||
        resolver.AddAdd() != kTfLiteOk ||
        resolver.AddConcatenation() != kTfLiteOk ||
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
    // Two inputs (beat, RR intervals). Each is found by its size: the
    // TFLite input order doesn't necessarily follow the Keras model's.
    for (size_t i = 0; i < interpreter->inputs_size(); ++i) {
        TfLiteTensor *tensor = interpreter->input(i);
        if (element_count(tensor) == kInputSamples) {
            beat_tensor = tensor;
        } else if (element_count(tensor) == kRrFeatures) {
            rr_tensor = tensor;
        }
    }
    if (beat_tensor == nullptr || rr_tensor == nullptr) {
        ESP_LOGE(TAG, "Model inputs are not 256 beat samples + 4 RR values");
        return false;
    }
    output_tensor = interpreter->output(0);

    ESP_LOGI(TAG, "Model size: %u bytes, arena used: %u / %u bytes",
             static_cast<unsigned>(sizeof(cnn_classifier_tflite)),
             static_cast<unsigned>(interpreter->arena_used_bytes()),
             static_cast<unsigned>(arena_size));
    return true;
}

void normalize_beat(const float *beat, float *normalized)
{
    float mean = 0.0f;
    for (int i = 0; i < kInputSamples; ++i) {
        mean += beat[i];
    }
    mean /= kInputSamples;

    float variance = 0.0f;
    for (int i = 0; i < kInputSamples; ++i) {
        const float delta = beat[i] - mean;
        variance += delta * delta;
    }
    const float std_dev = std::sqrt(variance / kInputSamples);

    for (int i = 0; i < kInputSamples; ++i) {
        normalized[i] = (beat[i] - mean) / (std_dev + 1e-6f);
    }
}

// Copies values into a float32 or int8 (quantized) input tensor.
bool fill_input(TfLiteTensor *tensor, const float *values, int count)
{
    if (tensor->type == kTfLiteFloat32) {
        std::memcpy(tensor->data.f, values, count * sizeof(float));
        return true;
    }
    if (tensor->type == kTfLiteInt8) {
        const float scale = tensor->params.scale;
        const int zero_point = tensor->params.zero_point;
        for (int i = 0; i < count; ++i) {
            const long q = std::lround(values[i] / scale) + zero_point;
            tensor->data.int8[i] = static_cast<int8_t>(std::clamp(q, -128L, 127L));
        }
        return true;
    }
    return false;
}

int classify(const float *beat, const float *rr)
{
    if (interpreter == nullptr) {
        ESP_LOGE(TAG, "TFLite model is not initialized");
        return -1;
    }

    // Same per-beat standardization as normalize_beats() in
    // cnn_classifier.py.
    float normalized[kInputSamples];
    normalize_beat(beat, normalized);

    if (!fill_input(beat_tensor, normalized, kInputSamples) ||
        !fill_input(rr_tensor, rr, kRrFeatures)) {
        ESP_LOGE(TAG, "Unsupported TFLite input type: %d", beat_tensor->type);
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
                static_cast<unsigned>(sizeof(cnn_classifier_tflite)),
                interpreter != nullptr
                    ? static_cast<unsigned>(interpreter->arena_used_bytes())
                    : 0u);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Serial entry
// ---------------------------------------------------------------------------
// Parses kInputSamples beat samples followed by kRrFeatures RR values.
bool parse_beat(char *line, float *beat, float *rr)
{
    char *token = std::strtok(line, ", \r\n");
    for (int i = 0; i < kInputSamples + kRrFeatures; ++i) {
        float *value = i < kInputSamples ? &beat[i] : &rr[i - kInputSamples];
        if (token == nullptr || std::sscanf(token, "%f", value) != 1) {
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

// Timing uses the CPU cycle counter (CCOUNT), the same method as the
// STM32 firmware (DWT->CYCCNT), instead of esp_timer, which has 1 us
// resolution and its own call overhead. CCOUNT is per core: the main task
// is pinned to CPU0 (CONFIG_ESP_MAIN_TASK_AFFINITY_CPU0), so start and end
// are read on the same core.
uint32_t esp32_timer_get_cycles()
{
    return esp_cpu_get_cycle_count();
}

// CCOUNT wraps every 2^32 cycles (~17.9 s at 240 MHz), so intervals are
// taken as an unsigned cycle difference, which stays correct across the
// wrap, and only then converted to microseconds.
long long esp32_cycles_to_us(uint32_t start, uint32_t end)
{
    const uint32_t cycles = end - start;
    return static_cast<long long>(
        (static_cast<uint64_t>(cycles) * 1000000u) /
        static_cast<uint64_t>(esp_clk_cpu_freq()));
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
    ESP_LOGI(TAG, "Ready. Send one 256-sample CSV beat + 4 RR values per line.");

    // static so the main task stack is not blown
    static char line[kLineSize];
    static float beat[kInputSamples];
    static float rr[kRrFeatures];

    while (std::fgets(line, sizeof(line), stdin) != nullptr) {
        if (is_blank(line)) {
            continue;
        }
        if (!parse_beat(line, beat, rr)) {
            ESP_LOGW(TAG, "Expected 256 samples + 4 RR values, comma-separated");
            continue;
        }

        // Measured on the device: preprocessing + inference only, with the
        // serial transfer left out.
        const uint32_t filter_start = esp32_timer_get_cycles();
        filter_sos(beat);
        const uint32_t inference_start = esp32_timer_get_cycles();
        const int prediction = classify(beat, rr);
        const uint32_t end = esp32_timer_get_cycles();

        const long long filter_us = esp32_cycles_to_us(filter_start, inference_start);
        const long long inference_us = esp32_cycles_to_us(inference_start, end);

        // Machine-readable result line for the host script.
        std::printf("RESULT,%d,%lld,%lld\n", prediction, inference_us, filter_us);
        std::fflush(stdout);
    }

    ESP_LOGE(TAG, "stdin closed, leaving tinyml_app_main");
}