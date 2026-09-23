#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdlib.h>

#include "stm32f7xx_hal.h"

// ---------------------------------------------------------------------------
// Model imports
// ---------------------------------------------------------------------------
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define DWT_LAR_UNLOCK_KEY 0xC5ACCE55

extern UART_HandleTypeDef huart3;

extern "C" int _write(int file, char *ptr, int len)
{
    HAL_UART_Transmit(&huart3, (uint8_t *)ptr, len, HAL_MAX_DELAY);
    return len;
}

extern const unsigned char model_start[] asm("_binary_cnn_classifier_tflite_start");
extern const unsigned char model_end[] asm("_binary_cnn_classifier_tflite_end");

namespace {

constexpr int kInputSamples = 256;
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
TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_tensor = nullptr;

bool allocate_arena()
{
    tensor_arena = static_cast<uint8_t *>(malloc(kArenaSize));
    
    if (tensor_arena == nullptr) {
        std::printf("[tinyml] Unable to allocate TFLite tensor arena");
        return false;
    }
    arena_size = kArenaSize;
    std::printf("[tinyml] Tensor arena: %u bytes in internal RAM",
             static_cast<unsigned>(arena_size));
    return true;
}

bool init_model()
{
    if (!allocate_arena()) {
        return false;
    }

    const tflite::Model *model = tflite::GetModel(model_start);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        std::printf("[tinyml] Unsupported TFLite schema version: %lu",
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
        std::printf("[tinyml] Unable to register TFLite operators");
        return false;
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, arena_size);
    if (static_interpreter.AllocateTensors() != kTfLiteOk) {
        std::printf("[tinyml] AllocateTensors failed");
        return false;
    }

    interpreter = &static_interpreter;
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    std::printf("[tinyml] Model size: %u bytes, arena used: %u / %u bytes",
             static_cast<unsigned>(model_end - model_start),
             static_cast<unsigned>(interpreter->arena_used_bytes()),
             static_cast<unsigned>(arena_size));
    return true;
}

int classify(const float *beat)
{
    if (interpreter == nullptr) {
        std::printf("[tinyml] TFLite model is not initialized");
        return -1;
    }

    if (input_tensor->type == kTfLiteFloat32) {
        const int count = static_cast<int>(input_tensor->bytes / sizeof(float));
        std::memcpy(input_tensor->data.f, beat,
                    std::min(count, kInputSamples) * sizeof(float));
    } else if (input_tensor->type == kTfLiteInt8) {
        const int count = static_cast<int>(
            std::min(input_tensor->bytes, static_cast<size_t>(kInputSamples)));
        const float scale = input_tensor->params.scale;
        const int zero_point = input_tensor->params.zero_point;
        for (int i = 0; i < count; ++i) {
            const long q = std::lround(beat[i] / scale) + zero_point;
            input_tensor->data.int8[i] = static_cast<int8_t>(std::clamp(q, -128L, 127L));
        }
    } else {
        std::printf("[tinyml] Unsupported TFLite input type: %d", input_tensor->type);
        return -1;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        std::printf("[tinyml] TFLite Invoke failed");
        return -1;
    }

    int classes = kClassCount;
    if (output_tensor->type == kTfLiteFloat32) {
        classes = std::min(classes, static_cast<int>(output_tensor->bytes / sizeof(float)));
    } else if (output_tensor->type == kTfLiteInt8) {
        classes = std::min(classes, static_cast<int>(output_tensor->bytes));
    } else {
        std::printf("[tinyml] Unsupported TFLite output type: %d", output_tensor->type);
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
                static_cast<unsigned>(model_end - model_start),
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

void stm_timer_init() 
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = DWT_LAR_UNLOCK_KEY;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

int64_t stm_timer_get_time() 
{
    uint32_t cycles = DWT->CYCCNT;
    return ((int64_t)cycles * 1000000) / SystemCoreClock;
}

char *uart_read_line(UART_HandleTypeDef *huart, char *buffer, size_t buffer_size)
{
    size_t index = 0;
    while (index < buffer_size - 1) {
        uint8_t byte;
        if (HAL_UART_Receive(huart, &byte, 1, HAL_MAX_DELAY) != HAL_OK) {
            return nullptr;
        }
        buffer[index++] = (char)byte;
        if (byte == '\n') {
            break;
        }
    }
    buffer[index] = '\0';
    return buffer;
}

}

extern "C" void tinyml_app_main(void)
{
    stm_timer_init();
    
    std::printf("[tinyml] Model: %s", kModelName);
    if (!init_model()) {
        std::printf("[tinyml] Model initialization failed");
        return;
    }

    print_model_info();
    std::printf("[tinyml] Ready. Send one 256-sample CSV beat per line.");

    // static so the main task stack is not blown
    static char line[kLineSize];
    static float beat[kInputSamples];

    while (uart_read_line(&huart3, line, sizeof(line)) != nullptr) {
        if (is_blank(line)) {
            continue;
        }
        if (!parse_beat(line, beat)) {
            std::printf("[tinyml] Expected 256 comma-separated samples");
            continue;
        }

        // Measured on the device: preprocessing + inference only, with the
        // serial transfer left out.
        const int64_t filter_start_us = stm_timer_get_time();
        filter_sos(beat);
        const int64_t inference_start_us = stm_timer_get_time();
        const int prediction = classify(beat);
        const int64_t end_us = stm_timer_get_time();

        const long long filter_us = inference_start_us - filter_start_us;
        const long long inference_us = end_us - inference_start_us;

        // Machine-readable result line for the host script.
        std::printf("RESULT,%d,%lld,%lld\n", prediction, inference_us, filter_us);
        std::fflush(stdout);
    }

    std::printf("[tinyml] stdin closed, leaving tinyml_app_main");
}