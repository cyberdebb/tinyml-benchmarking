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
#include "svm_classifier.h"
#include "svm_classifier_scaler.h"

#define DWT_LAR_UNLOCK_KEY 0xC5ACCE55

extern UART_HandleTypeDef huart3;

extern "C" int _write(int file, char *ptr, int len)
{
    HAL_UART_Transmit(&huart3, (uint8_t *)ptr, len, HAL_MAX_DELAY);
    return len;
}

namespace {

constexpr int kInputSamples = 256;
constexpr int kClassCount = 5;
constexpr int kLineSize = 4096;
constexpr char kModelName[] = "SVM";

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

bool init_model()
{
    return true;
}

int classify(const float *beat)
{
    float features[kFeatureCount];
    extract_features(beat, features);

    float scaled[kFeatureCount];
    for (int i = 0; i < kFeatureCount; ++i) {
        scaled[i] = (features[i] - svm_scaler_mean[i]) / svm_scaler_scale[i];
    }
    return svm_predict(scaled);
}

void print_model_info()
{
    // Classic models live in the binary itself: no runtime arena. Model
    // size is SVM_MODEL_BYTES (svm_classifier.h): the actual flash
    // footprint of the support vectors/coefficients/intercepts arrays.
    std::printf("INFO,%s,%u,0\n", kModelName,
                static_cast<unsigned>(SVM_MODEL_BYTES));
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Serial entry
// ---------------------------------------------------------------------------
bool parse_beat(char *line, float *beat)
{
    char *token = std::strtok(line, ", \r\n");
    for (int i = 0; i < kInputSamples; ++i) {
        // strtof instead of sscanf("%f"): newlib-nano leaves float support
        // out of scanf unless linked with -u _scanf_float, so sscanf
        // would silently convert nothing.
        char *end = nullptr;
        if (token == nullptr) {
            return false;
        }
        beat[i] = std::strtof(token, &end);
        if (end == token) {
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

    std::printf("[tinyml] Model: %s\n", kModelName);
    if (!init_model()) {
        std::printf("[tinyml] Model initialization failed\n");
        return;
    }

    print_model_info();
    std::printf("[tinyml] Ready. Send one 256-sample CSV beat per line.\n");

    // static so the main task stack is not blown
    static char line[kLineSize];
    static float beat[kInputSamples];

    while (uart_read_line(&huart3, line, sizeof(line)) != nullptr) {
        if (is_blank(line)) {
            continue;
        }
        if (!parse_beat(line, beat)) {
            std::printf("[tinyml] Expected 256 comma-separated samples\n");
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

    std::printf("[tinyml] stdin closed, leaving tinyml_app_main\n");
}