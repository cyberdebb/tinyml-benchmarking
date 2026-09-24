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
#include "svm_classifier.h"
#include "svm_classifier_scaler.h"

namespace {

constexpr char TAG[] = "tinyml";
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
        const uint32_t filter_start = esp32_timer_get_cycles();
        filter_sos(beat);
        const uint32_t inference_start = esp32_timer_get_cycles();
        const int prediction = classify(beat);
        const uint32_t end = esp32_timer_get_cycles();

        const long long filter_us = esp32_cycles_to_us(filter_start, inference_start);
        const long long inference_us = esp32_cycles_to_us(inference_start, end);

        // Machine-readable result line for the host script.
        std::printf("RESULT,%d,%lld,%lld\n", prediction, inference_us, filter_us);
        std::fflush(stdout);
    }

    ESP_LOGE(TAG, "stdin closed, leaving tinyml_app_main");
}