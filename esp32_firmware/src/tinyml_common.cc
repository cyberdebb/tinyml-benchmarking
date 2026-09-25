#include "tinyml_common.h"

#include <cstdlib>
#include <cstring>

namespace {

constexpr float kSos[4][6] = {
    {0.00983804f, 0.01967608f, 0.00983804f, 1.0f, -0.87180078f, 0.21744963f},
    {1.0f, 2.0f, 1.0f, 1.0f, -1.12015830f, 0.58081390f},
    {1.0f, -2.0f, 1.0f, 1.0f, -1.98368737f, 0.98376571f},
    {1.0f, -2.0f, 1.0f, 1.0f, -1.99336671f, 0.99344307f},
};

}

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

bool parse_beat(char *line, float *beat, float *rr)
{
    char *token = std::strtok(line, ", \r\n");
    for (int i = 0; i < kInputSamples + kRrFeatures; ++i) {
        float *value = i < kInputSamples ? &beat[i] : &rr[i - kInputSamples];
        // strtof instead of sscanf("%f"): newlib-nano (STM32) leaves float
        // support out of scanf unless linked with -u _scanf_float, so
        // sscanf would silently convert nothing.
        if (token == nullptr) {
            return false;
        }
        char *end = nullptr;
        *value = std::strtof(token, &end);
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
