#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "random_forest_classifier.h"
#include "svm_classifier.h"
#include "svm_classifier_scaler.h"

extern "C" const unsigned char cnn_model_start[] asm("_binary_cnn_classifier_tflite_start");
extern "C" const unsigned char cnn_model_end[] asm("_binary_cnn_classifier_tflite_end");
extern "C" const unsigned char mlp_model_start[] asm("_binary_mlp_classifier_tflite_start");
extern "C" const unsigned char mlp_model_end[] asm("_binary_mlp_classifier_tflite_end");

namespace {
constexpr char TAG[] = "tinyml";
constexpr int kInputSamples = 256;
constexpr int kFeatureCount = 12;
constexpr int kClassCount = 5;
constexpr int kFeatureWindow = 90;
constexpr int kArenaSize = 2 * 1024 * 1024;
uint8_t *tensor_arena;

enum class Model { Cnn, RandomForest, Mlp, Svm };
Model selected_model = Model::Cnn;
const tflite::Model *tflite_model;
tflite::MicroInterpreter *interpreter;
TfLiteTensor *input_tensor;
TfLiteTensor *output_tensor;

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
    float absolute_centered = 0.0f;
    float mean;
    for (float value : values) {
        sum += value;
        energy += value * value;
    }
    mean = sum / kFeatureWindow;
    for (float value : values) {
        absolute_centered += std::fabs(value - mean);
    }
    float variance = 0.0f;
    for (float value : values) {
        const float delta = value - mean;
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

int run_tflite(const float *beat)
{
    if (interpreter == nullptr || input_tensor == nullptr || output_tensor == nullptr) {
        ESP_LOGE(TAG, "TFLite model is not initialized");
        return -1;
    }
    if (input_tensor->type == kTfLiteFloat32) {
        const int count = input_tensor->bytes / sizeof(float);
        const int limit = std::min(count, kInputSamples);
        std::memcpy(input_tensor->data.f, beat, limit * sizeof(float));
    } else if (input_tensor->type == kTfLiteInt8) {
        const int count = static_cast<int>(
            std::min(input_tensor->bytes, static_cast<size_t>(kInputSamples)));
        for (int i = 0; i < count; ++i) {
            input_tensor->data.int8[i] = static_cast<int8_t>(
                std::lround(beat[i] / input_tensor->params.scale) +
                input_tensor->params.zero_point);
        }
    } else {
        ESP_LOGE(TAG, "Unsupported TFLite input type: %d", input_tensor->type);
        return -1;
    }
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "TFLite Invoke failed");
        return -1;
    }
    int best = 0;
    float best_score = -INFINITY;
    for (int i = 0; i < kClassCount; ++i) {
        float score;
        if (output_tensor->type == kTfLiteFloat32) {
            score = output_tensor->data.f[i];
        } else {
            score = (output_tensor->data.int8[i] - output_tensor->params.zero_point) *
                    output_tensor->params.scale;
        }
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

bool init_tflite(Model model)
{
    const unsigned char *model_data = model == Model::Cnn ? cnn_model_start : mlp_model_start;
    tflite_model = tflite::GetModel(model_data);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Unsupported TFLite schema version");
        return false;
    }
    static tflite::MicroMutableOpResolver<16> resolver;
    static bool resolver_initialized = false;
    if (!resolver_initialized) {
        if (resolver.AddConv2D() != kTfLiteOk ||
            resolver.AddDepthwiseConv2D() != kTfLiteOk ||
            resolver.AddFullyConnected() != kTfLiteOk ||
            resolver.AddMaxPool2D() != kTfLiteOk ||
            resolver.AddAveragePool2D() != kTfLiteOk ||
            resolver.AddReshape() != kTfLiteOk ||
            resolver.AddAdd() != kTfLiteOk ||
            resolver.AddMul() != kTfLiteOk ||
            resolver.AddRelu() != kTfLiteOk ||
            resolver.AddSoftmax() != kTfLiteOk ||
            resolver.AddQuantize() != kTfLiteOk ||
            resolver.AddDequantize() != kTfLiteOk) {
            ESP_LOGE(TAG, "Unable to register TFLite operators");
            return false;
        }
        resolver_initialized = true;
    }
    static tflite::MicroInterpreter static_interpreter(
        tflite_model, resolver, tensor_arena, kArenaSize);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return false;
    }
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    return true;
}

int classify(const float *beat)
{
    if (selected_model == Model::Cnn || selected_model == Model::Mlp) {
        return run_tflite(beat);
    }
    float features[kFeatureCount];
    extract_features(beat, features);
    if (selected_model == Model::RandomForest) {
        int16_t quantized[kFeatureCount];
        for (int i = 0; i < kFeatureCount; ++i) {
            quantized[i] = static_cast<int16_t>(std::lround(features[i] * 1000.0f));
        }
        return random_forest_predict(quantized, kFeatureCount);
    }
    float scaled[kFeatureCount];
    for (int i = 0; i < kFeatureCount; ++i) {
        scaled[i] = (features[i] - svm_scaler_mean[i]) / svm_scaler_scale[i];
    }
    return svm_predict(scaled);
}

bool parse_beat(char *line, float *beat)
{
    char *token = std::strtok(line, ", ");
    for (int i = 0; i < kInputSamples; ++i) {
        if (token == nullptr || std::sscanf(token, "%f", &beat[i]) != 1) {
            return false;
        }
        token = std::strtok(nullptr, ", ");
    }
    return true;
}

}  // namespace

extern "C" void tinyml_app_main(void)
{
    tensor_arena = static_cast<uint8_t *>(
        heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (tensor_arena == nullptr) {
        tensor_arena = static_cast<uint8_t *>(heap_caps_malloc(kArenaSize, MALLOC_CAP_8BIT));
    }
    if (tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Unable to allocate TFLite tensor arena");
        return;
    }
    init_tflite(selected_model);
    ESP_LOGI(TAG, "Ready. Send cnn, mlp, rf or svm, then one 256-sample CSV beat.");

    char line[4096];
    float beat[kInputSamples];
    while (std::fgets(line, sizeof(line), stdin) != nullptr) {
        if (std::strncmp(line, "cnn", 3) == 0) {
            selected_model = Model::Cnn;
            init_tflite(selected_model);
            continue;
        }
        if (std::strncmp(line, "mlp", 3) == 0) {
            selected_model = Model::Mlp;
            init_tflite(selected_model);
            continue;
        }
        if (std::strncmp(line, "rf", 2) == 0) {
            selected_model = Model::RandomForest;
            continue;
        }
        if (std::strncmp(line, "svm", 3) == 0) {
            selected_model = Model::Svm;
            continue;
        }
        if (!parse_beat(line, beat)) {
            ESP_LOGW(TAG, "Expected 256 comma-separated samples");
            continue;
        }
        filter_sos(beat);
        ESP_LOGI(TAG, "model=%d class=%d", static_cast<int>(selected_model), classify(beat));
    }
}
