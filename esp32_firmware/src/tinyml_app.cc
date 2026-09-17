#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include "driver/uart.h"
#include "driver/uart_vfs.h"  // IDF < 5.3: use "esp_vfs_dev.h" + esp_vfs_dev_uart_use_driver
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "random_forest_classifier.h"
#include "svm_classifier.h"
#include "svm_classifier_scaler.h"

extern const unsigned char cnn_model_start[] asm("_binary_cnn_classifier_tflite_start");
extern const unsigned char cnn_model_end[] asm("_binary_cnn_classifier_tflite_end");
extern const unsigned char mlp_model_start[] asm("_binary_mlp_classifier_tflite_start");
extern const unsigned char mlp_model_end[] asm("_binary_mlp_classifier_tflite_end");

namespace {

constexpr char TAG[] = "tinyml";
constexpr int kInputSamples = 256;
constexpr int kFeatureCount = 12;
constexpr int kClassCount = 5;
constexpr int kFeatureWindow = 90;
constexpr size_t kArenaSizePsram = 2 * 1024 * 1024;
constexpr size_t kArenaSizeInternal = 256 * 1024;
constexpr int kLineSize = 4096;

uint8_t *tensor_arena = nullptr;
size_t arena_size = 0;

enum class Model { Cnn, RandomForest, Mlp, Svm };
Model selected_model = Model::Cnn;

const tflite::Model *tflite_model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_tensor = nullptr;

// Armazenamento para recriar o interpreter com placement new a cada troca de modelo
alignas(tflite::MicroInterpreter)
uint8_t interpreter_storage[sizeof(tflite::MicroInterpreter)];

tflite::MicroMutableOpResolver<16> resolver;
bool resolver_initialized = false;

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

void destroy_interpreter()
{
    if (interpreter != nullptr) {
        interpreter->~MicroInterpreter();
        interpreter = nullptr;
    }
    tflite_model = nullptr;
    input_tensor = nullptr;
    output_tensor = nullptr;
}

bool init_resolver()
{
    if (resolver_initialized) {
        return true;
    }
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddAveragePool2D() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddExpandDims() != kTfLiteOk ||   // gerado por Conv1D do Keras
        resolver.AddSqueeze() != kTfLiteOk ||      // gerado por Conv1D do Keras
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
    return true;
}

bool init_tflite(Model model)
{
    destroy_interpreter();

    if (!init_resolver()) {
        return false;
    }

    const bool is_cnn = (model == Model::Cnn);
    const unsigned char *model_data = is_cnn ? cnn_model_start : mlp_model_start;
    const size_t model_size = is_cnn
        ? static_cast<size_t>(cnn_model_end - cnn_model_start)
        : static_cast<size_t>(mlp_model_end - mlp_model_start);

    const tflite::Model *candidate = tflite::GetModel(model_data);
    if (candidate->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Unsupported TFLite schema version: %lu",
                 static_cast<unsigned long>(candidate->version()));
        return false;
    }
    tflite_model = candidate;

    interpreter = new (interpreter_storage)
        tflite::MicroInterpreter(tflite_model, resolver, tensor_arena, arena_size);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        destroy_interpreter();
        return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    ESP_LOGI(TAG, "Loaded %s model (%u bytes), arena used: %u / %u bytes",
             is_cnn ? "CNN" : "MLP",
             static_cast<unsigned>(model_size),
             static_cast<unsigned>(interpreter->arena_used_bytes()),
             static_cast<unsigned>(arena_size));
    return true;
}

int run_tflite(const float *beat)
{
    if (interpreter == nullptr || input_tensor == nullptr || output_tensor == nullptr) {
        ESP_LOGE(TAG, "TFLite model is not initialized");
        return -1;
    }

    if (input_tensor->type == kTfLiteFloat32) {
        const int count = static_cast<int>(input_tensor->bytes / sizeof(float));
        const int limit = std::min(count, kInputSamples);
        std::memcpy(input_tensor->data.f, beat, limit * sizeof(float));
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
            const long q = std::lround(features[i] * 1000.0f);
            quantized[i] = static_cast<int16_t>(std::clamp(q, -32768L, 32767L));
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
    char *token = std::strtok(line, ", \r\n");
    for (int i = 0; i < kInputSamples; ++i) {
        if (token == nullptr || std::sscanf(token, "%f", &beat[i]) != 1) {
            return false;
        }
        token = std::strtok(nullptr, ", \r\n");
    }
    return true;
}

bool allocate_arena()
{
    tensor_arena = static_cast<uint8_t *>(
        heap_caps_malloc(kArenaSizePsram, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (tensor_arena != nullptr) {
        arena_size = kArenaSizePsram;
        ESP_LOGI(TAG, "Tensor arena: %u bytes in PSRAM", static_cast<unsigned>(arena_size));
        return true;
    }

    tensor_arena = static_cast<uint8_t *>(
        heap_caps_malloc(kArenaSizeInternal, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (tensor_arena != nullptr) {
        arena_size = kArenaSizeInternal;
        ESP_LOGW(TAG, "No PSRAM, tensor arena: %u bytes in internal RAM",
                 static_cast<unsigned>(arena_size));
        return true;
    }

    ESP_LOGE(TAG, "Unable to allocate TFLite tensor arena");
    return false;
}

void init_console()
{
    // Sem o driver instalado, o stdin da UART é não-bloqueante e fgets retorna NULL na hora.
    // Se o console for a USB nativa (USB-Serial-JTAG), troque por
    // usb_serial_jtag_driver_install + usb_serial_jtag_vfs_use_driver.
    setvbuf(stdin, nullptr, _IONBF, 0);
    uart_driver_install(UART_NUM_0, 2 * kLineSize, 0, 0, nullptr, 0);
    uart_vfs_dev_use_driver(UART_NUM_0);
}

void select_tflite_model(Model model)
{
    selected_model = model;
    if (!init_tflite(model)) {
        ESP_LOGE(TAG, "Failed to load %s model", model == Model::Cnn ? "CNN" : "MLP");
    }
}

}  // namespace

extern "C" void tinyml_app_main(void)
{
    init_console();

    if (!allocate_arena()) {
        return;
    }

    select_tflite_model(selected_model);
    ESP_LOGI(TAG, "Ready. Send cnn, mlp, rf or svm, then one 256-sample CSV beat.");

    // static para não estourar a stack da main task
    static char line[kLineSize];
    static float beat[kInputSamples];

    while (std::fgets(line, sizeof(line), stdin) != nullptr) {
        if (std::strncmp(line, "cnn", 3) == 0) {
            select_tflite_model(Model::Cnn);
            continue;
        }
        if (std::strncmp(line, "mlp", 3) == 0) {
            select_tflite_model(Model::Mlp);
            continue;
        }
        if (std::strncmp(line, "rf", 2) == 0) {
            selected_model = Model::RandomForest;
            ESP_LOGI(TAG, "Selected Random Forest");
            continue;
        }
        if (std::strncmp(line, "svm", 3) == 0) {
            selected_model = Model::Svm;
            ESP_LOGI(TAG, "Selected SVM");
            continue;
        }
        if (!parse_beat(line, beat)) {
            ESP_LOGW(TAG, "Expected 256 comma-separated samples");
            continue;
        }
        filter_sos(beat);
        ESP_LOGI(TAG, "model=%d class=%d", static_cast<int>(selected_model), classify(beat));
    }

    ESP_LOGE(TAG, "stdin closed, leaving tinyml_app_main");
}