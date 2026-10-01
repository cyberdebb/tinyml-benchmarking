#include "beat_features.h"

#include <algorithm>
#include <cmath>
#include <cstring>

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

void build_features(const float *beat, const float *rr, float *features)
{
    extract_features(beat, features);
    for (int i = 0; i < kRrFeatures; ++i) {
        features[kMorphologyFeatures + i] = rr[i];
    }
}
