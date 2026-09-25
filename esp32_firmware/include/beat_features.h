#pragma once

#include "tinyml_common.h"

// Features of the MLP, RF and SVM: 12 morphology features followed by the
// RR values, same order as extract_neurokit_features() in ml/src/helpers.py.
constexpr int kMorphologyFeatures = 12;
constexpr int kFeatureCount = kMorphologyFeatures + kRrFeatures;
// 0.6 s around the R peak at 360 Hz (P wave, QRS and most of the T wave).
// Must match beat_features() in ml/src/helpers.py.
constexpr int kFeatureWindow = 216;

void extract_features(const float *signal, float *features);

// extract_features() + the RR values: the full kFeatureCount input.
void build_features(const float *beat, const float *rr, float *features);
