#pragma once

// Input format and preprocessing shared by every model's firmware.
constexpr int kInputSamples = 256;
// RR features sent after the samples on each line, in the order of
// RR_FEATURES in ml/src/load_data.py: log_pre_rr_local, log_post_rr_local,
// log_post_pre_rr, log_pre_rr_long (logs of ratios between RR intervals).
constexpr int kRrFeatures = 4;
constexpr int kClassCount = 5;
constexpr int kLineSize = 4096;

// Band-pass filter applied in place to one beat window, the same as
// filter_beats() in ml/src/load_data.py.
void filter_sos(float *signal);

// Parses kInputSamples beat samples followed by kRrFeatures RR values.
bool parse_beat(char *line, float *beat, float *rr);

bool is_blank(const char *line);
