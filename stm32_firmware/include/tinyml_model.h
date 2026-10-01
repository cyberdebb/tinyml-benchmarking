#pragma once

// Implemented by exactly one model_<name>.cc per build (the one the build
// environment selects), so every model can use the same names.
extern const char kModelName[];

bool init_model();

// Class id (0..kClassCount-1) of one already filtered beat, -1 on error.
int classify(const float *beat, const float *rr);

// Prints "INFO,<model name>,<model flash bytes>,<model RAM bytes>".
void print_model_info();
