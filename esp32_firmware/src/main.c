#if defined(TINYML_MODEL_CNN)
#include "src/tinyml_app_cnn.cc"

#elif defined(TINYML_MODEL_MLP)
#include "src/tinyml_app_mlp.cc"

#elif defined(TINYML_MODEL_RF)
#include "src/tinyml_app_rf.cc"

#elif defined(TINYML_MODEL_SVM)
#include "src/tinyml_app_svm.cc"

#endif

extern void tinyml_app_main(void);

void app_main(void) {
    tinyml_app_main();
}