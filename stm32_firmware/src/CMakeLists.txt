#########################
#  How to upload:
#########################

# pio run -e cnn -t upload
# pio run -e mlp -t upload
# pio run -e rf -t upload
# pio run -e svm -t upload

if(NOT DEFINED TINYML_MODEL)
    set(TINYML_MODEL "cnn")
endif()

set(model_embed "")
if(TINYML_MODEL STREQUAL "cnn")
    set(model_embed "../cnn_classifier.tflite")
    set(model_define TINYML_MODEL_CNN)
    set(model_source "tinyml_app_cnn.cc")
elseif(TINYML_MODEL STREQUAL "mlp")
    set(model_embed "../mlp_classifier.tflite")
    set(model_define TINYML_MODEL_MLP)
    set(model_source "tinyml_app_mlp.cc")
elseif(TINYML_MODEL STREQUAL "rf")
    set(model_define TINYML_MODEL_RF)
    set(model_source "tinyml_app_rf.cc")
elseif(TINYML_MODEL STREQUAL "svm")
    set(model_define TINYML_MODEL_SVM)
    set(model_source "tinyml_app_svm.cc")
else()
    message(FATAL_ERROR "TINYML_MODEL invalido: ${TINYML_MODEL} (use cnn, mlp, rf ou svm)")
endif()

idf_component_register(
    SRCS
        "main.c"
        "${model_source}"
    INCLUDE_DIRS
        "."
        "${CMAKE_CURRENT_LIST_DIR}/../../ml/models"
    EMBED_FILES ${model_embed}
    REQUIRES
        espressif__esp-tflite-micro
        esp_driver_uart
)

target_compile_definitions(${COMPONENT_LIB} PRIVATE ${model_define})
message(STATUS "TinyML model: ${TINYML_MODEL}")
