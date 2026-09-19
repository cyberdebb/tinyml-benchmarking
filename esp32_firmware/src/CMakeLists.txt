#########################
#  How to upload:
#########################

# pio run -e cnn -t upload
# pio run -e mlf -t upload
# pio run -e rf -t upload
# pio run -e svm -t upload

FILE(GLOB_RECURSE app_sources
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cc
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

if(NOT DEFINED TINYML_MODEL)
    set(TINYML_MODEL "cnn")
endif()

set(model_embed "")
if(TINYML_MODEL STREQUAL "cnn")
    set(model_embed "../cnn_classifier.tflite")
    set(model_define TINYML_MODEL_CNN)
elseif(TINYML_MODEL STREQUAL "mlp")
    set(model_embed "../mlp_classifier.tflite")
    set(model_define TINYML_MODEL_MLP)
elseif(TINYML_MODEL STREQUAL "rf")
    set(model_define TINYML_MODEL_RF)
elseif(TINYML_MODEL STREQUAL "svm")
    set(model_define TINYML_MODEL_SVM)
else()
    message(FATAL_ERROR "TINYML_MODEL invalido: ${TINYML_MODEL} (use cnn, mlp, rf ou svm)")
endif()

idf_component_register(
    SRCS ${app_sources}
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