#########################
#  How to upload:
#########################

# pio run -e cnn -t upload
# pio run -e mlp -t upload
# pio run -e rf -t upload
# pio run -e svm -t upload
#
# Current/energy test firmware (power_logger/README.md):
# pio run -e cnn_energy -t upload   (same for mlp, rf, svm)

if(NOT DEFINED TINYML_MODEL)
    set(TINYML_MODEL "cnn")
endif()

if(TINYML_MODEL STREQUAL "cnn")
    set(model_define TINYML_MODEL_CNN)
    set(model_source "model_cnn.cc")
elseif(TINYML_MODEL STREQUAL "mlp")
    set(model_define TINYML_MODEL_MLP)
    set(model_source "model_mlp.cc")
elseif(TINYML_MODEL STREQUAL "rf")
    set(model_define TINYML_MODEL_RF)
    set(model_source "model_rf.cc")
elseif(TINYML_MODEL STREQUAL "svm")
    set(model_define TINYML_MODEL_SVM)
    set(model_source "model_svm.cc")
else()
    message(FATAL_ERROR "TINYML_MODEL invalido: ${TINYML_MODEL} (use cnn, mlp, rf ou svm)")
endif()

# energy_app.cc replaces the serial DS2 benchmark app (tinyml_app.cc) in the
# <model>_energy environments.
if(TINYML_ENERGY)
    set(app_source "energy_app.cc")
else()
    set(app_source "tinyml_app.cc")
endif()

idf_component_register(
    SRCS
        "main.c"
        "${app_source}"
        "tinyml_common.cc"
        "beat_features.cc"
        "${model_source}"
    INCLUDE_DIRS
        "../include"
    REQUIRES
        espressif__esp-tflite-micro
        esp_driver_uart
        esp_driver_gpio
)

target_compile_definitions(${COMPONENT_LIB} PRIVATE ${model_define})
message(STATUS "TinyML model: ${TINYML_MODEL} (app: ${app_source})")
