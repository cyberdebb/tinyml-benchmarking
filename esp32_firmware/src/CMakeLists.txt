FILE(GLOB_RECURSE app_sources
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cc
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

idf_component_register(
    SRCS ${app_sources}
    INCLUDE_DIRS
        "."
        "${CMAKE_CURRENT_LIST_DIR}/../../ml/models"
    EMBED_FILES
        "../cnn_classifier.tflite"
        "../mlp_classifier.tflite"
    REQUIRES espressif__esp-tflite-micro
)