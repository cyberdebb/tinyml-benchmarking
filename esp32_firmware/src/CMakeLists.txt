FILE(GLOB_RECURSE app_sources ${CMAKE_SOURCE_DIR}/src/*.*)

idf_component_register(
    SRCS ${app_sources}
    INCLUDE_DIRS
        "${CMAKE_CURRENT_LIST_DIR}/../../ml/models"
    REQUIRES espressif__esp-tflite-micro
)
