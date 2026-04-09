# Copyright (c) Meta Platforms, Inc. and affiliates.

set(UNIFLOW_NVML_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/NvmlApi.cpp
)

set(UNIFLOW_NVML_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/NvmlApi.h
)

add_library(uniflow_nvml OBJECT
    ${UNIFLOW_NVML_SOURCES}
    ${UNIFLOW_NVML_HEADERS}
)

target_compile_features(uniflow_nvml PUBLIC cxx_std_20)

target_compile_options(uniflow_nvml PRIVATE -fPIC)

target_include_directories(uniflow_nvml PUBLIC
    ${ROOT}
)

target_link_libraries(uniflow_nvml PRIVATE
    ${CMAKE_DL_LIBS}
)
