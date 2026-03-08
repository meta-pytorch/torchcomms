# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(CUDA REQUIRED)

file(GLOB TEST_UTILS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)

add_library(test_utils OBJECT
    ${TEST_UTILS_SOURCES}
)

target_compile_features(test_utils PRIVATE cxx_std_20)
target_compile_options(test_utils PRIVATE -fPIC)
target_include_directories(test_utils PUBLIC
    ${ROOT}
    ${CONDA_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
)
