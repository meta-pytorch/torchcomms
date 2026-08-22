# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(CUDA REQUIRED)
find_package(spdlog 1.16.0 CONFIG REQUIRED)

file(GLOB TEST_UTILS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)

add_library(test_utils OBJECT
    ${TEST_UTILS_SOURCES}
)

target_compile_features(test_utils PRIVATE cxx_std_20)
target_compile_options(test_utils PRIVATE -fPIC)
target_compile_definitions(test_utils PRIVATE
    SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO
    SPDLOG_FMT_EXTERNAL
)
target_include_directories(test_utils PUBLIC
    ${ROOT}
    ${CONDA_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(test_utils PUBLIC fmt::fmt)
target_link_libraries(test_utils PRIVATE spdlog::spdlog)
