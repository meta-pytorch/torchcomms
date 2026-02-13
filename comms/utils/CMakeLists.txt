# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(CUDA REQUIRED)

file(GLOB_RECURSE UTILS_SOURCES
    "*.cc"
    "*.cpp"
)
list(FILTER UTILS_SOURCES EXCLUDE REGEX ".*/tests/.*")

add_library(utils OBJECT
    ${UTILS_SOURCES}
)

target_compile_features(utils PRIVATE cxx_std_20)
target_compile_options(utils PRIVATE
    -fPIC
)
target_include_directories(utils PUBLIC
    ${ROOT}
    ${CONDA_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
)
