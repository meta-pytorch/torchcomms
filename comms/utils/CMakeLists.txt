# Copyright (c) Meta Platforms, Inc. and affiliates.
file(GLOB_RECURSE UTILS_SOURCES
    "*.cc"
)
list(FILTER UTILS_SOURCES EXCLUDE REGEX ".*/tests/.*")

add_library(utils OBJECT
    ${UTILS_SOURCES}
)
target_compile_options(utils PRIVATE
    -fPIC
)

target_include_directories(utils PUBLIC
    ${ROOT}
    ${CONDA_INCLUDE}
    ${CUDA_INCLUDE}
)
