# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(CUDA REQUIRED)

file(GLOB COMMON_ROOT_SOURCES "*.cc" "*.cu")
file(GLOB_RECURSE COMMON_ALGO_SOURCES "algorithms/*.cu" "algorithms/*.cc")
list(FILTER COMMON_ALGO_SOURCES EXCLUDE REGEX ".*/tests/.*")
list(FILTER COMMON_ALGO_SOURCES EXCLUDE REGEX ".*/fb/.*")

add_library(common OBJECT
    ${COMMON_ROOT_SOURCES}
    ${COMMON_ALGO_SOURCES}
)

target_compile_features(common PRIVATE cxx_std_20)
target_compile_options(common PRIVATE -fPIC)
target_compile_options(common PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)
set_target_properties(common PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(common PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
)
