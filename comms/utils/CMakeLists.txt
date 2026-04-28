# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(CUDA REQUIRED)

file(GLOB_RECURSE UTILS_SOURCES
    "*.cc"
    "*.cu"
)
file(GLOB_RECURSE UTILS_CU_SOURCES
    "*.cu"
)
list(FILTER UTILS_SOURCES EXCLUDE REGEX ".*/tests/.*")
list(FILTER UTILS_SOURCES EXCLUDE REGEX ".*/test_utils/.*")
list(FILTER UTILS_CU_SOURCES EXCLUDE REGEX ".*/tests/.*")
list(FILTER UTILS_CU_SOURCES EXCLUDE REGEX ".*/test_utils/.*")

add_library(utils OBJECT
    ${UTILS_SOURCES}
    ${UTILS_CU_SOURCES}
)

target_compile_features(utils PRIVATE cxx_std_20)
target_compile_options(utils PRIVATE
    -fPIC
)
set_target_properties(utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(utils PUBLIC
    ${ROOT}
    ${CONDA_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
)
# Link folly for transitive include dirs (dwarf.h via Symbolizer.h).
# Only when the Folly::folly CMake target exists (standalone builds).
if(TARGET Folly::folly)
    target_link_libraries(utils PRIVATE Folly::folly)
endif()
