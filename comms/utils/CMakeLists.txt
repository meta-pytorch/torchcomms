# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(CUDA REQUIRED)
find_package(spdlog 1.16.0 CONFIG QUIET)
if(NOT spdlog_FOUND)
    include(FetchContent)
    set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)
    set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.16.0
    )
    FetchContent_MakeAvailable(spdlog)
endif()

# Globbed relative to this directory so the filters below see only the path
# inside it. Matching absolute paths would also catch a checkout that merely
# sits under a directory of the same name (~/fb/torchcomms), and interpolating
# the checkout path into a regex breaks on any metacharacter in it (c++).
#
# fb/ holds source that depends on internal-only libraries -- today the
# switchstats client, which reaches a switch agent over ServiceRouter. Without
# the filter the recursive glob folds it into the utils object library, and so
# into libmccl.so, where the build fails on headers the conda env does not
# carry. The mirror strips fb/ for the same reason.
file(GLOB_RECURSE UTILS_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cc"
    "*.cu"
)
file(GLOB_RECURSE UTILS_CU_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cu"
)
list(FILTER UTILS_SOURCES EXCLUDE REGEX "(^|/)(tests|test_utils|fb)/")
list(FILTER UTILS_CU_SOURCES EXCLUDE REGEX "(^|/)(tests|test_utils|fb)/")

add_library(utils OBJECT
    ${UTILS_SOURCES}
    ${UTILS_CU_SOURCES}
)

target_compile_features(utils PRIVATE cxx_std_20)
target_compile_options(utils PRIVATE
    -fPIC
)
target_compile_definitions(utils PRIVATE
    SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO
    SPDLOG_FMT_EXTERNAL
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
target_link_libraries(utils PUBLIC spdlog::spdlog_header_only)
