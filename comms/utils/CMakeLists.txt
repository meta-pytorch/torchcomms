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
target_compile_definitions(utils PRIVATE
    SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
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
target_link_libraries(utils PUBLIC spdlog::spdlog)
