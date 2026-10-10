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

# libcommsutils.so: comms/utils -- the cvars table, the logger, the trainer
# context, memtrace, colltrace and the shared helpers -- as one shared library
# per process, built once by comms/utils/Makefile and shipped by the commsutils
# package. This tree never builds a second copy: libctran/libmccl link the
# prebuilt one, so NCCLX and MCCL share NCCL's configuration and trainer state
# at runtime. The headers are the ones in this source tree, which is the same
# revision the library was built from. COMMSUTILS_LIBRARY_DIR names the
# directory holding it (the commsutils wheel's lib dir in a wheel build);
# otherwise it is looked up in the environment's lib dir, where the conda
# package installs it.
find_library(COMMSUTILS_LIBRARY
    NAMES commsutils
    HINTS ENV COMMSUTILS_LIBRARY_DIR ${CONDA_LIB}
)
if(NOT COMMSUTILS_LIBRARY)
    message(FATAL_ERROR
        "libcommsutils.so not found (COMMSUTILS_LIBRARY_DIR=$ENV{COMMSUTILS_LIBRARY_DIR}, "
        "CONDA_LIB=${CONDA_LIB}); it is installed by the commsutils package.")
endif()
add_library(commsutils SHARED IMPORTED GLOBAL)
set_target_properties(commsutils PROPERTIES
    IMPORTED_LOCATION "${COMMSUTILS_LIBRARY}"
)
target_compile_definitions(commsutils INTERFACE
    SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO
    SPDLOG_FMT_EXTERNAL
)
target_include_directories(commsutils INTERFACE
    ${ROOT}
    ${CONDA_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(commsutils INTERFACE spdlog::spdlog_header_only)
