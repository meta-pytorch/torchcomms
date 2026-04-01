# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(fmt REQUIRED)
elseif(CONDA_PREFIX)
    find_package(fmt QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
endif()

if(NOT TARGET fmt::fmt)
    include(FetchContent)
    set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
    set(FMT_TEST OFF CACHE BOOL "" FORCE)
    set(FMT_DOC OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG ${TORCHCOMMS_FMT_VERSION}
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(fmt)

    # Set variables so downstream find_package(fmt CONFIG) succeeds.
    set(fmt_FOUND TRUE CACHE BOOL "" FORCE)
    set(fmt_CONFIG "${fmt_SOURCE_DIR}/support/cmake/fmt-config.cmake" CACHE PATH "" FORCE)
    set(fmt_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config-stubs" CACHE PATH "" FORCE)
endif()
