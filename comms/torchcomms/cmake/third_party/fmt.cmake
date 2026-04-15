# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

# We need static fmt — shared libs can't be shipped in wheels.
# Check for static library first (e.g., installed by build_ncclx.sh).
# This avoids find_package creating SHARED IMPORTED targets that can't be
# rejected without a duplicate-target conflict in the FetchContent fallback.
if(EXISTS "${CONDA_INCLUDE}/fmt/format.h")
    # Prefer static, fall back to shared.
    if(EXISTS "${CONDA_LIB}/libfmt.a")
        set(_FMT_LIB "${CONDA_LIB}/libfmt.a")
    elseif(EXISTS "${CONDA_LIB}/libfmt.so")
        set(_FMT_LIB "${CONDA_LIB}/libfmt.so")
    else()
        set(_FMT_LIB "fmt")
    endif()
    add_library(fmt::fmt INTERFACE IMPORTED GLOBAL)
    set_target_properties(fmt::fmt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CONDA_INCLUDE}"
        INTERFACE_LINK_LIBRARIES "${_FMT_LIB}"
    )
    message(STATUS "Using fmt: ${_FMT_LIB}")
else()
    find_package(fmt 11.2.0 QUIET CONFIG NO_CMAKE_PACKAGE_REGISTRY)
    if(fmt_FOUND)
        message(STATUS "Found system fmt: ${fmt_VERSION}")
    else()
        message(STATUS "System fmt not found, fetching 11.2.0 header-only via FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            fmt
            GIT_REPOSITORY https://github.com/fmtlib/fmt.git
            GIT_TAG 11.2.0
        )
        FetchContent_Populate(fmt)
        add_library(fmt::fmt INTERFACE IMPORTED GLOBAL)
        set_target_properties(fmt::fmt PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${fmt_SOURCE_DIR}/include"
            INTERFACE_COMPILE_DEFINITIONS "FMT_HEADER_ONLY=1"
        )
    endif()
endif()
