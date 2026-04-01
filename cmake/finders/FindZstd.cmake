# Copyright (c) Meta Platforms, Inc. and affiliates.
# Stub Find module: short-circuits when zstd is provided via FetchContent.

# Check cache variables set by our zstd.cmake FetchContent wrapper.
# Note: cannot check ZSTD_FOUND here because find_package(Zstd) sets a local
# ZSTD_FOUND=FALSE that shadows the cache value before invoking this module.
if(ZSTD_LIBRARY AND ZSTD_INCLUDE_DIR)
    set(Zstd_FOUND TRUE)
    set(ZSTD_LIBRARIES "${ZSTD_LIBRARY}")
    set(ZSTD_INCLUDE_DIRS "${ZSTD_INCLUDE_DIR}")
    return()
endif()

# Check for zstd targets (FetchContent or conda find_package(zstd CONFIG)).
foreach(_zstd_target zstd::libzstd_static libzstd_static zstd::libzstd_shared libzstd_shared)
    if(TARGET ${_zstd_target})
        set(ZSTD_FOUND TRUE)
        set(Zstd_FOUND TRUE)
        set(ZSTD_LIBRARY ${_zstd_target})
        get_target_property(ZSTD_INCLUDE_DIR ${_zstd_target} INTERFACE_INCLUDE_DIRECTORIES)
        set(ZSTD_LIBRARIES "${ZSTD_LIBRARY}")
        set(ZSTD_INCLUDE_DIRS "${ZSTD_INCLUDE_DIR}")
        return()
    endif()
endforeach()

# Fallback: system search
find_path(ZSTD_INCLUDE_DIR zstd.h)
find_library(ZSTD_LIBRARY zstd)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Zstd DEFAULT_MSG ZSTD_LIBRARY ZSTD_INCLUDE_DIR)
set(ZSTD_LIBRARIES "${ZSTD_LIBRARY}")
set(ZSTD_INCLUDE_DIRS "${ZSTD_INCLUDE_DIR}")
