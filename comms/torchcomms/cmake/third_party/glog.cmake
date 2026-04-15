# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

# glog::glog        — links the glog library (static or shared).
# glog::glog_headers — INTERFACE (header-only), no library linked.
#
# libtorchcomms.so links glog::glog (gets symbols). Extensions that only
# need headers (gloo, nccl) use glog::glog_headers and resolve symbols
# from libtorchcomms.so at runtime via DT_NEEDED.
if(EXISTS "${CONDA_INCLUDE}/glog/logging.h")
    # Prefer static, fall back to shared, fall back to -lglog.
    if(EXISTS "${CONDA_LIB}/libglog.a")
        set(_GLOG_LIB "${CONDA_LIB}/libglog.a")
    elseif(EXISTS "${CONDA_LIB}/libglog.so")
        set(_GLOG_LIB "${CONDA_LIB}/libglog.so")
    else()
        set(_GLOG_LIB "glog")
    endif()
    add_library(glog::glog INTERFACE IMPORTED GLOBAL)
    set_target_properties(glog::glog PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CONDA_INCLUDE}"
        INTERFACE_LINK_LIBRARIES "${_GLOG_LIB}"
    )
    add_library(glog::glog_headers INTERFACE IMPORTED GLOBAL)
    set_target_properties(glog::glog_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CONDA_INCLUDE}"
    )
    message(STATUS "Using glog: ${_GLOG_LIB}")
else()
    find_package(glog 0.4.0 QUIET CONFIG NO_CMAKE_PACKAGE_REGISTRY)
    if(glog_FOUND)
        message(STATUS "Found system glog: ${glog_VERSION}")
        get_target_property(_glog_inc glog::glog INTERFACE_INCLUDE_DIRECTORIES)
        add_library(glog::glog_headers INTERFACE IMPORTED GLOBAL)
        set_target_properties(glog::glog_headers PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_glog_inc}"
        )
    else()
        message(STATUS "System glog not found, fetching v0.4.0 via FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            glog
            GIT_REPOSITORY https://github.com/google/glog.git
            GIT_TAG v0.4.0
        )
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
        # Keep build artifacts in the build tree
        set(_save_archive_dir ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
        set(_save_lib_dir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
        FetchContent_Populate(glog)
        add_subdirectory(${glog_SOURCE_DIR} ${glog_BINARY_DIR} EXCLUDE_FROM_ALL)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_save_archive_dir})
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${_save_lib_dir})
        # glog's CMake build creates a glog::glog target.
        # Create glog_headers from its include directories.
        get_target_property(_glog_inc glog::glog INTERFACE_INCLUDE_DIRECTORIES)
        add_library(glog::glog_headers INTERFACE IMPORTED GLOBAL)
        set_target_properties(glog::glog_headers PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_glog_inc}"
        )
    endif()
endif()
