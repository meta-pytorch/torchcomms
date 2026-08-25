# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

# glog::glog        — links the glog library (static or shared).
# glog::glog_headers — INTERFACE (header-only), no library linked.
#
# libtorchcomms.so links glog::glog (gets symbols). Extensions that only
# need headers (gloo, nccl) use glog::glog_headers and resolve symbols
# from libtorchcomms.so at runtime via DT_NEEDED.
if(EXISTS "${CONDA_INCLUDE}/glog/logging.h" AND EXISTS "${CONDA_LIB}/libglog.a")
    # Static lib available from conda — use it directly (no runtime .so dependency).
    set(_GLOG_LIB "${CONDA_LIB}/libglog.a")
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
elseif(EXISTS "${CONDA_INCLUDE}/glog/logging.h")
    # Conda provides headers but only libglog.so (no libglog.a). Running
    # find_package here would discover the conda cmake config and create a
    # shared-lib target — defeating static linking. Skip find_package and
    # build glog statically from source instead.
    message(STATUS "Conda glog has no libglog.a — building glog statically via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        glog
        GIT_REPOSITORY https://github.com/google/glog.git
        GIT_TAG v0.4.0
    )
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
    set(_save_archive_dir ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    set(_save_lib_dir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    FetchContent_Populate(glog)
    add_subdirectory(${glog_SOURCE_DIR} ${glog_BINARY_DIR} EXCLUDE_FROM_ALL)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_save_archive_dir})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${_save_lib_dir})
    get_target_property(_glog_inc glog::glog INTERFACE_INCLUDE_DIRECTORIES)
    add_library(glog::glog_headers INTERFACE IMPORTED GLOBAL)
    set_target_properties(glog::glog_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_glog_inc}"
    )
else()
    # No conda glog — try system find_package, then fall back to FetchContent.
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
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
        set(_save_archive_dir ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
        set(_save_lib_dir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
        FetchContent_Populate(glog)
        add_subdirectory(${glog_SOURCE_DIR} ${glog_BINARY_DIR} EXCLUDE_FROM_ALL)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_save_archive_dir})
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${_save_lib_dir})
        get_target_property(_glog_inc glog::glog INTERFACE_INCLUDE_DIRECTORIES)
        add_library(glog::glog_headers INTERFACE IMPORTED GLOBAL)
        set_target_properties(glog::glog_headers PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_glog_inc}"
        )
    endif()
endif()
