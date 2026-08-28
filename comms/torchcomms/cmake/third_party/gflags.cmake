# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

# We need static gflags — shared libs can't be shipped in wheels.
# Check for static library first (e.g., installed by build_ncclx.sh).
# This avoids find_package creating SHARED IMPORTED targets that can't be
# rejected without a duplicate-target conflict in the FetchContent fallback.
if(EXISTS "${CONDA_INCLUDE}/gflags/gflags.h")
    # Prefer static, fall back to shared.
    if(EXISTS "${CONDA_LIB}/libgflags.a")
        set(_GFLAGS_LIB "${CONDA_LIB}/libgflags.a")
    elseif(EXISTS "${CONDA_LIB}/libgflags.so")
        set(_GFLAGS_LIB "${CONDA_LIB}/libgflags.so")
    else()
        set(_GFLAGS_LIB "gflags")
    endif()
    add_library(gflags::gflags INTERFACE IMPORTED GLOBAL)
    set_target_properties(gflags::gflags PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CONDA_INCLUDE}"
        INTERFACE_LINK_LIBRARIES "${_GFLAGS_LIB}"
    )
    message(STATUS "Using gflags: ${_GFLAGS_LIB}")
else()
    find_package(gflags 2.2.2 QUIET CONFIG NO_CMAKE_PACKAGE_REGISTRY)
    if(gflags_FOUND)
        message(STATUS "Found system gflags: ${gflags_VERSION}")
    else()
        message(STATUS "System gflags not found, fetching v2.2.2 via FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            gflags
            GIT_REPOSITORY https://github.com/gflags/gflags.git
            GIT_TAG v2.2.2
        )
        set(GFLAGS_SHARED OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
        FetchContent_Populate(gflags)
        # Patch cmake_minimum_required: gflags v2.2.2 uses VERSION 2.8.12 which is
        # a fatal error on CMake 3.31+ ("Compatibility with CMake < 3.5 removed").
        file(READ "${gflags_SOURCE_DIR}/CMakeLists.txt" _gflags_cml)
        string(REPLACE
            "cmake_minimum_required (VERSION 2.8.12)"
            "cmake_minimum_required (VERSION 3.5)"
            _gflags_cml "${_gflags_cml}")
        file(WRITE "${gflags_SOURCE_DIR}/CMakeLists.txt" "${_gflags_cml}")
        # Keep build artifacts in the build tree, not the source/install dir
        set(_save_archive_dir ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
        set(_save_lib_dir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
        add_subdirectory(${gflags_SOURCE_DIR} ${gflags_BINARY_DIR} EXCLUDE_FROM_ALL)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_save_archive_dir})
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${_save_lib_dir})
    endif()

    # gflags v2.2.2 doesn't always provide a gflags::gflags target:
    #   - Installed configs create gflags::gflags_nothreads_shared etc.
    #   - FetchContent/add_subdirectory creates gflags_nothreads_static etc.
    # Create gflags::gflags as an alias to the first available target.
    if(NOT TARGET gflags::gflags)
        foreach(_tgt
            gflags::gflags_nothreads_static gflags::gflags_static
            gflags_nothreads_static gflags_static
            gflags
            gflags::gflags_nothreads_shared gflags::gflags_shared
            gflags_nothreads_shared gflags_shared)
            if(TARGET ${_tgt})
                add_library(gflags::gflags ALIAS ${_tgt})
                message(STATUS "Created gflags::gflags alias -> ${_tgt}")
                break()
            endif()
        endforeach()
        if(NOT TARGET gflags::gflags)
            message(FATAL_ERROR "gflags found/built but no usable target exists")
        endif()
    endif()
endif()
