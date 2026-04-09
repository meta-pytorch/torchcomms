# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(zstd REQUIRED)
elseif(CONDA_PREFIX)
    find_package(zstd QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
endif()

if(NOT TARGET zstd::libzstd_static AND NOT TARGET zstd::libzstd_shared)
    include(FetchContent)
    FetchContent_Declare(
        zstd
        GIT_REPOSITORY https://github.com/facebook/zstd.git
        GIT_TAG ${TORCHCOMMS_ZSTD_VERSION}
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
        SOURCE_SUBDIR build/cmake
    )
    FetchContent_MakeAvailable(zstd)

    # Pre-set find_library/find_path result variables for downstream FindZstd.cmake
    set(ZSTD_FOUND TRUE CACHE BOOL "" FORCE)
    set(ZSTD_INCLUDE_DIR "${zstd_SOURCE_DIR}/lib" CACHE PATH "" FORCE)
    set(ZSTD_LIBRARY_RELEASE libzstd_static CACHE FILEPATH "" FORCE)
    set(ZSTD_LIBRARY_DEBUG libzstd_static CACHE FILEPATH "" FORCE)
    set(ZSTD_LIBRARY libzstd_static CACHE STRING "" FORCE)
    set(ZSTD_INCLUDE_DIRS "${zstd_SOURCE_DIR}/lib" CACHE PATH "" FORCE)
    set(ZSTD_LIBRARIES libzstd_static CACHE STRING "" FORCE)
else()
    # Conda/system zstd found — set cache vars for downstream FindZstd.cmake stubs.
    foreach(_zstd_tgt zstd::libzstd_static zstd::libzstd_shared)
        if(TARGET ${_zstd_tgt})
            set(ZSTD_FOUND TRUE CACHE BOOL "" FORCE)
            set(ZSTD_LIBRARY ${_zstd_tgt} CACHE STRING "" FORCE)
            set(ZSTD_LIBRARIES ${_zstd_tgt} CACHE STRING "" FORCE)
            get_target_property(_zstd_inc ${_zstd_tgt} INTERFACE_INCLUDE_DIRECTORIES)
            if(_zstd_inc)
                set(ZSTD_INCLUDE_DIR "${_zstd_inc}" CACHE PATH "" FORCE)
                set(ZSTD_INCLUDE_DIRS "${_zstd_inc}" CACHE PATH "" FORCE)
            endif()
            break()
        endif()
    endforeach()
endif()
