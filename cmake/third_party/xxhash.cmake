# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(xxHash REQUIRED)
elseif(CONDA_PREFIX)
    find_package(xxHash QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
endif()

if(NOT TARGET xxHash::xxhash)
    include(FetchContent)
    set(XXHASH_BUILD_XXHSUM OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        xxhash
        GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
        GIT_TAG ${TORCHCOMMS_XXHASH_VERSION}
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
        SOURCE_SUBDIR cmake_unofficial
    )
    FetchContent_MakeAvailable(xxhash)

    # Pre-set variables for fbthrift's FindXxhash.cmake
    set(Xxhash_INCLUDE_DIR "${xxhash_SOURCE_DIR}" CACHE PATH "" FORCE)
    set(Xxhash_LIBRARY_RELEASE xxhash CACHE FILEPATH "" FORCE)
endif()
