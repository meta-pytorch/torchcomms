# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(ZLIB REQUIRED)
elseif(CONDA_PREFIX)
    find_package(ZLIB QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
endif()

if(NOT TARGET ZLIB::ZLIB)
    include(FetchContent)
    set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    # zlib v1.2.13 has cmake_minimum_required(VERSION 2.4) which newer cmake rejects.
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
    FetchContent_Declare(
        zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG ${TORCHCOMMS_ZLIB_VERSION}
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(zlib)
    # zlib 1.2.13 unconditionally creates test/example targets that fail to
    # compile (zconf.h not on their include path).  Exclude them from ALL.
    foreach(_zlib_test example minigzip example64 minigzip64)
        if(TARGET ${_zlib_test})
            set_target_properties(${_zlib_test} PROPERTIES EXCLUDE_FROM_ALL TRUE)
        endif()
    endforeach()
    # Create ZLIB::ZLIB alias if zlib only exports 'zlibstatic'/'zlib'
    if(NOT TARGET ZLIB::ZLIB AND TARGET zlibstatic)
        add_library(ZLIB::ZLIB ALIAS zlibstatic)
    endif()

    # zconf.h is generated in the build dir. Copy it to the source dir so that
    # a single ZLIB_INCLUDE_DIR works (folly's FindZLIB expects one path).
    file(COPY "${zlib_BINARY_DIR}/zconf.h" DESTINATION "${zlib_SOURCE_DIR}")

    set(ZLIB_FOUND TRUE CACHE BOOL "" FORCE)
    set(ZLIB_INCLUDE_DIR "${zlib_SOURCE_DIR}" CACHE PATH "" FORCE)
    set(ZLIB_INCLUDE_DIRS "${zlib_SOURCE_DIR}" CACHE PATH "" FORCE)
    set(ZLIB_LIBRARY zlibstatic CACHE STRING "" FORCE)
    set(ZLIB_LIBRARIES zlibstatic CACHE STRING "" FORCE)
endif()
