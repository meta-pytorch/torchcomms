# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(Libevent REQUIRED)
elseif(CONDA_PREFIX)
    find_package(Libevent QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
endif()

if(NOT TARGET libevent::core)
    include(FetchContent)
    set(EVENT__DISABLE_TESTS ON CACHE BOOL "" FORCE)
    set(EVENT__DISABLE_SAMPLES ON CACHE BOOL "" FORCE)
    set(EVENT__DISABLE_BENCHMARK ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        libevent
        GIT_REPOSITORY https://github.com/libevent/libevent.git
        GIT_TAG ${TORCHCOMMS_LIBEVENT_VERSION}
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(libevent)

    # Prevent the internal find_package(Libevent CONFIG QUIET) inside
    # folly's FindLibEvent.cmake from finding the FetchContent target.
    # That code path uses CMP0026 (get_target_property LOCATION) which
    # is removed in CMake 4.x.  Disabling the CONFIG find forces the
    # fallback find_path/find_library branch which uses our cached vars.
    set(CMAKE_DISABLE_FIND_PACKAGE_Libevent TRUE CACHE BOOL "" FORCE)
    set(LibEvent_FOUND TRUE CACHE BOOL "" FORCE)
    set(LIBEVENT_FOUND TRUE CACHE BOOL "" FORCE)
    set(LIBEVENT_INCLUDE_DIR "${libevent_SOURCE_DIR}/include;${libevent_BINARY_DIR}/include" CACHE PATH "" FORCE)
    set(LIBEVENT_LIB event CACHE STRING "" FORCE)
    # mvfst's FindLibevent uses LIBEVENT_LIBRARY as find_library result var:
    set(LIBEVENT_LIBRARY event CACHE STRING "" FORCE)
endif()
