# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(glog REQUIRED)
elseif(CONDA_PREFIX)
    # Only try find_package if glog cmake config actually exists and is complete.
    if(EXISTS "${CONDA_PREFIX}/lib/cmake/glog/glog-targets.cmake")
        find_package(glog QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
    endif()
endif()

if(NOT TARGET glog::glog)
    include(FetchContent)
    set(WITH_GTEST OFF CACHE BOOL "" FORCE)
    set(WITH_UNWIND OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glog
        GIT_REPOSITORY https://github.com/google/glog.git
        GIT_TAG ${TORCHCOMMS_GLOG_VERSION}
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(glog)

    # Pre-set find_library/find_path result variables so folly's FindGlog.cmake
    # doesn't search for files (which don't exist with FetchContent).
    set(GLOG_FOUND TRUE CACHE BOOL "" FORCE)
    set(GLOG_INCLUDE_DIR "${glog_SOURCE_DIR}/src;${glog_BINARY_DIR}" CACHE PATH "" FORCE)
    set(GLOG_LIBRARY glog::glog CACHE STRING "" FORCE)
    set(GLOG_LIBRARY_RELEASE glog::glog CACHE FILEPATH "" FORCE)
    set(GLOG_LIBRARY_DEBUG glog::glog CACHE FILEPATH "" FORCE)

    # Make find_package(glog CONFIG) find our stub, not external installs.
    set(glog_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config-stubs" CACHE PATH "" FORCE)
endif()
