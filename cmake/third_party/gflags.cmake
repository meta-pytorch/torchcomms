# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(gflags REQUIRED)
elseif(CONDA_PREFIX)
    # NO_DEFAULT_PATH prevents finding gflags from unrelated environments.
    find_package(gflags QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
endif()

if(NOT TARGET gflags)
    include(FetchContent)
    set(BUILD_gflags_LIB ON CACHE BOOL "" FORCE)
    # Disable nothreads variant to avoid double-linking (both shared gflags
    # and static gflags_nothreads), which causes gflags flag registration
    # conflicts at thrift1 compiler runtime.
    set(BUILD_gflags_nothreads_LIB OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(INSTALL_HEADERS OFF CACHE BOOL "" FORCE)
    set(INSTALL_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(INSTALL_STATIC_LIBS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        gflags
        GIT_REPOSITORY https://github.com/gflags/gflags.git
        GIT_TAG ${TORCHCOMMS_GFLAGS_VERSION}
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(gflags)
    # gflags creates a shared lib by default. Create a static target alias
    # if only the shared target exists, or just ensure gflags_static is used.
    if(TARGET gflags_static AND NOT TARGET gflags)
        add_library(gflags ALIAS gflags_static)
    endif()

    # Pre-set find_library/find_path result variables for folly's FindGflags.cmake
    set(LIBGFLAGS_FOUND TRUE CACHE BOOL "" FORCE)
    set(LIBGFLAGS_INCLUDE_DIR "${gflags_SOURCE_DIR}/src" CACHE PATH "" FORCE)
    set(LIBGFLAGS_LIBRARY gflags CACHE STRING "" FORCE)
    set(LIBGFLAGS_LIBRARY_RELEASE gflags CACHE FILEPATH "" FORCE)
    set(LIBGFLAGS_LIBRARY_DEBUG gflags CACHE FILEPATH "" FORCE)

    # Make find_package(gflags CONFIG) find our stub, not external installs.
    set(gflags_FOUND TRUE CACHE BOOL "" FORCE)
    set(gflags_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config-stubs" CACHE PATH "" FORCE)
    set(GFLAGS_INCLUDE_DIRS "${gflags_SOURCE_DIR}/src;${gflags_BINARY_DIR}/include" CACHE PATH "" FORCE)
    set(GFLAGS_LIBRARIES gflags CACHE STRING "" FORCE)
endif()
