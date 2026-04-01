# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(mvfst REQUIRED)
elseif(CONDA_PREFIX)
    find_package(mvfst QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
endif()

if(NOT TARGET mvfst::mvfst)
    include(FetchContent)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        mvfst
        GIT_REPOSITORY https://github.com/facebook/mvfst.git
        GIT_TAG ${TORCHCOMMS_MVFST_VERSION}
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(mvfst)

    # mvfst's install(EXPORT ... NAMESPACE mvfst::) only applies at install
    # time.  fbthrift links against mvfst::mvfst_server_async_tran, so we
    # need to create the namespace alias for the in-tree build.
    if(TARGET mvfst_server_async_tran AND NOT TARGET mvfst::mvfst_server_async_tran)
        add_library(mvfst::mvfst_server_async_tran ALIAS mvfst_server_async_tran)
    endif()
    if(TARGET mvfst_server AND NOT TARGET mvfst::mvfst_server)
        add_library(mvfst::mvfst_server ALIAS mvfst_server)
    endif()

    # Make find_package(mvfst CONFIG) succeed for downstream libs.
    set(mvfst_FOUND TRUE CACHE BOOL "" FORCE)
    set(mvfst_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config-stubs" CACHE PATH "" FORCE)
endif()
