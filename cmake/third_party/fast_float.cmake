# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(FastFloat REQUIRED)
elseif(CONDA_PREFIX)
    find_package(FastFloat QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
endif()

if(NOT TARGET FastFloat::fast_float)
    include(FetchContent)
    set(FASTFLOAT_INSTALL ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        fast_float
        GIT_REPOSITORY https://github.com/fastfloat/fast_float.git
        GIT_TAG ${TORCHCOMMS_FAST_FLOAT_VERSION}
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(fast_float)

    # Set variables for downstream Find stubs
    set(FASTFLOAT_FOUND TRUE CACHE BOOL "" FORCE)
    set(FASTFLOAT_INCLUDE_DIR "${fast_float_SOURCE_DIR}/include" CACHE PATH "" FORCE)
endif()
