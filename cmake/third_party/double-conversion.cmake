# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(double-conversion REQUIRED)
elseif(CONDA_PREFIX)
    find_package(double-conversion QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
endif()

if(NOT TARGET double-conversion::double-conversion)
    include(FetchContent)
    FetchContent_Declare(
        double-conversion
        GIT_REPOSITORY https://github.com/google/double-conversion.git
        GIT_TAG ${TORCHCOMMS_DOUBLE_CONVERSION_VERSION}
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(double-conversion)

    # Set variables for downstream Find stubs
    set(DOUBLE_CONVERSION_FOUND TRUE CACHE BOOL "" FORCE)
    set(DOUBLE_CONVERSION_INCLUDE_DIR "${double-conversion_SOURCE_DIR}" CACHE PATH "" FORCE)
    set(DOUBLE_CONVERSION_LIBRARY double-conversion CACHE STRING "" FORCE)
endif()
