# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

# When USE_NCCLX is enabled and we're not using system libs, the NCCLX build
# dir must already exist: build it ahead of time with ./build_ncclx.sh from
# the repo root (python setup.py does this automatically when needed).
# Configure fails fast here so a missing build breaks in seconds with the
# recovery command instead of deep inside find_package().
if(USE_NCCLX AND NOT USE_SYSTEM_LIBS)
    set(NCCLX_BUILD_DIR $ENV{BUILDDIR})
    if(NOT NCCLX_BUILD_DIR)
        set(NCCLX_BUILD_DIR "${ROOT}/build/ncclx")
    endif()
    if(NOT EXISTS "${NCCLX_BUILD_DIR}")
        message(FATAL_ERROR
            "NCCLX build dir not found at ${NCCLX_BUILD_DIR}. "
            "Build it first from the repo root with ./build_ncclx.sh "
            "(or point BUILDDIR at an existing build dir).")
    endif()
endif()
