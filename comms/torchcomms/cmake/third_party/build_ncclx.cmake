# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

# When USE_NCCLX is enabled and we're not using system libs, run build_ncclx.sh
# to build gflags/glog/fmt/folly/fbthrift/NCCLX into CONDA_PREFIX. This must
# happen before the other third-party includes so find_package() can discover
# the libraries that build_ncclx.sh installs.
if(USE_NCCLX AND NOT USE_SYSTEM_LIBS)
    set(NCCLX_BUILD_DIR $ENV{BUILDDIR})
    if(NOT NCCLX_BUILD_DIR)
        set(NCCLX_BUILD_DIR "${ROOT}/build/ncclx")
    endif()
    if(NOT EXISTS "${NCCLX_BUILD_DIR}")
        message(STATUS "NCCLX build dir not found at ${NCCLX_BUILD_DIR}, running build_ncclx.sh...")
        execute_process(
            COMMAND ${ROOT}/build_ncclx.sh
            WORKING_DIRECTORY ${ROOT}
            RESULT_VARIABLE _ncclx_result
            ERROR_VARIABLE _ncclx_error
        )
        if(_ncclx_result)
            message(FATAL_ERROR "NCCLX build failed: ${_ncclx_result}\n${_ncclx_error}")
        endif()
    endif()
endif()
