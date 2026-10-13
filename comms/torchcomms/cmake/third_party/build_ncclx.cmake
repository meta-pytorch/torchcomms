# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

# When USE_NCCLX is enabled and we're not using system libs, run build_ncclx.sh
# to build gflags/glog/fmt/folly/NCCLX into CONDA_PREFIX. This must
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

    set(_torchcomms_source_cvars_dir "${ROOT}/comms/utils/cvars")
    set(_torchcomms_source_cvars_cc
        "${_torchcomms_source_cvars_dir}/nccl_cvars.cc")
    set(_torchcomms_source_cvars_h
        "${_torchcomms_source_cvars_dir}/nccl_cvars.h")
    if((EXISTS "${_torchcomms_source_cvars_cc}" AND
        NOT EXISTS "${_torchcomms_source_cvars_h}") OR
       (EXISTS "${_torchcomms_source_cvars_h}" AND
        NOT EXISTS "${_torchcomms_source_cvars_cc}"))
        message(FATAL_ERROR
            "NCCLX CVAR generation left a partial source-tree result under "
            "${_torchcomms_source_cvars_dir}")
    endif()

    # build_ncclx.sh normally generates these files in the source tree. A
    # caller that reuses an existing NCCLX build directory skips that script,
    # so generate the same inputs under the CMake build tree instead. This
    # keeps clean revision-addressed source materializations immutable.
    if(NOT EXISTS "${_torchcomms_source_cvars_cc}")
        if(NOT Python3_EXECUTABLE)
            find_package(Python3 COMPONENTS Interpreter REQUIRED)
        endif()
        set(TORCHCOMMS_GENERATED_CVARS_ROOT
            "${CMAKE_BINARY_DIR}/generated")
        set(TORCHCOMMS_GENERATED_CVARS_DIR
            "${TORCHCOMMS_GENERATED_CVARS_ROOT}/comms/utils/cvars")
        file(MAKE_DIRECTORY "${TORCHCOMMS_GENERATED_CVARS_DIR}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env
                "NCCL_CVARS_OUTPUT_DIR=${TORCHCOMMS_GENERATED_CVARS_DIR}"
                "${Python3_EXECUTABLE}"
                "${_torchcomms_source_cvars_dir}/extractcvars.py"
            WORKING_DIRECTORY "${_torchcomms_source_cvars_dir}"
            RESULT_VARIABLE _torchcomms_cvars_result
            OUTPUT_VARIABLE _torchcomms_cvars_stdout
            ERROR_VARIABLE _torchcomms_cvars_stderr
        )
        if(NOT _torchcomms_cvars_result EQUAL 0)
            message(FATAL_ERROR
                "NCCLX CVAR generation failed (${_torchcomms_cvars_result}):\n"
                "${_torchcomms_cvars_stdout}\n${_torchcomms_cvars_stderr}")
        endif()
        foreach(_torchcomms_generated_cvar nccl_cvars.cc nccl_cvars.h)
            if(NOT EXISTS
               "${TORCHCOMMS_GENERATED_CVARS_DIR}/${_torchcomms_generated_cvar}")
                message(FATAL_ERROR
                    "NCCLX CVAR generation did not produce "
                    "${_torchcomms_generated_cvar}")
            endif()
        endforeach()
    endif()
endif()
