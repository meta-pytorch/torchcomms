# Copyright (c) Meta Platforms, Inc. and affiliates.
# =============================================================================
# Triton device bitcode (libdevice_window.bc)
#
# Compiles TorchComms device API to LLVM bitcode (.bc) for Triton's extern_libs.
# Uses clang with CUDA device-only mode to produce portable bitcode.
# Mirrors the Buck genrule device_window_bitcode in
# comms/torchcomms/triton/BUCK.
#
# Key flags:
#   -DNCCL_GIN_PROXY_ENABLE=0: Disables GIN proxy backend
#   -D__clang_llvm_bitcode_lib__: Enables clang-compatible code paths in NCCLX
# =============================================================================
# Triton device bitcode requires NCCL Device API headers (NCCLX 2.28+).
# Silently skip if not available, matching the pattern in ncclx/CMakeLists.txt.
if(NOT EXISTS "${CONDA_INCLUDE}/nccl_device/core.h")
    message(STATUS
        "Triton bitcode: NCCL Device API headers NOT found"
        " (requires NCCLX 2.28+), skipping libdevice_window.bc build")
    return()
endif()

find_program(CLANG_EXECUTABLE clang REQUIRED)

# Detect CUDA path
if(DEFINED ENV{CUDA_HOME})
    set(CUDA_PATH $ENV{CUDA_HOME})
elseif(EXISTS "/usr/local/cuda")
    set(CUDA_PATH "/usr/local/cuda")
else()
    message(FATAL_ERROR "Cannot find CUDA installation. Set CUDA_HOME.")
endif()
message(STATUS "  TRITON CUDA_PATH : ${CUDA_PATH}")

# NCCLX device headers (installed by nccl conda package or system)
set(NCCLX_INCLUDE "${CONDA_INCLUDE}")

# Triton source directory
set(TRITON_SRC "${ROOT}/comms/torchcomms/triton")

# NCCLX device headers use #include "doca_gpunetio/..." but the actual
# headers live at a nested path under nccl_device/gin/gdaki/. Create a
# symlink so clang can resolve the include via a -I on the parent dir.
set(TRITON_TMPINC "${CMAKE_CURRENT_BINARY_DIR}/triton_tmpinc")
file(MAKE_DIRECTORY "${TRITON_TMPINC}")
if(EXISTS "${NCCLX_INCLUDE}/nccl_device/gin/gdaki/doca-gpunetio/include"
   AND NOT EXISTS "${TRITON_TMPINC}/doca_gpunetio")
    file(CREATE_LINK
        "${NCCLX_INCLUDE}/nccl_device/gin/gdaki/doca-gpunetio/include"
        "${TRITON_TMPINC}/doca_gpunetio"
        SYMBOLIC)
elseif(EXISTS "${NCCLX_INCLUDE}/nccl_device/gin/gdaki/doca_gpunetio"
       AND NOT EXISTS "${TRITON_TMPINC}/doca_gpunetio")
    file(CREATE_LINK
        "${NCCLX_INCLUDE}/nccl_device/gin/gdaki/doca_gpunetio"
        "${TRITON_TMPINC}/doca_gpunetio"
        SYMBOLIC)
endif()

# CUDA 13 removed texture_fetch_functions.h (deprecated texture reference API).
# Clang's __clang_cuda_runtime_wrapper.h unconditionally includes it, causing a
# fatal error. Provide an empty stub so clang's wrapper resolves the include.
# Torchcomms device code does not use texture references.
if(NOT EXISTS "${CUDA_PATH}/include/texture_fetch_functions.h")
    file(WRITE "${TRITON_TMPINC}/texture_fetch_functions.h"
        "// Empty stub: texture_fetch_functions.h was removed in CUDA 13.\n"
        "// Provided for clang CUDA wrapper compatibility.\n")
    message(STATUS
        "  Created texture_fetch_functions.h stub"
        " for CUDA 13+ compatibility")
endif()

# CUDA 13.2 added _NV_RSQRT_SPECIFIER (noexcept attribute on rsqrt) to
# crt/math_functions.hpp. Clang versions before LLVM commit ab048ac6c033
# (March 2026) don't define this macro, causing a parse error. Define it
# as empty for device bitcode compilation where noexcept has no effect.
# https://github.com/llvm/llvm-project/pull/185701
set(CUDA_13_2_COMPAT_FLAGS "")
if(EXISTS "${CUDA_PATH}/include/crt/math_functions.hpp")
    file(STRINGS "${CUDA_PATH}/include/crt/math_functions.hpp" _has_rsqrt_spec
        REGEX "_NV_RSQRT_SPECIFIER" LIMIT_COUNT 1)
    if(_has_rsqrt_spec)
        list(APPEND CUDA_13_2_COMPAT_FLAGS "-D_NV_RSQRT_SPECIFIER=")
        message(STATUS "  Added -D_NV_RSQRT_SPECIFIER for CUDA 13.2+ / pre-patch clang compatibility")
    endif()
endif()

# ---- GPU architecture detection ----
# Priority: TORCHCOMMS_CUDA_ARCH > TORCH_CUDA_ARCH_LIST
#         > nvidia-smi auto-detect > sm_90 default
#
# Clang's NVPTX backend doesn't support all SM codes that nvcc does (e.g.
# sm_103, sm_100a, sm_100f, sm_103a are not recognized by clang 21). Map
# unsupported codes to the nearest clang-supported base SM. Device bitcode
# compiled for the base SM is forward-compatible with higher SM variants.
if(DEFINED TORCHCOMMS_CUDA_ARCH)
    set(_raw_sm "${TORCHCOMMS_CUDA_ARCH}")
elseif(DEFINED ENV{TORCHCOMMS_CUDA_ARCH})
    set(_raw_sm "$ENV{TORCHCOMMS_CUDA_ARCH}")
elseif(DEFINED TORCH_CUDA_ARCH_LIST)
    # TORCH_CUDA_ARCH_LIST uses dot notation (e.g. "9.0;9.0a;10.0")
    # Take the first entry and convert to SM code.
    list(GET TORCH_CUDA_ARCH_LIST 0 _first_arch)
    string(REPLACE "." "" _raw_sm "${_first_arch}")
elseif(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    string(REPLACE " " ";" _arch_list "$ENV{TORCH_CUDA_ARCH_LIST}")
    list(GET _arch_list 0 _first_arch)
    string(REPLACE "." "" _raw_sm "${_first_arch}")
else()
    # Auto-detect from nvidia-smi
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _smi_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _smi_result
    )
    if(_smi_result EQUAL 0 AND _smi_output)
        # Take the first GPU's compute capability
        # (e.g. "9.0" -> "90", "10.0" -> "100")
        string(REGEX MATCH "[0-9]+\\.[0-9]+" _first_cc "${_smi_output}")
        if(_first_cc)
            string(REPLACE "." "" _raw_sm "${_first_cc}")
        else()
            set(_raw_sm "90")
        endif()
    else()
        set(_raw_sm "90")
    endif()
endif()

# Apply clang SM overrides for unsupported SM codes
if(_raw_sm STREQUAL "100a" OR _raw_sm STREQUAL "100f"
   OR _raw_sm STREQUAL "103" OR _raw_sm STREQUAL "103a")
    set(_clang_sm "100")
else()
    set(_clang_sm "${_raw_sm}")
endif()

set(TRITON_GPU_ARCH_FLAG "--cuda-gpu-arch=sm_${_clang_sm}")
message(STATUS "  TRITON GPU arch  : sm_${_clang_sm} (raw: ${_raw_sm})")

set(TRITON_BC_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms/triton/fb/libdevice_window.bc")

# Ensure output directory exists
get_filename_component(TRITON_BC_DIR "${TRITON_BC_OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${TRITON_BC_DIR}")

add_custom_command(
    OUTPUT "${TRITON_BC_OUTPUT}"
    COMMAND ${CMAKE_COMMAND} -E echo "Building Triton device bitcode..."
    COMMAND "${CLANG_EXECUTABLE}"
        -x cuda --cuda-device-only
        ${TRITON_GPU_ARCH_FLAG}
        -c -emit-llvm -O1
        -std=gnu++17
        -D__clang_llvm_bitcode_lib__
        -DNCCL_GIN_PROXY_ENABLE=0
        -Wno-unknown-cuda-version
        ${CUDA_13_2_COMPAT_FLAGS}
        -fcuda-flush-denormals-to-zero
        "--cuda-path=${CUDA_PATH}"
        -isystem "${CUDA_PATH}/include"
        -isystem "${CUDA_PATH}/include/cccl"
        -isystem "${CONDA_INCLUDE}/cccl"
        "-I${TRITON_SRC}/ir_include"
        "-I${TRITON_TMPINC}"
        "-I${NCCLX_INCLUDE}"
        "-I${NCCLX_INCLUDE}/nccl_device"
        "-I${ROOT}"
        "${TRITON_SRC}/device_window.cu"
        -o "${TRITON_BC_OUTPUT}"
    DEPENDS
        "${TRITON_SRC}/device_window.cu"
        "${TRITON_SRC}/device_window.h"
        "${TRITON_SRC}/ir_include/nccl.h"
    COMMENT "Compiling TorchComms Triton device bitcode"
    VERBATIM
)
add_custom_target(torchcomms_triton_bitcode ALL
    DEPENDS "${TRITON_BC_OUTPUT}"
)

install(FILES "${TRITON_BC_OUTPUT}"
    DESTINATION triton/fb/
)
