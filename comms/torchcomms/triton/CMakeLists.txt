# Copyright (c) Meta Platforms, Inc. and affiliates.
# =============================================================================
# Triton device bitcode (libtorchcomms_device.bc)
#
# Compiles TorchComms device API to LLVM bitcode (.bc) for Triton's extern_libs.
# Uses clang with CUDA device-only mode to produce portable bitcode.
# Mirrors the Buck genrule torchcomms_device_bitcode in
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
        " (requires NCCLX 2.28+), skipping libtorchcomms_device.bc build")
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

set(TRITON_BC_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms/triton/fb/libtorchcomms_device.bc")

# Ensure output directory exists
get_filename_component(TRITON_BC_DIR "${TRITON_BC_OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${TRITON_BC_DIR}")

add_custom_command(
    OUTPUT "${TRITON_BC_OUTPUT}"
    COMMAND ${CMAKE_COMMAND} -E echo "Building Triton device bitcode..."
    COMMAND "${CLANG_EXECUTABLE}"
        -x cuda --cuda-device-only
        --cuda-gpu-arch=sm_90
        -c -emit-llvm -O1
        -std=gnu++17
        -D__clang_llvm_bitcode_lib__
        -DNCCL_GIN_PROXY_ENABLE=0
        -Wno-unknown-cuda-version
        -fcuda-flush-denormals-to-zero
        "--cuda-path=${CUDA_PATH}"
        -isystem "${CUDA_PATH}/include"
        "-I${TRITON_SRC}/ir_include"
        "-I${TRITON_TMPINC}"
        "-I${NCCLX_INCLUDE}"
        "-I${NCCLX_INCLUDE}/nccl_device"
        "-I${ROOT}"
        "${TRITON_SRC}/torchcomms_device.cu"
        -o "${TRITON_BC_OUTPUT}"
    DEPENDS
        "${TRITON_SRC}/torchcomms_device.cu"
        "${TRITON_SRC}/torchcomms_device.h"
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
