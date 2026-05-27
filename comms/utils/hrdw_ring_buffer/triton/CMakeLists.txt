# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Standalone LLVM bitcode build for HRDWRingBuffer Triton integration.
# Independent of colltrace/torchcomms — only depends on comms/utils headers.
#
# Produces libhrdw_ring_buffer.bc for linking into Triton kernels via
# `extern_libs={"libhrdw_ring_buffer": <bc path>}` at JIT time (see
# `hrdw_ring_buffer.triton.wrap_kernel_inline` for the wiring).

cmake_minimum_required(VERSION 3.18)
project(hrdw_ring_buffer_bitcode LANGUAGES NONE)

# ---------- Toolchain detection ----------

set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")

if(NOT DEFINED CUDA_HOME AND DEFINED ENV{CUDA_HOME})
    set(CUDA_HOME "$ENV{CUDA_HOME}")
endif()
if(NOT DEFINED CUDA_HOME)
    message(FATAL_ERROR "Cannot find CUDA installation. Set CUDA_HOME.")
endif()
set(CUDA_PATH "${CUDA_HOME}")

if(NOT DEFINED CONDA_INCLUDE AND DEFINED ENV{CONDA_PREFIX})
    set(CONDA_INCLUDE "$ENV{CONDA_PREFIX}/include")
endif()

find_program(CLANG_EXECUTABLE clang REQUIRED)

# GPU arch — default to sm_90a (Hopper), override with -DGPU_ARCH=sm_100 etc.
if(NOT DEFINED GPU_ARCH)
    set(GPU_ARCH "sm_90a")
endif()

# ---------- Bitcode compilation ----------

set(UTILS_SRC "${ROOT}/comms/utils")
set(TRITON_SRC "${CMAKE_CURRENT_SOURCE_DIR}")

set(_out "${CMAKE_CURRENT_BINARY_DIR}/hrdw_ring_buffer/libhrdw_ring_buffer.bc")
get_filename_component(_out_dir "${_out}" DIRECTORY)
file(MAKE_DIRECTORY "${_out_dir}")

# Workarounds for compiling CUDA device-only TUs with clang against newer
# CUDA toolkits (≥13):
# - texture_fetch_functions.h was removed in CUDA 13 but clang's
#   __clang_cuda_runtime_wrapper.h still includes it; provide an empty stub
#   that resolves the include without polluting symbol space.
# - In NVPTX64 device mode __x86_64__ is undefined, so the system
#   <bits/wordsize.h> sets __WORDSIZE=32 and <stdint.h> typedefs
#   uintptr_t as 32-bit even though __SIZEOF_POINTER__=8. Override
#   __WORDSIZE=64.
set(_tmpinc "${CMAKE_CURRENT_BINARY_DIR}/cuda_workaround_include")
file(MAKE_DIRECTORY "${_tmpinc}/bits")
file(WRITE "${_tmpinc}/texture_fetch_functions.h" "")
file(WRITE "${_tmpinc}/bits/wordsize.h" "
#ifndef __WORDSIZE
# define __WORDSIZE 64
#endif
#define __WORDSIZE32_SIZE_ULONG 0
#define __WORDSIZE32_PTRDIFF_LONG 0
#ifdef __x86_64__
# define __WORDSIZE_TIME64_COMPAT32 1
# define __SYSCALL_WORDSIZE 64
#else
# define __WORDSIZE_TIME64_COMPAT32 0
#endif
")

add_custom_command(
    OUTPUT "${_out}"
    COMMAND ${CLANG_EXECUTABLE}
        --cuda-device-only
        --cuda-gpu-arch=${GPU_ARCH}
        --cuda-path=${CUDA_PATH}
        -x cuda
        -std=c++20
        -O2
        -emit-llvm
        -Wno-unknown-cuda-version
        # Triton's NVIDIA backend sets the LLVM module flag
        # `nvvm-reflect-ftz=1` on its own module at JIT-link time; pass
        # `-fcuda-flush-denormals-to-zero` so our bitcode carries the same
        # value (the LLVM linker errors out on "override"-style flag
        # conflicts between modules being linked).
        -fcuda-flush-denormals-to-zero
        -c
        -I${_tmpinc}
        -I${ROOT}
        -isystem ${CUDA_PATH}/include
        -isystem ${CUDA_PATH}/include/cccl
        -o "${_out}"
        "${TRITON_SRC}/hrdw_ring_buffer.cu"
    DEPENDS
        "${TRITON_SRC}/hrdw_ring_buffer.cu"
        "${TRITON_SRC}/hrdw_ring_buffer.h"
    COMMENT "Compiling HRDWRingBuffer Triton bitcode -> libhrdw_ring_buffer.bc"
)

add_custom_target(hrdw_ring_buffer_bitcode ALL DEPENDS "${_out}")

# Install destination is configurable so downstream feedstocks can drop
# the bitcode under a different package namespace (defaults to the
# canonical `hrdw_ring_buffer` package).
if(NOT DEFINED HRDW_INSTALL_DEST)
    set(HRDW_INSTALL_DEST "hrdw_ring_buffer")
endif()

install(FILES "${_out}" DESTINATION "${HRDW_INSTALL_DEST}")
