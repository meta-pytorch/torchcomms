# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# CMake build for the HRDWRingBuffer Python bindings (`_native`)
# and the host-side kernel launcher TU (`_native_kernels.cu`).
#
# Uses the standard CMake CUDA language so nvcc handles the device→host
# fatbinary flow natively, and pybind11's add_module helper handles the
# Python extension linking.
#
# Mirrors the buck target `fbcode//comms/utils/hrdw_ring_buffer:_native`
# (`gpu_cpp_python_extension` with `cuda_srcs`) at the conda-env level.
#
# Install destination is configurable via `HRDW_INSTALL_DEST` (default
# `hrdw_ring_buffer`) so downstream feedstocks can drop the artifacts
# under a different package namespace.

cmake_minimum_required(VERSION 3.18)

# Target SM. Accept either:
#   -DCMAKE_CUDA_ARCHITECTURES=100        (native CMake)
#   -DGPU_ARCH=sm_100                     (matches the bitcode CMake's flag,
#                                          and feedstock build.sh's value)
# Must run BEFORE `project(... LANGUAGES CUDA)` — otherwise CMake's CUDA
# initialization picks a default (often sm_75) and our late override is
# ignored, producing a .so that won't load on GB200/Blackwell.
#
# When GPU_ARCH is passed, write CMAKE_CUDA_ARCHITECTURES into the cache
# with FORCE so we override any stale value from a prior configure (e.g.
# the build dir was first created on an sm_75 host and reused on a
# GB200). Without FORCE, the cached arch wins and we silently build for
# the wrong device.
if(DEFINED GPU_ARCH)
    # Strip `sm_` prefix and any `a`/`f` suffix; e.g. sm_100a -> 100.
    string(REGEX REPLACE "^sm_" "" _cuda_arch "${GPU_ARCH}")
    string(REGEX REPLACE "[af]$" "" _cuda_arch "${_cuda_arch}")
    set(CMAKE_CUDA_ARCHITECTURES "${_cuda_arch}" CACHE STRING
        "CUDA architectures to build (derived from -DGPU_ARCH=)" FORCE)
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 100)  # GB200 default
endif()

project(hrdw_ring_buffer_pyext LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

# Build the extension. The .cpp holds the pybind11 module definition;
# the .cu holds the explicit template instantiation of
# launchRingBufferWrite that the bindings call into. Merging both into
# the same module makes the .so self-contained at runtime.
pybind11_add_module(_native
    _native.cpp
    _native_kernels.cu
)

target_include_directories(_native PRIVATE "${ROOT}")
# `atomic` resolves `__atomic_load_16` / `__atomic_compare_exchange_16`
# libcalls that the System-scope reader's 16B atomic load lowers to on
# platforms without an inline LDP/cmpxchg16b instruction (aarch64
# without -march=...+lse2, x86_64 without -mcx16). Device-scope poll
# uses cudaMemcpy so doesn't strictly need libatomic, but the System
# scope does and both compile into the same .so.
target_link_libraries(_native PRIVATE CUDA::cudart atomic)

# nvcc needs to know to forward C++20 to the host compiler.
target_compile_options(_native PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(NOT DEFINED HRDW_INSTALL_DEST)
    set(HRDW_INSTALL_DEST "hrdw_ring_buffer")
endif()

install(TARGETS _native LIBRARY DESTINATION "${HRDW_INSTALL_DEST}")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/_native.pyi"
    DESTINATION "${HRDW_INSTALL_DEST}")
