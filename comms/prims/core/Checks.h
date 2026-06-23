// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

// <cuda.h> (driver API) and CudaDriverLazy.h (real-cuda pfn_*, a non-hipified
// cpp_library) are NVIDIA-only. On AMD this header is hipified and consumed
// alongside <hip/hip_runtime.h>, so pulling the real driver headers would clash
// (ROCm vector-type aliases vs CUDA structs). Only checkCuError needs them, and
// no AMD code path uses checkCuError.
#if !defined(__HIP_PLATFORM_AMD__)
#include <cuda.h>

#include "comms/prims/platform/CudaDriverLazy.h"
#endif

#define PIPES_CUDA_CHECK(EXPR)                      \
  do {                                              \
    const cudaError_t err = EXPR;                   \
    if (err == cudaSuccess) {                       \
      break;                                        \
    }                                               \
    std::string error_message;                      \
    error_message.append(__FILE__);                 \
    error_message.append(":");                      \
    error_message.append(std::to_string(__LINE__)); \
    error_message.append(" CUDA error: ");          \
    error_message.append(cudaGetErrorString(err));  \
    throw std::runtime_error(error_message);        \
  } while (0)

#define PIPES_KERNEL_LAUNCH_CHECK() PIPES_CUDA_CHECK(cudaGetLastError())

namespace comms::prims {

// Throws std::runtime_error if `err` is not cudaSuccess. The thrown message is
// `"<msg>: <cudaGetErrorString(err)>"`. Callers typically pass a literal that
// identifies the failing CUDA runtime call. Prefer PIPES_CUDA_CHECK when source
// location is useful.
inline void checkCudaError(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
  }
}

// Throws std::runtime_error if `err` is not CUDA_SUCCESS. The thrown message is
// `"<msg>: <cuGetErrorString(err)>"`. Requires cuda_driver_lazy_init() to have
// resolved pfn_cuGetErrorString. NVIDIA-only (CUDA driver API); not available
// on AMD/HIP, where no caller references it.
#if !defined(__HIP_PLATFORM_AMD__)
inline void checkCuError(CUresult err, const char* msg) {
  if (err != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    pfn_cuGetErrorString(err, &errStr);
    throw std::runtime_error(
        std::string(msg) + ": " + (errStr ? errStr : "unknown error"));
  }
}
#endif // !defined(__HIP_PLATFORM_AMD__)

} // namespace comms::prims
