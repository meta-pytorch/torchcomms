// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include <fmt/format.h>

#include "comms/utils/logger/LogUtils.h"

#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>
#endif

namespace comms::utils::cuda {

/**
 * Dynamically loads a CUDA driver function using the CUDA runtime API.
 * This avoids the need to link against libcuda.so at build time.
 *
 * Returns nullptr if the function cannot be loaded (e.g., driver not
 * available).
 */
template <typename FuncPtr>
inline FuncPtr loadDriverFunction(const char* funcName) {
  FuncPtr func = nullptr;
#if CUDART_VERSION >= 12000
  cudaDriverEntryPointQueryResult driverStatus =
      cudaDriverEntryPointSymbolNotFound;
  cudaGetDriverEntryPoint(
      funcName,
      reinterpret_cast<void**>(&func),
      cudaEnableDefault,
      &driverStatus);
#elif CUDART_VERSION >= 11030
  cudaGetDriverEntryPoint(
      funcName, reinterpret_cast<void**>(&func), cudaEnableDefault);
#endif
  return func;
}

/**
 * Gets the error string for a CUresult error code.
 * Uses dynamic loading to avoid link-time dependency on libcuda.so.
 *
 * Returns "Unknown error" if the driver function cannot be loaded.
 */
inline const char* getCuErrorString(CUresult err) {
#if CUDART_VERSION >= 11030
  static auto pfn_cuGetErrorString =
      loadDriverFunction<PFN_cuGetErrorString_v6000>("cuGetErrorString");
  if (pfn_cuGetErrorString != nullptr) {
    const char* errStr = nullptr;
    pfn_cuGetErrorString(err, &errStr);
    if (errStr != nullptr) {
      return errStr;
    }
  }
#endif
  return "Unknown error";
}

/**
 * Wrapper for cuMemGetAddressRange that uses dynamic loading.
 * This avoids the need to link against libcuda.so at build time.
 *
 * Returns CUDA_ERROR_NOT_INITIALIZED if the driver function cannot be loaded.
 */
inline CUresult cuMemGetAddressRangeDynamic(
    CUdeviceptr* pbase,
    size_t* psize,
    CUdeviceptr dptr) {
#if CUDART_VERSION >= 11030
  static auto pfn_cuMemGetAddressRange =
      loadDriverFunction<PFN_cuMemGetAddressRange_v3020>(
          "cuMemGetAddressRange");
  if (pfn_cuMemGetAddressRange != nullptr) {
    return pfn_cuMemGetAddressRange(pbase, psize, dptr);
  }
#endif
  return CUDA_ERROR_NOT_INITIALIZED;
}

} // namespace comms::utils::cuda

/**
 * Checks a CUDA driver API command (CUresult) and throws std::runtime_error on
 * failure.
 *
 * This macro is for CUDA driver API calls that return CUresult.
 * For CUDA runtime API calls (cudaError_t), use FB_CUDACHECKTHROW instead.
 *
 * Note: This implementation uses dynamic loading via cudaGetDriverEntryPoint
 * to avoid requiring libcuda.so to be present at link time. This makes it
 * suitable for use in environments where only the CUDA runtime is available
 * (e.g., some CI build environments).
 *
 * Usage:
 *   FB_CUCHECKTHROW(comms::utils::cuda::cuMemGetAddressRangeDynamic(&pbase,
 * &size, devPtr));
 */
#define FB_CUCHECKTHROW(cmd)                                                   \
  do {                                                                         \
    CUresult err = cmd;                                                        \
    if (err != CUDA_SUCCESS) {                                                 \
      const char* errStr = ::comms::utils::cuda::getCuErrorString(err);        \
      CLOGF(ERR, "Cuda failure {} '{}'", static_cast<int>(err), errStr);       \
      throw std::runtime_error(                                                \
          fmt::format("Cuda failure {} '{}'", static_cast<int>(err), errStr)); \
    }                                                                          \
  } while (false)
