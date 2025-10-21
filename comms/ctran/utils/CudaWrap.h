// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <fmt/format.h>

#include "comms/ctran/utils/ErrorStackTraceUtil.h"
#include "comms/utils/logger/LogUtils.h"

#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>

#endif

#if defined(__HIP_PLATFORM_AMD__)
#include "comms/ctran/utils/HipGdrCheck.h"
#endif

template <>
struct fmt::formatter<cudaError_t> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(cudaError_t status, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};

#if defined(__HIP_PLATFORM_AMD__)
// In HIP, cudaError_t and CUresult are the same type (hipError_t), so we need
// to avoid redifinitions.
#else
template <>
struct fmt::formatter<CUresult> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(CUresult cuResult, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(cuResult), ctx);
  }
};
#endif

template <>
struct fmt::formatter<CUmemAllocationHandleType> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(CUmemAllocationHandleType status, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};

namespace ctran::utils {
#if defined(__HIP_PLATFORM_AMD__)
#define FB_CUPFN(symbol) symbol

// Check CUDA PFN driver calls
#define FB_CUCHECK(cmd)                                                  \
  do {                                                                   \
    CUresult err = cmd;                                                  \
    if (err != CUDA_SUCCESS) {                                           \
      const char* errStr;                                                \
      (void)cuGetErrorString(err, &errStr);                              \
      CLOGF(ERR, "Cuda failure {} '{}'", static_cast<int>(err), errStr); \
      ErrorStackTraceUtil::logErrorMessage(                              \
          "Cuda Error: " + std::string(errStr));                         \
      return commUnhandledCudaError;                                     \
    }                                                                    \
  } while (false)

#define FB_CUCHECK_RETURN(cmd, ret)                                      \
  do {                                                                   \
    CUresult err = cmd;                                                  \
    if (err != CUDA_SUCCESS) {                                           \
      const char* errStr;                                                \
      (void)cuGetErrorString(err, &errStr);                              \
      CLOGF(ERR, "Cuda failure {} '{}'", static_cast<int>(err), errStr); \
      ErrorStackTraceUtil::logErrorMessage(                              \
          "Cuda Error: " + std::string(errStr));                         \
      return ret;                                                        \
    }                                                                    \
  } while (false)

#define FB_CUCHECK_GOTO(cmd, ret, label)                                 \
  do {                                                                   \
    CUresult err = cmd;                                                  \
    if (err != CUDA_SUCCESS) {                                           \
      const char* errStr;                                                \
      cuGetErrorString(err, &errStr);                                    \
      CLOGF(ERR, "Cuda failure {} '{}'", static_cast<int>(err), errStr); \
      ErrorStackTraceUtil::logErrorMessage(                              \
          "Cuda Error: " + std::string(errStr));                         \
      ret = commUnhandledCudaError;                                      \
      goto label;                                                        \
    }                                                                    \
  } while (false)

#define FB_CUCHECKTHROW(cmd)                                                   \
  do {                                                                         \
    CUresult err = cmd;                                                        \
    if (err != CUDA_SUCCESS) {                                                 \
      const char* errStr;                                                      \
      (void)cuGetErrorString(err, &errStr);                                    \
      CLOGF(ERR, "Cuda failure {} '{}'", static_cast<int>(err), errStr);       \
      ErrorStackTraceUtil::logErrorMessage(                                    \
          "Cuda Error: " + std::string(errStr));                               \
      throw std::runtime_error(                                                \
          fmt::format("Cuda failure {} '{}'", static_cast<int>(err), errStr)); \
    }                                                                          \
  } while (false)

#define FB_CUCHECKRES(res)                                     \
  do {                                                         \
    if (res != CUDA_SUCCESS) {                                 \
      const char* errStr;                                      \
      (void)cuGetErrorString(res, &errStr);                    \
      CLOGF(ERR, "Cuda failure {} '{}'", res, errStr);         \
      return ErrorStackTraceUtil::log(commUnhandledCudaError); \
    }                                                          \
  } while (false)

// Report failure but clear error and continue
#define FB_CUCHECKIGNORE(cmd)               \
  do {                                      \
    CUresult err = cmd;                     \
    if (err != CUDA_SUCCESS) {              \
      const char* errStr;                   \
      (void)cuGetErrorString(err, &errStr); \
      CLOGF(                                \
          WARN,                             \
          "{}:{} Cuda failure {} '{}'",     \
          __FILE__,                         \
          __LINE__,                         \
          static_cast<int>(err),            \
          errStr);                          \
    }                                       \
  } while (false)

#define FB_CUCHECKTHREAD(cmd, args)       \
  do {                                    \
    CUresult err = cmd;                   \
    if (err != CUDA_SUCCESS) {            \
      CLOGF(                              \
          ERR,                            \
          "{}:{} -> {} [Async thread]",   \
          __FILE__,                       \
          __LINE__,                       \
          static_cast<int>(err));         \
      args->ret = commUnhandledCudaError; \
      return args;                        \
    }                                     \
  } while (0)

#else

#define FB_CUPFN(symbol) ::ctran::utils::pfn_##symbol

// Check CUDA PFN driver calls
#define FB_CUCHECK(cmd)                                                  \
  do {                                                                   \
    CUresult err = ::ctran::utils::pfn_##cmd;                            \
    if (err != CUDA_SUCCESS) {                                           \
      const char* errStr;                                                \
      (void)::ctran::utils::pfn_cuGetErrorString(err, &errStr);          \
      CLOGF(ERR, "Cuda failure {} '{}'", static_cast<int>(err), errStr); \
      ErrorStackTraceUtil::logErrorMessage(                              \
          "Cuda Error: " + std::string(errStr));                         \
      return commUnhandledCudaError;                                     \
    }                                                                    \
  } while (false)

#define FB_CUCHECKTHROW(cmd)                                                   \
  do {                                                                         \
    CUresult err = ::ctran::utils::pfn_##cmd;                                  \
    if (err != CUDA_SUCCESS) {                                                 \
      const char* errStr;                                                      \
      (void)::ctran::utils::pfn_cuGetErrorString(err, &errStr);                \
      CLOGF(ERR, "Cuda failure {} '{}'", static_cast<int>(err), errStr);       \
      ErrorStackTraceUtil::logErrorMessage(                                    \
          "Cuda Error: " + std::string(errStr));                               \
      throw std::runtime_error(                                                \
          fmt::format("Cuda failure {} '{}'", static_cast<int>(err), errStr)); \
    }                                                                          \
  } while (false)

#define FB_CUCHECK_RETURN(cmd, ret)                                      \
  do {                                                                   \
    CUresult err = ::ctran::utils::pfn_##cmd;                            \
    if (err != CUDA_SUCCESS) {                                           \
      const char* errStr;                                                \
      (void)::ctran::utils::pfn_cuGetErrorString(err, &errStr);          \
      CLOGF(ERR, "Cuda failure {} '{}'", static_cast<int>(err), errStr); \
      ErrorStackTraceUtil::logErrorMessage(                              \
          "Cuda Error: " + std::string(errStr));                         \
      return ret;                                                        \
    }                                                                    \
  } while (false)

#define FB_CUCHECK_GOTO(cmd, ret, label)                                 \
  do {                                                                   \
    CUresult err = ::ctran::utils::pfn_##cmd;                            \
    if (err != CUDA_SUCCESS) {                                           \
      const char* errStr;                                                \
      (void)::ctran::utils::pfn_cuGetErrorString(err, &errStr);          \
      CLOGF(ERR, "Cuda failure {} '{}'", static_cast<int>(err), errStr); \
      ErrorStackTraceUtil::logErrorMessage(                              \
          "Cuda Error: " + std::string(errStr));                         \
      ret = commUnhandledCudaError;                                      \
      goto label;                                                        \
    }                                                                    \
  } while (false)

#define FB_CUCHECKRES(res)                                      \
  do {                                                          \
    if (res != CUDA_SUCCESS) {                                  \
      const char* errStr;                                       \
      (void)::ctran::utils::pfn_cuGetErrorString(res, &errStr); \
      CLOGF(ERR, "Cuda failure {} '{}'", res, errStr);          \
      return ErrorStackTraceUtil::log(commUnhandledCudaError);  \
    }                                                           \
  } while (false)

// Report failure but clear error and continue
#define FB_CUCHECKIGNORE(cmd)                                   \
  do {                                                          \
    CUresult err = ::ctran::utils::pfn_##cmd;                   \
    if (err != CUDA_SUCCESS) {                                  \
      const char* errStr;                                       \
      (void)::ctran::utils::pfn_cuGetErrorString(err, &errStr); \
      CLOGF(                                                    \
          WARN,                                                 \
          "{}:{} Cuda failure {} '{}'",                         \
          __FILE__,                                             \
          __LINE__,                                             \
          static_cast<int>(err),                                \
          errStr);                                              \
    }                                                           \
  } while (false)

#define FB_CUCHECKTHREAD(cmd, args)           \
  do {                                        \
    CUresult err = ::ctran::utils::pfn_##cmd; \
    if (err != CUDA_SUCCESS) {                \
      CLOGF(                                  \
          ERR,                                \
          "{}:{} -> {} [Async thread]",       \
          __FILE__,                           \
          __LINE__,                           \
          static_cast<int>(err));             \
      args->ret = commUnhandledCudaError;     \
      return args;                            \
    }                                         \
  } while (0)

#define FB_DECLARE_CUDA_PFN_EXTERN(symbol, version) \
  extern PFN_##symbol##_v##version pfn_##symbol

#if CUDART_VERSION >= 11030
/* CUDA Driver functions loaded with cuGetProcAddress for versioning */
FB_DECLARE_CUDA_PFN_EXTERN(cuDeviceGet, 2000);
FB_DECLARE_CUDA_PFN_EXTERN(cuDeviceGetAttribute, 2000);
FB_DECLARE_CUDA_PFN_EXTERN(cuGetErrorString, 6000);
FB_DECLARE_CUDA_PFN_EXTERN(cuGetErrorName, 6000);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemGetAddressRange, 3020);
FB_DECLARE_CUDA_PFN_EXTERN(cuLaunchKernel, 4000);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemHostGetDevicePointer, 3020);
#if CUDA_VERSION >= 11080
FB_DECLARE_CUDA_PFN_EXTERN(cuLaunchKernelEx, 11060);
#endif
FB_DECLARE_CUDA_PFN_EXTERN(cuCtxCreate, 11040);
FB_DECLARE_CUDA_PFN_EXTERN(cuCtxDestroy, 4000);
FB_DECLARE_CUDA_PFN_EXTERN(cuCtxGetCurrent, 4000);
FB_DECLARE_CUDA_PFN_EXTERN(cuCtxSetCurrent, 4000);
FB_DECLARE_CUDA_PFN_EXTERN(cuCtxGetDevice, 2000);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemAddressReserve, 10020);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemAddressFree, 10020);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemCreate, 10020);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemGetAllocationGranularity, 10020);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemExportToShareableHandle, 10020);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemImportFromShareableHandle, 10020);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemMap, 10020);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemRelease, 10020);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemRetainAllocationHandle, 11000);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemSetAccess, 10020);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemGetAccess, 10020);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemUnmap, 10020);
FB_DECLARE_CUDA_PFN_EXTERN(cuMemGetAllocationPropertiesFromHandle, 10020);
FB_DECLARE_CUDA_PFN_EXTERN(cuPointerGetAttribute, 4000);
#if CUDA_VERSION >= 11070
FB_DECLARE_CUDA_PFN_EXTERN(cuMemGetHandleForAddressRange, 11070);
#endif
#if CUDA_VERSION >= 12010
/* NVSwitch Multicast support */
FB_DECLARE_CUDA_PFN_EXTERN(cuMulticastAddDevice, 12010);
FB_DECLARE_CUDA_PFN_EXTERN(cuMulticastBindMem, 12010);
FB_DECLARE_CUDA_PFN_EXTERN(cuMulticastBindAddr, 12010);
FB_DECLARE_CUDA_PFN_EXTERN(cuMulticastCreate, 12010);
FB_DECLARE_CUDA_PFN_EXTERN(cuMulticastGetGranularity, 12010);
FB_DECLARE_CUDA_PFN_EXTERN(cuMulticastUnbind, 12010);
#endif
#endif

#endif

inline void setCuMemHandleTypeForProp(
    CUmemAllocationProp& prop,
    const CUmemAllocationHandleType handleType) {
#if defined(__HIP_PLATFORM_AMD__)
  prop.requestedHandleType = handleType;
#else
  prop.requestedHandleTypes = handleType;
#endif
}

inline CUmemAllocationHandleType getCuMemHandleTypeFromProp(
    const CUmemAllocationProp& prop) {
#if defined(__HIP_PLATFORM_AMD__)
  return prop.requestedHandleType;
#else
  return prop.requestedHandleTypes;
#endif
}

inline bool gpuDirectRdmaWithCudaVmmSupported(
    const CUdevice& cuDev,
    const int cudaDev) {
#if defined(__HIP_PLATFORM_AMD__)
  // TODO: It checks and returns GDR functionality of AMD GPUs as
  // "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED" is not
  // currently supported by HIP.
  // Revisit here after deciding cuMemSys support of CTRAN for HIP.
  return getGpuDirectRDMASupported();
#else
  int flag = 0;
  FB_CUCHECK(cuDeviceGetAttribute(
      &flag,
      CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
      cuDev));
  return flag != 0;
#endif
}

// From definition of CUdeviceptr in cuda.h
#if defined(_WIN64) || defined(__LP64__)
typedef unsigned long long __CUdeviceptr_t;
#else
typedef unsigned int __CUdeviceptr_t;
#endif

inline CUdeviceptr addDevicePtr(const CUdeviceptr base, size_t offset) {
#if defined(__HIP_PLATFORM_AMD__)
  return reinterpret_cast<CUdeviceptr>(
      reinterpret_cast<__CUdeviceptr_t>(base) + offset);
#else
  return base + offset;
#endif
}

inline CUdeviceptr subDevicePtr(const CUdeviceptr a, const void* b) {
#if defined(__HIP_PLATFORM_AMD__)
  return reinterpret_cast<CUdeviceptr>(
      reinterpret_cast<__CUdeviceptr_t>(a) -
      reinterpret_cast<__CUdeviceptr_t>(b));
#else
  return a - (CUdeviceptr)b;
#endif
}

#if defined(__HIP_PLATFORM_AMD__)
inline void* toFormattableHandle(const CUmemGenericAllocationHandle handle) {
  return reinterpret_cast<void*>(handle);
}
#else
inline CUmemGenericAllocationHandle toFormattableHandle(
    const CUmemGenericAllocationHandle handle) {
  return handle;
}

#endif

inline int getCuMemDmaBufFd(const void* buf, const size_t len) {
#if defined(__HIP_PLATFORM_AMD__)
  // TODO: Implement this feature for HIP with ROCm 7.0.
  // `cuMemGetHandleForAddressRange` will be supported in ROCm 7.0
  // (https://ontrack.amd.com/browse/FBA-621).
  return -1;
#else
  int dmabufFd = -1;
  FB_CUPFN(cuMemGetHandleForAddressRange(
      &dmabufFd,
      reinterpret_cast<CUdeviceptr>(buf),
      len,
      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
      0));
  return dmabufFd;
#endif
}

#if defined(__HIP_PLATFORM_AMD__)
// Copied from CUDA "driver_types.h".
#ifdef _WIN32
#define CUDART_CB __stdcall
#else
#define CUDART_CB
#endif
#endif

bool getCuMemSysSupported();

bool isCuMemSupported();

commResult_t commCudaLibraryInit();

bool isCommCudaLibraryInited();

commResult_t dmaBufDriverSupport(int cudaDev);

} // namespace ctran::utils
