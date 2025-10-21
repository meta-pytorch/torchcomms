// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <folly/ScopeGuard.h>

#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#if CUDART_VERSION >= 11030

#if CUDART_VERSION >= 12000
#define FB_LOAD_SYM(symbol, ignore)                                          \
  do {                                                                       \
    cudaDriverEntryPointQueryResult driverStatus =                           \
        cudaDriverEntryPointSymbolNotFound;                                  \
    res = cudaGetDriverEntryPoint(                                           \
        #symbol, (void**)(&pfn_##symbol), cudaEnableDefault, &driverStatus); \
    if (res != cudaSuccess || driverStatus != cudaDriverEntryPointSuccess) { \
      if (!ignore) {                                                         \
        CLOGF(                                                               \
            WARN,                                                            \
            "Retrieve {} failed with {} status {}",                          \
            #symbol,                                                         \
            static_cast<int>(res),                                           \
            static_cast<int>(driverStatus));                                 \
        return commSystemError;                                              \
      }                                                                      \
    }                                                                        \
  } while (0)
#else
#define FB_LOAD_SYM(symbol, ignore)                           \
  do {                                                        \
    res = cudaGetDriverEntryPoint(                            \
        #symbol, (void**)(&pfn_##symbol), cudaEnableDefault); \
    if (res != cudaSuccess) {                                 \
      if (!ignore) {                                          \
        CLOGF(                                                \
            WARN,                                             \
            "Retrieve {} failed with {}",                     \
            #symbol,                                          \
            static_cast<int>(res));                           \
        return commSystemError;                               \
      }                                                       \
    }                                                         \
  } while (0)
#endif
#endif

namespace ctran::utils {

#define DECLARE_CUDA_PFN(symbol, version) \
  PFN_##symbol##_v##version pfn_##symbol = nullptr

#if CUDART_VERSION >= 11030
/* CUDA Driver functions loaded with cuGetProcAddress for versioning */
DECLARE_CUDA_PFN(cuDeviceGet, 2000);
DECLARE_CUDA_PFN(cuDeviceGetAttribute, 2000);
DECLARE_CUDA_PFN(cuGetErrorString, 6000);
DECLARE_CUDA_PFN(cuGetErrorName, 6000);
DECLARE_CUDA_PFN(cuMemGetAddressRange, 3020);
DECLARE_CUDA_PFN(cuLaunchKernel, 4000);
DECLARE_CUDA_PFN(cuMemHostGetDevicePointer, 3020);
#if CUDA_VERSION >= 11080
DECLARE_CUDA_PFN(cuLaunchKernelEx, 11060);
#endif
DECLARE_CUDA_PFN(cuCtxCreate, 11040);
DECLARE_CUDA_PFN(cuCtxDestroy, 4000);
DECLARE_CUDA_PFN(cuCtxGetCurrent, 4000);
DECLARE_CUDA_PFN(cuCtxSetCurrent, 4000);
DECLARE_CUDA_PFN(cuCtxGetDevice, 2000);
DECLARE_CUDA_PFN(cuMemAddressReserve, 10020);
DECLARE_CUDA_PFN(cuMemAddressFree, 10020);
DECLARE_CUDA_PFN(cuMemCreate, 10020);
DECLARE_CUDA_PFN(cuMemGetAllocationGranularity, 10020);
DECLARE_CUDA_PFN(cuMemExportToShareableHandle, 10020);
DECLARE_CUDA_PFN(cuMemImportFromShareableHandle, 10020);
DECLARE_CUDA_PFN(cuMemMap, 10020);
DECLARE_CUDA_PFN(cuMemRelease, 10020);
DECLARE_CUDA_PFN(cuMemRetainAllocationHandle, 11000);
DECLARE_CUDA_PFN(cuMemSetAccess, 10020);
DECLARE_CUDA_PFN(cuMemGetAccess, 10020);
DECLARE_CUDA_PFN(cuMemUnmap, 10020);
DECLARE_CUDA_PFN(cuMemGetAllocationPropertiesFromHandle, 10020);
DECLARE_CUDA_PFN(cuPointerGetAttribute, 4000);
#if CUDA_VERSION >= 11070
DECLARE_CUDA_PFN(cuMemGetHandleForAddressRange, 11070); // DMA-BUF support
#endif
#if CUDA_VERSION >= 12010
/* NVSwitch Multicast support */
DECLARE_CUDA_PFN(cuMulticastAddDevice, 12010);
DECLARE_CUDA_PFN(cuMulticastBindMem, 12010);
DECLARE_CUDA_PFN(cuMulticastBindAddr, 12010);
DECLARE_CUDA_PFN(cuMulticastCreate, 12010);
DECLARE_CUDA_PFN(cuMulticastGetGranularity, 12010);
DECLARE_CUDA_PFN(cuMulticastUnbind, 12010);
#endif
#endif

bool getCuMemSysSupported() {
  static bool cuMemSupported = isCuMemSupported();
  return cuMemSupported;
}

bool isCuMemSupported() {
#if defined(__HIP_PLATFORM_AMD__)
  // cuMem API doesn't work properly on AMD yet.
  // see details in https://ontrack.amd.com/browse/FBA-633
  // TODO: update it after AMD confirms cuMem support
  return false;
#elif CUDART_VERSION < 11030
  return false;
#else
  CUdevice currentDev;
  int cudaDev;
  int cudaDriverVersion;
  int flag = 0;
  FB_CUDACHECK_RETURN(cudaDriverGetVersion(&cudaDriverVersion), false);
  if (cudaDriverVersion < 12000) {
    return false; // Need CUDA_VISIBLE_DEVICES support
  }
  FB_CUDACHECK_RETURN(cudaGetDevice(&cudaDev), false);
  if (FB_CUPFN(cuMemCreate) == nullptr) {
    return false;
  }
  FB_CUCHECK_RETURN(cuDeviceGet(&currentDev, cudaDev), false);
  // Query device to see if CUMEM VMM support is available
  FB_CUCHECK_RETURN(
      cuDeviceGetAttribute(
          &flag,
          CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
          currentDev),
      false);
  if (!flag) {
    return false;
  }
  return true;
#endif
}

inline bool isCuMemHostSupported(int driverVersion) {
#if CUDART_VERSION < 12020
  if (NCCL_CUMEM_HOST_ENABLE == 1) {
    CLOGF(
        WARN,
        "NCCL_CUMEM_HOST_ENABLE is set to 1, but CUDA runtime library is too old, required 12.2 or later");
  }
  return false;
#else
  if (NCCL_CUMEM_HOST_ENABLE == -1) {
    if (driverVersion < 12020) {
      return false;
    } else {
      // Automatically enable NCCL_CUMEM_HOST_ENABLE if the  driver supports it,
      // i.e., requires 12.6 or later, to be consistent with baseline NCCL
      // (https://fburl.com/code/eld8g6at)
      return (driverVersion >= 12060) ? true : false;
    }
  }
  return (NCCL_CUMEM_HOST_ENABLE == 1);
#endif
}

static commResult_t cudaPfnFuncLoader(void) {
#if defined(__HIP_PLATFORM_AMD__)
  // AMD doesn't use dynamic loading for cuda driver APIs
  return commSuccess;
#elif CUDART_VERSION < 11030
  return commSystemError;
#else
  cudaError_t res;

  FB_LOAD_SYM(cuGetErrorString, 0);
  FB_LOAD_SYM(cuGetErrorName, 0);
  FB_LOAD_SYM(cuDeviceGet, 0);
  FB_LOAD_SYM(cuDeviceGetAttribute, 0);
  FB_LOAD_SYM(cuMemGetAddressRange, 1);
  FB_LOAD_SYM(cuCtxCreate, 1);
  FB_LOAD_SYM(cuCtxDestroy, 1);
  FB_LOAD_SYM(cuCtxGetCurrent, 1);
  FB_LOAD_SYM(cuCtxSetCurrent, 1);
  FB_LOAD_SYM(cuCtxGetDevice, 1);
  FB_LOAD_SYM(cuLaunchKernel, 1);
  FB_LOAD_SYM(cuMemHostGetDevicePointer, 1);
#if CUDA_VERSION >= 11080
  FB_LOAD_SYM(cuLaunchKernelEx, 1);
#endif
  /* cuMem API support */
  FB_LOAD_SYM(cuMemAddressReserve, 1);
  FB_LOAD_SYM(cuMemAddressFree, 1);
  FB_LOAD_SYM(cuMemCreate, 1);
  FB_LOAD_SYM(cuMemGetAllocationGranularity, 1);
  FB_LOAD_SYM(cuMemExportToShareableHandle, 1);
  FB_LOAD_SYM(cuMemImportFromShareableHandle, 1);
  FB_LOAD_SYM(cuMemMap, 1);
  FB_LOAD_SYM(cuMemRelease, 1);
  FB_LOAD_SYM(cuMemRetainAllocationHandle, 1);
  FB_LOAD_SYM(cuMemSetAccess, 1);
  FB_LOAD_SYM(cuMemGetAccess, 1);
  FB_LOAD_SYM(cuMemUnmap, 1);
  FB_LOAD_SYM(cuMemGetAllocationPropertiesFromHandle, 1);
  /* MemAlloc/Free */
  FB_LOAD_SYM(cuPointerGetAttribute, 1);
#if CUDA_VERSION >= 11070
  FB_LOAD_SYM(cuMemGetHandleForAddressRange, 1); // DMA-BUF support
#endif
#if CUDA_VERSION >= 12010
  /* NVSwitch Multicast support */
  FB_LOAD_SYM(cuMulticastAddDevice, 1);
  FB_LOAD_SYM(cuMulticastBindMem, 1);
  FB_LOAD_SYM(cuMulticastBindAddr, 1);
  FB_LOAD_SYM(cuMulticastCreate, 1);
  FB_LOAD_SYM(cuMulticastGetGranularity, 1);
  FB_LOAD_SYM(cuMulticastUnbind, 1);
#endif
  return commSuccess;
#endif
}

static std::once_flag commCudaLibraryInitFlag;
static commResult_t commCudaLibraryInitResult;

static commResult_t initCommCudaLibraryOnce_() {
  int cudaDev;
  int driverVersion;
  commCudaLibraryInitResult = commSystemError;

  FB_CUDACHECK_RETURN(cudaGetDevice(&cudaDev), commCudaLibraryInitResult);
  FB_CUDACHECK_RETURN(
      cudaDriverGetVersion(&driverVersion), commCudaLibraryInitResult);
  CLOGF_SUBSYS(INFO, INIT, "cudaDriverVersion {}", driverVersion);
  if (cudaPfnFuncLoader() != commSuccess) {
    CLOGF(WARN, "CUDA some PFN functions not found in the library");
    return commCudaLibraryInitResult;
  }
  auto cuMemSupported = isCuMemSupported();
  // To use cuMem* for host memory allocation, we need to create context on
  // each visible device. This is a workaround needed in CUDA 12.2 and
  // CUDA 12.3 which is fixed in 12.4.
  if (cuMemSupported && isCuMemHostSupported(driverVersion) &&
      12020 <= driverVersion && driverVersion <= 12030) {
    int deviceCnt, saveDevice;
    FB_CUDACHECK_RETURN(cudaGetDevice(&saveDevice), commCudaLibraryInitResult);
    FB_CUDACHECK_RETURN(
        cudaGetDeviceCount(&deviceCnt), commCudaLibraryInitResult);
    for (int i = 0; i < deviceCnt; ++i) {
      FB_CUDACHECK_RETURN(cudaSetDevice(i), commCudaLibraryInitResult);
      FB_CUDACHECK_RETURN(cudaFree(nullptr), commCudaLibraryInitResult);
    }
    FB_CUDACHECK_RETURN(cudaSetDevice(saveDevice), commCudaLibraryInitResult);
  }
  commCudaLibraryInitResult = commSuccess;

  return commCudaLibraryInitResult;
}

commResult_t commCudaLibraryInit() {
  std::call_once(commCudaLibraryInitFlag, initCommCudaLibraryOnce_);
  return commCudaLibraryInitResult;
}

bool isCommCudaLibraryInited() {
  return commCudaLibraryInitResult == commSuccess;
}

commResult_t dmaBufDriverSupport(int cudaDev) {
#if CUDA_VERSION >= 11070
  int flag = 0;
  CUdevice dev;
  int cudaDriverVersion;
  FB_CUDACHECK(cudaDriverGetVersion(&cudaDriverVersion));
  if (FB_CUPFN(cuDeviceGet) == NULL || cudaDriverVersion < 11070) {
    return commInternalError;
  }
  FB_CUCHECK(cuDeviceGet(&dev, cudaDev));
  // Query device to see if DMA-BUF support is available
  (void)FB_CUPFN(
      cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev));
  if (flag == 0) {
    return commInternalError;
  }
  CLOGF_SUBSYS(INFO, INIT, "DMA-BUF is available on GPU device {}", cudaDev);
  return commSuccess;
#endif
  return commInternalError;
}

} // namespace ctran::utils
