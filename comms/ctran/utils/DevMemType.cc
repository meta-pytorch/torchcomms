// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/DevMemType.h"

commResult_t
getDevMemType(const void* addr, const int cudaDev, DevMemType& memType) {
  if (addr == nullptr) {
    return commInvalidUsage;
  }

  if (cudaDev < 0) {
    return commInvalidUsage;
  }

  cudaPointerAttributes attr;
  FB_CUDACHECK(cudaPointerGetAttributes(&attr, addr));
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  if (attr.type == hipMemoryTypeUnregistered) {
#else
  if (attr.type == cudaMemoryTypeUnregistered) {
#endif
    memType = DevMemType::kHostUnregistered;
    return commSuccess;
  } else if (attr.type == cudaMemoryTypeHost) {
    memType = DevMemType::kHostPinned;
    return commSuccess;
  } else if (attr.type == cudaMemoryTypeManaged) {
    memType = DevMemType::kManaged;
    return commSuccess;
  }

  // Check if the memory is allocated by cuMem APIs.
  /*
    In CUDA, for cudaMalloc-ed buffers,
    - If the CUDA driver version supports CUDA runtime version 12.8,
      cuMemGetAccess returns CUDA_SUCCESS and cuMemRetainAllocationHandle
      returns CUDA_ERROR_INVALID_VALUE.
    - If the CUDA driver version only supports CUDA runtime version less
    than 12.8, cuMemGetAccess returns CUDA_ERROR_INVALID_VALUE and
    cuMemRetainAllocationHandle also reutrns CUDA_ERROR_INVALID_VALUE.

    In ROCm, for cudaMalloc-ed buffers,
    - cuMemRetainAllocationHandle results in segfault.
      TODO: follow up with AMD here https://ontrack.amd.com/browse/FBA-649
    - cuMemGetAccess returns CUDA_ERROR_INVALID_VALUE

    Therefore, we use cuMemRetainAllocationHandle in CUDA and use cuMemGetAccess
    in ROCm to determine whether the memory is cudaMalloc-ed.
  */

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__) || \
    defined(__HIP_PLATFORM_HCC__)
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;

  unsigned long long flags = 0;
  CUresult ret =
      FB_CUPFN(cuMemGetAccess)(&flags, &accessDesc.location, (CUdeviceptr)addr);

  if (ret == CUDA_ERROR_INVALID_VALUE) {
    // On ROCm, cuMemGetAccess returns CUDA_ERROR_INVALID_VALUE for cudaMalloc
    // memory. This may also leave a benign error in the HIP runtime error
    // queue. Clear it to prevent propagation to user code.
    (void)cudaGetLastError();
    memType = DevMemType::kCudaMalloc;
    return commSuccess;
  } else if (ret != CUDA_SUCCESS) {
    // Other unexpected error
    FB_CUCHECKRES(ret);
  }
  memType = DevMemType::kCumem;
  return commSuccess;
#else
  CUmemGenericAllocationHandle handle;
  CUresult ret =
      FB_CUPFN(cuMemRetainAllocationHandle)(&handle, const_cast<void*>(addr));
  if (ret == CUDA_ERROR_INVALID_VALUE) {
    memType = DevMemType::kCudaMalloc;
    return commSuccess;
  } else if (ret != CUDA_SUCCESS) {
    FB_CUCHECKRES(ret);
  }
  // cuMemRetainAllocationHandle needs to be paired with cuMemRelease
  FB_CUCHECK(cuMemRelease(handle));
  memType = DevMemType::kCumem;
  return commSuccess;

#endif
}
