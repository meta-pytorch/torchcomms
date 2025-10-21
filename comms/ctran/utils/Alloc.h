// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <folly/ScopeGuard.h>

#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/DevUtils.cuh"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/alloc.h"

namespace ctran::utils {

CUmemAllocationHandleType getCuMemAllocHandleType();

commResult_t commCuMemAlloc(
    void** ptr,
    CUmemGenericAllocationHandle* handlep,
    CUmemAllocationHandleType type,
    size_t size,
    const CommLogData* logMetaData,
    const char* callsite);

commResult_t commCuMemFree(void* ptr, const CommLogData* logMetaData = nullptr);

bool isCuMemFabricHandleSupported();

template <typename T>
commResult_t commCudaMallocDebug(
    T** ptr,
    size_t nelem,
    const CommLogData* logMetaData,
    const char* callsite,
    const char* filefunc,
    int line) {
  commResult_t result = commSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  FB_CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    if (getCuMemSysSupported()) {
      FB_COMMCHECKGOTO(
          commCuMemAlloc(
              (void**)ptr,
              NULL,
              getCuMemAllocHandleType(),
              nelem * sizeof(T),
              logMetaData,
              callsite),
          result,
          finish);
    } else {
      FB_CUDACHECKGOTO(cudaMalloc(ptr, nelem * sizeof(T)), result, finish);
    }
  }
finish:
  FB_CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) {
    CLOGF(WARN, "Failed to CUDA malloc {} bytes", nelem * sizeof(T));
  }
  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "{}:{} Cuda Alloc Size {} pointer {}",
      filefunc,
      line,
      nelem * sizeof(T),
      (void*)*ptr);
  if (!getCuMemSysSupported()) {
    logMemoryEvent(
        logMetaData ? *logMetaData : CommLogData{},
        callsite,
        "commCudaMalloc",
        reinterpret_cast<uintptr_t>(*ptr),
        nelem * sizeof(T));
  }

  return result;
}

#define commCudaMalloc(...) commCudaMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
commResult_t commCudaFree(T* ptr, const CommLogData* logMetaData = nullptr) {
  commResult_t result = commSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CLOGF_TRACE(ALLOC, "Cuda Free pointer {}", (void*)ptr);

  FB_CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), result, finish);
  if (getCuMemSysSupported()) {
    FB_COMMCHECKGOTO(commCuMemFree((void*)ptr, logMetaData), result, finish);
  } else {
    FB_CUDACHECKGOTO(cudaFree(ptr), result, finish);
  }
finish:
  if (!getCuMemSysSupported()) {
    logMemoryEvent(
        logMetaData ? *logMetaData : CommLogData{},
        "",
        "commCudaFree",
        reinterpret_cast<uintptr_t>(ptr));
  }
  FB_CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T>
commResult_t commCudaCallocAsync(
    T** ptr,
    size_t nelem,
    cudaStream_t stream,
    const CommLogData* logMetaData,
    const char* callsite) {
  static_assert(
      !std::is_same<T, void>::value,
      "void pointers must be casted to valid types when calloc-ing for 'nelem' objects");
  commResult_t result = commSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  FB_CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    if (getCuMemSysSupported()) {
      FB_COMMCHECKGOTO(
          commCuMemAlloc(
              (void**)ptr,
              NULL,
              getCuMemAllocHandleType(),
              nelem * sizeof(T),
              logMetaData,
              callsite),
          result,
          finish);
    } else {
      FB_CUDACHECKGOTO(cudaMalloc(ptr, nelem * sizeof(T)), result, finish);
    }
    FB_CUDACHECKGOTO(
        cudaMemsetAsync(*ptr, 0, nelem * sizeof(T), stream), result, finish);
  }
finish:
  FB_CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0)
    CLOGF(WARN, "Failed to CUDA calloc async {} bytes", nelem * sizeof(T));
  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "{}:{} Cuda Alloc Size {} pointer {}",
      __FILE__,
      __LINE__,
      nelem * sizeof(T),
      (void*)*ptr);
  if (!getCuMemSysSupported()) {
    logMemoryEvent(
        logMetaData ? *logMetaData : CommLogData{},
        callsite,
        "commCudaCallocAsync",
        reinterpret_cast<uintptr_t>(*ptr),
        nelem * sizeof(T));
  }
  return result;
}

} // namespace ctran::utils
