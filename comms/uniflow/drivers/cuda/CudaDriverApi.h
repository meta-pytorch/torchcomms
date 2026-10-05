// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>

#include "comms/uniflow/Result.h"

#if defined(__HIP_PLATFORM_AMD__)
// Kept out of this (hipified) header; see CudaDriverApiHipCompat.h.
#include "comms/uniflow/drivers/cuda/CudaDriverApiHipCompat.h"
#endif

namespace uniflow {

/// Thin wrapper around CUDA Driver (cu*) APIs loaded via
/// cudaGetDriverEntryPoint.
class CudaDriverApi {
 public:
  virtual ~CudaDriverApi() = default;

  /// Load CUDA driver function pointers via cudaGetDriverEntryPoint.
  /// Called implicitly by every other method; safe to call multiple times.
  virtual Status init();

  // --- Device management ---

  virtual Status cuDeviceGet(CUdevice* device, int ordinal);

  virtual Status
  cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);

  // --- Error ---

  virtual Status cuGetErrorString(CUresult error, const char** pStr);

  virtual Status cuGetErrorName(CUresult error, const char** pStr);

  // --- cuMem VMM ---

  virtual Status cuMemCreate(
      CUmemGenericAllocationHandle* handle,
      size_t size,
      const CUmemAllocationProp* prop,
      unsigned long long flags);

  virtual Status cuMemRelease(CUmemGenericAllocationHandle handle);

  virtual Status cuMemAddressReserve(
      CUdeviceptr* ptr,
      size_t size,
      size_t alignment,
      CUdeviceptr addr,
      unsigned long long flags);

  virtual Status cuMemAddressFree(CUdeviceptr ptr, size_t size);

  virtual Status cuMemMap(
      CUdeviceptr ptr,
      size_t size,
      size_t offset,
      CUmemGenericAllocationHandle handle,
      unsigned long long flags);

  virtual Status cuMemUnmap(CUdeviceptr ptr, size_t size);

  virtual Status cuMemSetAccess(
      CUdeviceptr ptr,
      size_t size,
      const CUmemAccessDesc* desc,
      size_t count);

  virtual Status cuMemGetAllocationGranularity(
      size_t* granularity,
      const CUmemAllocationProp* prop,
      CUmemAllocationGranularity_flags option);

  virtual Status cuMemRetainAllocationHandle(
      CUmemGenericAllocationHandle* handle,
      void* addr);

  /// Get the base and size of the VMM allocation containing dptr.
  /// Requires an active CUDA context (call cudaSetDevice first).
  virtual Status
  cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);

  virtual Status cuMemExportToShareableHandle(
      void* shareableHandle,
      CUmemGenericAllocationHandle handle,
      CUmemAllocationHandleType handleType,
      unsigned long long flags);

  virtual Status cuMemImportFromShareableHandle(
      CUmemGenericAllocationHandle* handle,
      void* osHandle,
      CUmemAllocationHandleType shHandleType);
#if defined(__HIP_PLATFORM_AMD__)

  // On AMD, a failing call on this class does not leave a runtime error
  // pending unless one was already pending.

  /// Fails for memory that was not mapped through the VMM API, which makes it
  /// a safe VMM classifier: unlike cuMemRetainAllocationHandle it does not
  /// crash for hipMalloc memory.
  virtual Status cuMemGetAccess(
      unsigned long long* flags,
      const CUmemLocation* location,
      CUdeviceptr ptr);

  /// Import a POSIX fd produced by cuMemExportToShareableHandle. The caller
  /// keeps ownership of fd. Absorbs the ROCm change from passing the fd
  /// through a pointer to passing it by value.
  virtual Status importPosixFd(CUmemGenericAllocationHandle* handle, int fd);
#endif

  // Wraps cuMemGetHandleForAddressRange, which hipify-perl from ROCm 7.2 and
  // TheRock translates but torch hipify does not.
  virtual Status memGetHandleForAddressRange(
      void* handle,
      CUdeviceptr dptr,
      size_t size,
      CUmemRangeHandleType handleType,
      unsigned long long flags);

  // --- Stream memory ops ---

  virtual Status streamWriteValue64(
      CUstream stream,
      CUdeviceptr addr,
      uint64_t value,
      unsigned int flags);

  // --- supported APIs ---
  virtual Result<bool> isDmaBufSupported(int cudaDev);

  virtual Result<bool> isCuMemSupported();

  virtual Result<CUmemAllocationHandleType> getCuMemHandleType();
};

} // namespace uniflow
