// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>

#include "comms/uniflow/Result.h"

#if defined(__HIP_PLATFORM_AMD__)
// hipify-perl does not map this newer VMM dma-buf handle type; alias the CUDA
// spelling (which survives hipification untranslated) to the HIP type so the
// hipified interface still names a valid type on AMD. The dma-buf export path
// that uses it is implemented for AMD in a later diff.
using CUmemRangeHandleType = hipMemRangeHandleType;
// Same hipify gap for the dma-buf-fd enumerator used by callers of
// cuMemGetHandleForAddressRange.
inline constexpr hipMemRangeHandleType CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD =
    hipMemRangeHandleTypeDmaBufFd;
// hipify-perl also misses the stream write-value flag enumerator used by
// streamWriteValue64. HIP has no named constant for it; the CUDA default value
// is 0x0 (no memory barrier), which is also HIP's default, so alias to 0.
inline constexpr unsigned int CU_STREAM_WRITE_VALUE_DEFAULT = 0u;
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

  virtual Status cuMemGetHandleForAddressRange(
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
