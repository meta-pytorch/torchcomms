// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include <cstddef>
#include <cstdint>

#include "comms/mccl/utils/McclLogger.h"
#include "comms/utils/memtrace/GpuMemoryTracker.h"

namespace meta::comms::memtrace {

#ifdef __HIP_PLATFORM_AMD__
using McclCudaError = hipError_t;
#else
using McclCudaError = cudaError_t;
#endif

struct GpuMemoryAllocationMetadata {
  GpuMemoryResourceType resourceType{GpuMemoryResourceType::kUnclassified};
  uint64_t logicalBytes{0};

  uint64_t logicalBytesOrAccounted(std::size_t accountedBytes) const noexcept {
    return logicalBytes == 0 ? accountedBytes : logicalBytes;
  }
};

McclCudaError mcclCudaMalloc(
    void** ptr,
    std::size_t accountedBytes,
    const GpuMemoryAllocationMetadata& metadata = {}) noexcept;

template <typename T>
McclCudaError mcclCudaMalloc(
    T** ptr,
    std::size_t accountedBytes,
    const GpuMemoryAllocationMetadata& metadata = {}) noexcept {
  return mcclCudaMalloc(
      reinterpret_cast<void**>(ptr), accountedBytes, metadata);
}

McclCudaError mcclCudaMallocUncached(
    void** ptr,
    std::size_t accountedBytes,
    const GpuMemoryAllocationMetadata& metadata = {}) noexcept;

template <typename T>
McclCudaError mcclCudaMallocUncached(
    T** ptr,
    std::size_t accountedBytes,
    const GpuMemoryAllocationMetadata& metadata = {}) noexcept {
  return mcclCudaMallocUncached(
      reinterpret_cast<void**>(ptr), accountedBytes, metadata);
}

McclCudaError mcclCudaFree(void* ptr) noexcept;

#ifndef __HIP_PLATFORM_AMD__
template <typename CreateFn>
CUresult mcclCuMemCreate(
    CreateFn createFn,
    CUmemGenericAllocationHandle* handle,
    std::size_t accountedBytes,
    const CUmemAllocationProp* prop,
    unsigned long long flags,
    const GpuMemoryAllocationMetadata& metadata = {}) noexcept {
  const CUresult status = createFn(handle, accountedBytes, prop, flags);
  const auto logicalBytes = metadata.logicalBytesOrAccounted(accountedBytes);
  const auto backingId =
      status == CUDA_SUCCESS ? static_cast<uintptr_t>(*handle) : uintptr_t{0};
  if (status == CUDA_SUCCESS) {
    recordGpuMemoryAllocation(
        metadata.resourceType,
        backingId,
        logicalBytes,
        accountedBytes,
        GpuMemoryBackingKind::kVirtualMemoryHandle);
  }
  MCCL_LOG_SUBSYS(
      INFO,
      ALLOC,
      "GPU memory call=mcclCuMemCreate resource={} logical_bytes={} "
      "accounted_bytes={} backing=0x{:x} status={}",
      gpuMemoryResourceTypeName(metadata.resourceType),
      logicalBytes,
      accountedBytes,
      backingId,
      static_cast<int>(status));
  return status;
}

template <typename ReleaseFn>
CUresult mcclCuMemRelease(
    ReleaseFn releaseFn,
    CUmemGenericAllocationHandle handle) noexcept {
  const auto backingId = static_cast<uintptr_t>(handle);
  constexpr auto kBackingKind = GpuMemoryBackingKind::kVirtualMemoryHandle;
  const auto generation = beginGpuMemoryFree(backingId, kBackingKind);
  const CUresult status = releaseFn(handle);
  finishGpuMemoryFree(
      backingId, kBackingKind, generation, status == CUDA_SUCCESS);
  MCCL_LOG_SUBSYS(
      INFO,
      ALLOC,
      "GPU memory call=mcclCuMemRelease backing=0x{:x} was_tracked={} "
      "status={}",
      backingId,
      generation != 0,
      static_cast<int>(status));
  return status;
}
#endif

} // namespace meta::comms::memtrace
