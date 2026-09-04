// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/memtrace/McclCudaMemory.h"

#include "comms/mccl/utils/McclLogger.h"

namespace meta::comms::memtrace {
namespace {

void recordSuccessfulAllocation(
    void* ptr,
    std::size_t accountedBytes,
    const GpuMemoryAllocationMetadata& metadata) noexcept {
  recordGpuMemoryAllocation(
      metadata.resourceType,
      reinterpret_cast<uintptr_t>(ptr),
      metadata.logicalBytesOrAccounted(accountedBytes),
      accountedBytes);
}

} // namespace

McclCudaError mcclCudaMalloc(
    void** ptr,
    std::size_t accountedBytes,
    const GpuMemoryAllocationMetadata& metadata) noexcept {
#ifdef __HIP_PLATFORM_AMD__
  const auto status = hipMalloc(ptr, accountedBytes);
  const bool succeeded = status == hipSuccess;
#else
  const auto status = cudaMalloc(ptr, accountedBytes);
  const bool succeeded = status == cudaSuccess;
#endif
  if (succeeded) {
    recordSuccessfulAllocation(*ptr, accountedBytes, metadata);
  }
  MCCL_LOG_SUBSYS(
      INFO,
      ALLOC,
      "GPU memory call=mcclCudaMalloc resource={} logical_bytes={} "
      "accounted_bytes={} backing=0x{:x} status={}",
      gpuMemoryResourceTypeName(metadata.resourceType),
      metadata.logicalBytesOrAccounted(accountedBytes),
      accountedBytes,
      succeeded ? reinterpret_cast<uintptr_t>(*ptr) : 0,
      static_cast<int>(status));
  return status;
}

McclCudaError mcclCudaMallocUncached(
    void** ptr,
    std::size_t accountedBytes,
    const GpuMemoryAllocationMetadata& metadata) noexcept {
#ifdef __HIP_PLATFORM_AMD__
  const auto status =
      hipExtMallocWithFlags(ptr, accountedBytes, hipDeviceMallocUncached);
  if (status == hipSuccess) {
    recordSuccessfulAllocation(*ptr, accountedBytes, metadata);
  }
  MCCL_LOG_SUBSYS(
      INFO,
      ALLOC,
      "GPU memory call=mcclCudaMallocUncached resource={} logical_bytes={} "
      "accounted_bytes={} backing=0x{:x} status={}",
      gpuMemoryResourceTypeName(metadata.resourceType),
      metadata.logicalBytesOrAccounted(accountedBytes),
      accountedBytes,
      status == hipSuccess ? reinterpret_cast<uintptr_t>(*ptr) : 0,
      static_cast<int>(status));
  return status;
#else
  const auto status = cudaMalloc(ptr, accountedBytes);
  if (status == cudaSuccess) {
    recordSuccessfulAllocation(*ptr, accountedBytes, metadata);
  }
  MCCL_LOG_SUBSYS(
      INFO,
      ALLOC,
      "GPU memory call=mcclCudaMallocUncached resource={} logical_bytes={} "
      "accounted_bytes={} backing=0x{:x} status={}",
      gpuMemoryResourceTypeName(metadata.resourceType),
      metadata.logicalBytesOrAccounted(accountedBytes),
      accountedBytes,
      status == cudaSuccess ? reinterpret_cast<uintptr_t>(*ptr) : 0,
      static_cast<int>(status));
  return status;
#endif
}

McclCudaError mcclCudaFree(void* ptr) noexcept {
  const auto backingId = reinterpret_cast<uintptr_t>(ptr);
  const auto generation = beginGpuMemoryFree(backingId);
#ifdef __HIP_PLATFORM_AMD__
  const auto status = hipFree(ptr);
  finishGpuMemoryFree(
      backingId,
      GpuMemoryBackingKind::kRuntimeAllocation,
      generation,
      status == hipSuccess);
#else
  const auto status = cudaFree(ptr);
  finishGpuMemoryFree(
      backingId,
      GpuMemoryBackingKind::kRuntimeAllocation,
      generation,
      status == cudaSuccess);
#endif
  MCCL_LOG_SUBSYS(
      INFO,
      ALLOC,
      "GPU memory call=mcclCudaFree backing=0x{:x} was_tracked={} status={}",
      backingId,
      generation != 0,
      static_cast<int>(status));
  return status;
}

} // namespace meta::comms::memtrace
