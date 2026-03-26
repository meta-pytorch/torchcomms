// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <cstdint>

#include "comms/utils/HRDWRingBuffer.h"

namespace meta::comms::colltrace {

namespace {

__device__ __forceinline__ uint64_t readGlobaltimer() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return wall_clock64();
#else
  uint64_t timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
#endif
}

__global__ void ringBufferStartWriteKernel(
    HRDWEntry* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    void* data,
    uint64_t* slotOut) {
  uint64_t slot =
      atomicAdd(reinterpret_cast<unsigned long long*>(writeIdx), 1ULL);
  uint64_t idx = slot & mask;
  ring[idx].start_ns = readGlobaltimer();
  ring[idx].data = data;
  ring[idx].sequence = HRDW_RINGBUFFER_WRITE_PENDING;
  // Ensure start_ns, data, and WRITE_PENDING are visible to the CPU before
  // publishing the slot index. The CPU reader may observe sequence ==
  // WRITE_PENDING and then snapshot start_ns/data for per-collective detection.
  __threadfence_system();
  *slotOut = slot;
}

__global__ void ringBufferEndWriteKernel(
    HRDWEntry* ring,
    uint64_t* slotIn,
    uint32_t mask,
    uint64_t* completionCounter) {
  uint64_t slot = *slotIn;
  uint64_t idx = slot & mask;
  ring[idx].end_ns = readGlobaltimer();
  // Ensure end_ns is visible to the CPU before the sequence stamp.
  // The sequence acts as a release flag: the CPU reader checks
  // sequence == slot before reading other fields.
  __threadfence_system();
  ring[idx].sequence = slot;
  if (completionCounter) {
    atomicAdd(reinterpret_cast<unsigned long long*>(completionCounter), 1ULL);
  }
}

} // namespace

cudaError_t launchRingBufferStartWrite(
    cudaStream_t stream,
    void* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    void* data,
    uint64_t* slotOut) {
  ringBufferStartWriteKernel<<<1, 1, 0, stream>>>(
      static_cast<HRDWEntry*>(ring), writeIdx, mask, data, slotOut);
  return cudaGetLastError();
}

cudaError_t launchRingBufferEndWrite(
    cudaStream_t stream,
    void* ring,
    uint64_t* slotIn,
    uint32_t mask,
    uint64_t* completionCounter) {
  ringBufferEndWriteKernel<<<1, 1, 0, stream>>>(
      static_cast<HRDWEntry*>(ring), slotIn, mask, completionCounter);
  return cudaGetLastError();
}

} // namespace meta::comms::colltrace
