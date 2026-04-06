// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <cstdint>

#include "comms/utils/HRDWRingBuffer.h"
#include "comms/utils/colltrace/GpuClockCalibration.h"

namespace meta::comms::colltrace {

namespace {

__global__ void ringBufferWriteKernel(
    HRDWEntry* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    void* data) {
  uint64_t slot =
      atomicAdd(reinterpret_cast<unsigned long long*>(writeIdx), 1ULL);
  uint64_t idx = slot & mask;
  ring[idx].timestamp_ns = readGlobaltimer();
  ring[idx].data = data;
  // guarantee that when if the reader sees sequence == slot,
  // the updated timestamp_ns and data are updated.
  __threadfence_system();
  ring[idx].sequence = slot;
}

} // namespace

cudaError_t launchRingBufferWrite(
    cudaStream_t stream,
    void* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    void* data) {
  ringBufferWriteKernel<<<1, 1, 0, stream>>>(
      static_cast<HRDWEntry*>(ring), writeIdx, mask, data);
  return cudaGetLastError();
}

} // namespace meta::comms::colltrace
