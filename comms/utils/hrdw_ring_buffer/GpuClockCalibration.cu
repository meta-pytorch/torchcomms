// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <cstdint>

#include "comms/utils/hrdw_ring_buffer/GpuClockCalibration.h"

namespace hrdw_ring_buffer {

namespace {

__global__ void readGlobaltimerKernel(uint64_t* out) {
  *out = readGlobaltimer();
}

} // namespace

cudaError_t launchReadGlobaltimer(cudaStream_t stream, uint64_t* out) {
  readGlobaltimerKernel<<<1, 1, 0, stream>>>(out);
  return cudaGetLastError();
}

} // namespace hrdw_ring_buffer
