// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <cstdint>

#include "comms/utils/GpuClockCalibration.h"

namespace meta::comms::colltrace {

namespace {

__global__ void readGlobaltimerKernel(uint64_t* out) {
  *out = readGlobaltimer();
}

} // namespace

cudaError_t launchReadGlobaltimer(cudaStream_t stream, uint64_t* out) {
  readGlobaltimerKernel<<<1, 1, 0, stream>>>(out);
  return cudaGetLastError();
}

} // namespace meta::comms::colltrace
