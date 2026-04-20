// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/tests/cudagraph/DeviceVerify.h"

namespace ctran::testing {

__global__ void compareBuffersKernel(
    const char* actual,
    const char* expected,
    size_t numBytes,
    unsigned int* mismatchCount) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numBytes && actual[idx] != expected[idx]) {
    atomicAdd(mismatchCount, 1);
  }
}

void launchCompareBuffers(
    const void* actual,
    const void* expected,
    size_t numBytes,
    unsigned int* mismatchCount_d,
    cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  int numBlocks = (numBytes + kBlockSize - 1) / kBlockSize;
  compareBuffersKernel<<<numBlocks, kBlockSize, 0, stream>>>(
      static_cast<const char*>(actual),
      static_cast<const char*>(expected),
      numBytes,
      mismatchCount_d);
}

} // namespace ctran::testing
