// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <cuda_runtime.h>

namespace ctran::algos {

// Blocks co-resident on one SM. dynamicSMemSize must match the launch, or the
// answer describes a launch that never happens. Returns 0 on failure: callers
// only report this in stats.
inline int
getBlocksPerSM(const void* func, int blockSize, size_t dynamicSMemSize) {
  if (func == nullptr || blockSize <= 0) {
    return 0;
  }

  int blocksPerSM = 0;
  const cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocksPerSM, func, blockSize, dynamicSMemSize);
  if (err != cudaSuccess) {
    // Consume only our own error; an unconditional clear would hide a sticky
    // async fault from the next FB_CUDACHECK.
    if (cudaPeekAtLastError() == err) {
      (void)cudaGetLastError();
    }
    return 0;
  }
  return blocksPerSM;
}

} // namespace ctran::algos
