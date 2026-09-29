// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/Checks.h"

namespace comms::prims::test {

using namespace comms::prims;

__global__ void testCopyChunkVectorizedKernel(
    char* dst_d,
    const char* src_d,
    std::size_t chunk_bytes,
    uint32_t* errorCount_d) {
  auto warp = make_warp_group();

  memcpy_vectorized(dst_d, src_d, chunk_bytes, warp);

  __syncthreads();

  if (warp.is_leader() && warp.group_id == 0) {
    for (std::size_t i = 0; i < chunk_bytes; i++) {
      if (dst_d[i] != src_d[i]) {
        atomicAdd(errorCount_d, 1);
      }
    }
  }
}

void testCopyChunkVectorized(
    char* dst_d,
    const char* src_d,
    std::size_t chunk_bytes,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testCopyChunkVectorizedKernel<<<numBlocks, blockSize>>>(
      dst_d, src_d, chunk_bytes, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
