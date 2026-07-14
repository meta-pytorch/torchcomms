// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/prims/tests/AllToAllvTest.cuh"

#include "comms/prims/tests/Checks.h"

namespace comms::prims::test {

// Kernel that calls all_to_allv
__global__ __launch_bounds__(512, 1) void testAllToAllvKernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    int nranks,
    DeviceSpan<Transport> transports,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    Timeout timeout) {
  timeout.start();
  // Call all_to_allv - it will perform actual data transfers
  all_to_allv(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      transports,
      send_chunk_infos,
      recv_chunk_infos,
      timeout);
}

void testAllToAllv(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    int nranks,
    DeviceSpan<Transport> transports,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    int numBlocks,
    int blockSize) {
  Timeout timeout; // Default no timeout
  testAllToAllvKernel<<<numBlocks, blockSize>>>(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      nranks,
      transports,
      send_chunk_infos,
      recv_chunk_infos,
      timeout);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
