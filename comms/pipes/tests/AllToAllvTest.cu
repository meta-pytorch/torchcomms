// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/pipes/tests/AllToAllvTest.cuh"

#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::test {

// Kernel that calls allToAllv
__global__ void testAllToAllvKernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    int nranks,
    DeviceSpan<Transport> transports,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos) {
  // Call allToAllv - it will perform actual data transfers
  allToAllv(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      transports,
      send_chunk_infos,
      recv_chunk_infos);
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
  testAllToAllvKernel<<<numBlocks, blockSize>>>(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      nranks,
      transports,
      send_chunk_infos,
      recv_chunk_infos);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
