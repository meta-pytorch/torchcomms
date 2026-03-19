// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/tests/AllToAllvLl128Test.cuh"

#include <cstddef>

#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::test {

// Kernel that calls all_to_allv_ll128
__global__ void test_all_to_allv_ll128_kernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    int nranks,
    DeviceSpan<Transport> transports,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    Timeout timeout) {
  timeout.start();
  all_to_allv_ll128(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      transports,
      send_chunk_infos,
      recv_chunk_infos,
      timeout);
}

void test_all_to_allv_ll128(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    int nranks,
    DeviceSpan<Transport> transports,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    int numBlocks,
    int blockSize,
    Timeout timeout) {
  test_all_to_allv_ll128_kernel<<<numBlocks, blockSize>>>(
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

} // namespace comms::pipes::test
