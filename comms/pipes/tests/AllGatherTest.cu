// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/AllGatherTest.cuh"

#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::test {

// Kernel that calls all_gather
__global__ void testAllGatherKernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    std::size_t sendcount,
    int my_rank_id,
    int nranks,
    DeviceSpan<Transport> transports) {
  // Call all_gather - it will perform actual data transfers
  all_gather(
      recvbuff_d, sendbuff_d, sendcount, my_rank_id, transports, Timeout());
}

void testAllGather(
    void* recvbuff_d,
    const void* sendbuff_d,
    std::size_t sendcount,
    int my_rank_id,
    int nranks,
    DeviceSpan<Transport> transports,
    int numBlocks,
    int blockSize) {
  testAllGatherKernel<<<numBlocks, blockSize>>>(
      recvbuff_d, sendbuff_d, sendcount, my_rank_id, nranks, transports);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
