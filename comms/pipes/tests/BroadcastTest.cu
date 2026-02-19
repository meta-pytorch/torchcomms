// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/BroadcastTest.cuh"

#include "comms/common/CudaWrap.h"
#include "comms/pipes/collectives/BroadcastContext.cuh"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::collectives::test {

__global__ void testBroadcastKernel(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes) {
  broadcast(buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
}

void testBroadcast(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim,
    cudaStream_t stream) {
  void* args[] = {
      &buff_d, &my_rank_id, &root_rank_id, &transports_per_rank, &nbytes};
  PIPES_CUDA_CHECK(
      comms::common::launchKernel(
          (void*)testBroadcastKernel,
          dim3(numBlocks, 1, 1),
          dim3(blockSize, 1, 1),
          args,
          stream,
          clusterDim));
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::collectives::test
