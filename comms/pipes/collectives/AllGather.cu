// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/AllGather.h"

#include "comms/common/CudaWrap.h"
#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes {

/**
 * AllGather kernel.
 * Wrapper kernel that calls the device all_gather function.
 */
__global__ void allGatherKernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    std::size_t sendcount,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    Timeout timeout) {
  timeout.start();
  all_gather(
      recvbuff_d,
      sendbuff_d,
      sendcount,
      my_rank_id,
      transports_per_rank,
      timeout);
}

void all_gather(
    void* recvbuff_d,
    const void* sendbuff_d,
    std::size_t sendcount,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::chrono::milliseconds timeout,
    cudaStream_t stream,
    int num_blocks,
    int num_threads,
    std::optional<dim3> cluster_dim) {
  // Get current device for timeout creation
  int device = 0;
  PIPES_CUDA_CHECK(cudaGetDevice(&device));
  Timeout timeout_config =
      makeTimeout(static_cast<uint32_t>(timeout.count()), device);

  void* args[] = {
      &recvbuff_d,
      &sendbuff_d,
      &sendcount,
      &my_rank_id,
      &transports_per_rank,
      &timeout_config};

  comms::common::launchKernel(
      (void*)allGatherKernel,
      dim3(num_blocks),
      dim3(num_threads),
      args,
      stream,
      cluster_dim);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes
