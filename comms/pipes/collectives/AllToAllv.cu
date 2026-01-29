// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/AllToAllv.h"

#include "comms/common/CudaWrap.h"
#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes {

/**
 * AllToAllv kernel.
 * Wrapper kernel that calls the device all_to_allv function.
 */
__global__ void allToAllvKernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    Timeout timeout) {
  timeout.start();
  all_to_allv(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      transports_per_rank,
      send_chunk_infos,
      recv_chunk_infos,
      timeout);
}

void all_to_allv(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
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
      &my_rank_id,
      &transports_per_rank,
      &send_chunk_infos,
      &recv_chunk_infos,
      &timeout_config};

  comms::common::launchKernel(
      (void*)allToAllvKernel,
      dim3(num_blocks),
      dim3(num_threads),
      args,
      stream,
      cluster_dim);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes
