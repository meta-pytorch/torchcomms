// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Include .cuh (not .h) so the __global__ kernel below only sees the __device__
// overload of all_to_allv_ll128.  Including .h would bring in the host overload
// whose first 9 parameter types are identical, causing an ambiguous-overload
// error in NVCC (it resolves C++ overloads before __host__/__device__
// filtering).
#include "comms/pipes/collectives/AllToAllvLl128.cuh"

#include <chrono>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes {

/**
 * AllToAllv LL128 kernel.
 * Wrapper kernel that calls the device all_to_allv_ll128 function.
 */
__global__ void all_to_allv_ll128_kernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    Timeout timeout) {
  timeout.start();
  all_to_allv_ll128(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      transports_per_rank,
      send_chunk_infos,
      recv_chunk_infos,
      timeout);
}

void all_to_allv_ll128(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    Timeout timeout_config,
    cudaStream_t stream,
    int num_blocks,
    int num_threads) {
  void* args[] = {
      &recvbuff_d,
      &sendbuff_d,
      &my_rank_id,
      &transports_per_rank,
      &send_chunk_infos,
      &recv_chunk_infos,
      &timeout_config};

  comms::common::launchKernel(
      (void*)all_to_allv_ll128_kernel,
      dim3(num_blocks),
      dim3(num_threads),
      args,
      stream,
      std::nullopt); // No cluster launch — LL128 volatile stores bypass L1
  PIPES_KERNEL_LAUNCH_CHECK();
}

void all_to_allv_ll128(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    std::chrono::milliseconds timeout,
    cudaStream_t stream,
    int num_blocks,
    int num_threads) {
  // Get current device for timeout creation
  int device = 0;
  PIPES_CUDA_CHECK(cudaGetDevice(&device));
  Timeout timeout_config =
      makeTimeout(static_cast<uint32_t>(timeout.count()), device);

  all_to_allv_ll128(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      transports_per_rank,
      send_chunk_infos,
      recv_chunk_infos,
      timeout_config,
      stream,
      num_blocks,
      num_threads);
}

} // namespace comms::pipes
