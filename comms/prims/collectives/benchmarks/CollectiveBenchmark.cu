// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/benchmarks/CollectiveBenchmark.cuh"

namespace comms::prims::benchmark {

__global__ void all_to_allv_kernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    AbortDevice abortDevice) {
  all_to_allv(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      transports_per_rank,
      send_chunk_infos,
      recv_chunk_infos,
      abortDevice);
}

__global__ void all_gather_kernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    std::size_t sendcount,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    AbortDevice abortDevice) {
  all_gather(
      recvbuff_d,
      sendbuff_d,
      sendcount,
      my_rank_id,
      transports_per_rank,
      abortDevice);
}

} // namespace comms::prims::benchmark
