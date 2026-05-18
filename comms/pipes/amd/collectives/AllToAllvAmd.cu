#if defined(__HIPCC__) || !defined(__CUDACC__)
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "AllToAllvAmd.cuh"

#include <hip/hip_runtime.h>
#include "comms/pipes/Checks.h"

namespace comms::pipes {

__global__ void allToAllvAmdKernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    Timeout timeout,
    void* ibgda_transport_base,
    std::size_t ibgda_transport_stride,
    int* ibgda_peer_ranks,
    int num_ibgda_peers,
    IbgdaLocalBuffer ibgda_send_buf,
    IbgdaRemoteBuffer* ibgda_recv_bufs) {
  timeout.start();
  all_to_allv_hybrid_amd(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      transports_per_rank,
      send_chunk_infos,
      recv_chunk_infos,
      timeout,
      ibgda_transport_base,
      ibgda_transport_stride,
      ibgda_peer_ranks,
      num_ibgda_peers,
      ibgda_send_buf,
      ibgda_recv_bufs);
}

void all_to_allv_amd(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    void* ibgda_transport_base,
    std::size_t ibgda_transport_stride,
    int* ibgda_peer_ranks,
    int num_ibgda_peers,
    IbgdaLocalBuffer ibgda_send_buf,
    IbgdaRemoteBuffer* ibgda_recv_bufs,
    hipStream_t stream,
    int num_blocks,
    int num_threads) {
  Timeout timeout; // disabled (0ms)

  allToAllvAmdKernel<<<num_blocks, num_threads, 0, stream>>>(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      transports_per_rank,
      send_chunk_infos,
      recv_chunk_infos,
      timeout,
      ibgda_transport_base,
      ibgda_transport_stride,
      ibgda_peer_ranks,
      num_ibgda_peers,
      ibgda_send_buf,
      ibgda_recv_bufs);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes
#endif
