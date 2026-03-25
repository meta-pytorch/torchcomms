// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/AllToAllvAuto.h"

#include "comms/pipes/collectives/AllToAllv.h"
#include "comms/pipes/collectives/AllToAllvLl128.h"
#include "comms/pipes/ll128/Ll128AutoTune.cuh"

namespace comms::pipes {

void all_to_allv_auto(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    int nranks,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    std::size_t max_bytes_per_peer,
    const AllToAllvAutoConfig& config,
    std::chrono::milliseconds timeout,
    cudaStream_t stream) {
  if (max_bytes_per_peer <= config.ll128Threshold) {
    int blocks = config.ll128NumBlocks;
    if (blocks <= 0) {
      blocks = ll128_auto_tune_alltoallv(max_bytes_per_peer, nranks).numBlocks;
    }

    all_to_allv_ll128(
        recvbuff_d,
        sendbuff_d,
        my_rank_id,
        transports_per_rank,
        send_chunk_infos,
        recv_chunk_infos,
        timeout,
        stream,
        blocks,
        config.ll128NumThreads);
  } else {
    all_to_allv(
        recvbuff_d,
        sendbuff_d,
        my_rank_id,
        transports_per_rank,
        send_chunk_infos,
        recv_chunk_infos,
        timeout,
        stream,
        config.simpleNumBlocks,
        config.simpleNumThreads,
        config.simpleClusterDim);
  }
}

} // namespace comms::pipes
