// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/AllToAllvAuto.h"

#include "comms/pipes/collectives/AllToAllv.h"
#include "comms/pipes/collectives/AllToAllvAutoTuneConfig.h"
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
    bool has_ibgda_peers,
    const AllToAllvAutoConfig& config,
    std::chrono::milliseconds timeout,
    cudaStream_t stream) {
  if (has_ibgda_peers) {
    // IBGDA peers present — use IBGDA-capable path which handles
    // both NVL and IBGDA via per-peer dispatch.
    //
    // Use autotune lookup table for per-msg-size config when no explicit
    // override is provided (ibgdaNumBlocks <= 0 means use autotune).
    auto hybridCfg = getHybridConfigForMsgSize(max_bytes_per_peer);
    int blocks = (config.ibgdaNumBlocks > 0) ? config.ibgdaNumBlocks
                                             : hybridCfg.numBlocks;

    // Compute optimal thread count dynamically:
    // Minimum = 2 (send/recv partitions) * nranks * 32 (warpSize)
    int ibgdaThreads = 2 * nranks * 32;

    all_to_allv(
        recvbuff_d,
        sendbuff_d,
        my_rank_id,
        transports_per_rank,
        send_chunk_infos,
        recv_chunk_infos,
        timeout,
        stream,
        blocks,
        ibgdaThreads);
  } else if (max_bytes_per_peer <= config.ll128Threshold) {
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
