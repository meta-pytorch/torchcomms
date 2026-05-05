// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#if defined(ENABLE_PIPES)

#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/Timeout.cuh"

__device__ __forceinline__ size_t devRankChunkOffset(
    int anchorNode,
    int localRank,
    int nLocalRanks,
    int nNodes,
    int step,
    int j) {
  const int stride = nNodes >> step;
  const int nodePos = j * stride + (anchorNode % stride);
  return static_cast<size_t>(nodePos) * nLocalRanks + localRank;
}

namespace ctran::allgatherp {

__global__ __launch_bounds__(512, 1) void ncclKernelAllGatherPNvlDissem(
    int* /* flag */,
    CtranAlgoDeviceState* /* devState */,
    NvlDissemKernArgs args) {
  comms::pipes::Timeout timeout(args.timeoutCycles);
  timeout.start();

  auto group = comms::pipes::make_warp_group();

  // Split into send and recv halves
  auto [partition_id, send_recv_group] = group.partition_interleaved(2);
  // Distribute across local peers (including self)
  auto [local_peer_idx, group_per_peer] =
      send_recv_group.partition_interleaved(args.nLocalRanks);

  if (local_peer_idx == args.localRank) {
    // Self-copy: staging → own recvbuff
    if (partition_id == 0) {
      for (int j = 0; j < args.nChunks; j++) {
        const size_t chunkIdx = devRankChunkOffset(
            args.peerNode,
            args.localRank,
            args.nLocalRanks,
            args.nNodes,
            args.step,
            j);
        char* dst =
            static_cast<char*>(args.recvbuff) + chunkIdx * args.chunkSize;
        char* src =
            static_cast<char*>(args.stagingRecvBuf) + j * args.chunkSize;
        args.nvlTransportsBase[args.localRank].put_group(
            group_per_peer, dst, src, args.chunkSize);
      }
    }
  } else if (partition_id == 0) {
    // Send: staging → peer via NVLink
    for (int j = 0; j < args.nChunks; j++) {
      char* src = static_cast<char*>(args.stagingRecvBuf) + j * args.chunkSize;
      args.nvlTransportsBase[local_peer_idx].send_group(
          group_per_peer, src, args.chunkSize, timeout);
    }
  } else {
    // Recv: peer → own recvbuff via NVLink
    for (int j = 0; j < args.nChunks; j++) {
      const size_t chunkIdx = devRankChunkOffset(
          args.peerNode,
          local_peer_idx,
          args.nLocalRanks,
          args.nNodes,
          args.step,
          j);
      char* dst = static_cast<char*>(args.recvbuff) + chunkIdx * args.chunkSize;
      args.nvlTransportsBase[local_peer_idx].recv_group(
          group_per_peer, dst, args.chunkSize, timeout);
    }
  }
}

} // namespace ctran::allgatherp

#endif // ENABLE_PIPES
