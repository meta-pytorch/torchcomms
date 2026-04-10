// Copyright (c) Meta Platforms, Inc. and affiliates.

#if defined(ENABLE_PIPES)

#include <cstddef>
#include <new>
#include "comms/ctran/algos/AllToAll/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/AllToAllv.cuh"
#include "comms/pipes/collectives/AllToAllvAutoTuneConfig.h"
#include "comms/pipes/ll128/Ll128Packet.cuh"

// Compute the exclusive prefix sum of counts[0..rank-1] to get the
// displacement for the given rank. Each thread computes its own peer's
// offset independently — no shared memory or __syncthreads() needed.
// For GB200 the NVL domain can have up to 72 ranks, so this is at most
// 71 additions from L1-cached global memory.
__device__ __forceinline__ int64_t
computeDisplacement(const int64_t* counts_d, int rank) {
  int64_t displ = 0;
  for (int r = 0; r < rank; r++) {
    displ += counts_d[r];
  }
  return displ;
}

// Select LL128 or Simple protocol and send data to peer via NVLink.
template <PipeProtocol Proto>
__device__ __forceinline__ void send_peer(
    comms::pipes::Transport& transport,
    comms::pipes::ThreadGroup& group,
    const char* src,
    size_t bytes,
    size_t ll128ThresholdBytes,
    comms::pipes::Timeout timeout) {
  if constexpr (Proto == PipeProtocol::LL128) {
    bool use_ll128 = (bytes <= ll128ThresholdBytes) &&
        comms::pipes::can_use_ll128(src, bytes);
    if (use_ll128) {
      transport.p2p_nvl.ll128_send(
          group, const_cast<char*>(src), bytes, timeout);
    } else {
      transport.p2p_nvl.send(group, const_cast<char*>(src), bytes, timeout);
    }
  } else {
    transport.p2p_nvl.send(group, const_cast<char*>(src), bytes, timeout);
  }
}

// Select LL128 or Simple protocol and receive data from peer via NVLink.
template <PipeProtocol Proto>
__device__ __forceinline__ void recv_peer(
    comms::pipes::Transport& transport,
    comms::pipes::ThreadGroup& group,
    char* dst,
    size_t bytes,
    size_t ll128ThresholdBytes,
    comms::pipes::Timeout timeout) {
  if constexpr (Proto == PipeProtocol::LL128) {
    bool use_ll128 = (bytes <= ll128ThresholdBytes) &&
        comms::pipes::can_use_ll128(dst, bytes);
    if (use_ll128) {
      transport.p2p_nvl.ll128_recv(group, dst, bytes, timeout);
    } else {
      transport.p2p_nvl.recv(group, dst, bytes, timeout);
    }
  } else {
    transport.p2p_nvl.recv(group, dst, bytes, timeout);
  }
}

// Unified NVL+IBGDA kernel launched through GPE submit.
// Builds ChunkInfos in static shared memory from device-side counts,
// performs kernel-side autotune lookup and warp reserve resolution,
// then delegates to the pipes all_to_allv() device function.
// Excess warps (beyond effective count from autotune) early-return.
// No GPE device-side handshake (flag is always nullptr with empty opGroup).
__global__ void __launch_bounds__(512) ncclKernelDeviceAllToAllvPipesUnified(
    int* /* flag */,
    CtranAlgoDeviceState* /* devState */,
    ctran::device_alltoallv_pipes::KernArgs args) {
  __shared__ char
      smem[2 * CTRAN_MAX_TOTAL_RANK * sizeof(comms::pipes::ChunkInfo)];
  __shared__ uint32_t effectiveWarps;
  __shared__ comms::pipes::WarpReserveDeviceConfig resolvedWarpReserve;

  auto* sendChunks = reinterpret_cast<comms::pipes::ChunkInfo*>(smem);
  auto* recvChunks = sendChunks + args.nRanks;

  if (threadIdx.x == 0) {
    size_t sOff = 0, rOff = 0;
    size_t maxBPP = 0;
    for (int r = 0; r < args.nRanks; r++) {
      size_t sBytes = static_cast<size_t>(
                          args.sendcounts_d[r] * args.sendcountsMultiplier) *
          args.elementSize;
      size_t rBytes = static_cast<size_t>(
                          args.recvcounts_d[r] * args.recvcountsMultiplier) *
          args.elementSize;
      new (&sendChunks[r]) comms::pipes::ChunkInfo(sOff, sBytes);
      new (&recvChunks[r]) comms::pipes::ChunkInfo(rOff, rBytes);
      sOff += sBytes;
      rOff += rBytes;
      if (sBytes > maxBPP)
        maxBPP = sBytes;
      if (rBytes > maxBPP)
        maxBPP = rBytes;
    }

    // Per-msg autotune: look up effective grid dims from tables
    int effectiveBlocks;
    int effectiveThreadsPerBlock;
    comms::pipes::WarpReserveConfig mergedCfg = args.warpReserveCvars;

    if (args.hasIbgdaPeers) {
      auto hybridCfg = comms::pipes::getHybridConfigForMsgSize(maxBPP);
      effectiveBlocks = hybridCfg.numBlocks;
      effectiveThreadsPerBlock = hybridCfg.numThreads;
      // Merge per-msg warp reserve with CVAR overrides:
      // CVAR (non-zero) > per-msg autotune > 0 (auto-compute)
      if (mergedCfg.nvlSendWarps == 0)
        mergedCfg.nvlSendWarps = hybridCfg.nvlSendWarps;
      if (mergedCfg.nvlRecvWarps == 0)
        mergedCfg.nvlRecvWarps = hybridCfg.nvlRecvWarps;
      if (mergedCfg.ibgdaSendWarps == 0)
        mergedCfg.ibgdaSendWarps = hybridCfg.ibgdaSendWarps;
      if (mergedCfg.ibgdaRecvWarps == 0)
        mergedCfg.ibgdaRecvWarps = hybridCfg.ibgdaRecvWarps;
      if (mergedCfg.selfWarps == 0)
        mergedCfg.selfWarps = hybridCfg.selfWarps;
    } else {
      auto nvlCfg = comms::pipes::getNvlConfigForMsgSize(maxBPP);
      effectiveBlocks = nvlCfg.numBlocks;
      effectiveThreadsPerBlock = nvlCfg.numThreads;
    }

    // Resolve warp reserve boundaries (inlined resolveWarpReserve logic)
    int numNvlPeers = static_cast<int>(args.warpReserve.numNvlPeers);
    int numIbgdaPeers = static_cast<int>(args.warpReserve.numIbgdaPeers);

    bool anyExplicit = mergedCfg.nvlSendWarps > 0 ||
        mergedCfg.nvlRecvWarps > 0 || mergedCfg.ibgdaSendWarps > 0 ||
        mergedCfg.ibgdaRecvWarps > 0 || mergedCfg.selfWarps > 0;

    if (anyExplicit && (numNvlPeers > 0 || numIbgdaPeers > 0)) {
      int selfW = mergedCfg.selfWarps > 0 ? mergedCfg.selfWarps : 1;
      // Clamp transport-specific warps to 0 when no peers of that type exist,
      // AND round down to nearest multiple of peer count to ensure symmetric
      // channel assignment. Without rounding, partition_interleaved gives
      // uneven per-peer warp counts, creating send/recv channel mismatches
      // that deadlock (e.g., ibgdaSendWarps=8 with 7 peers → peer 0 gets
      // 2 warps/channels, but the remote rank only sends 1 → recv hangs).
      int nvlSendW = numNvlPeers > 0
          ? (mergedCfg.nvlSendWarps > 0
                 ? (mergedCfg.nvlSendWarps / numNvlPeers) * numNvlPeers
                 : 2 * numNvlPeers)
          : 0;
      int nvlRecvW = numNvlPeers > 0
          ? (mergedCfg.nvlRecvWarps > 0
                 ? (mergedCfg.nvlRecvWarps / numNvlPeers) * numNvlPeers
                 : 2 * numNvlPeers)
          : 0;
      int ibgdaSendW = numIbgdaPeers > 0
          ? (mergedCfg.ibgdaSendWarps > 0
                 ? (mergedCfg.ibgdaSendWarps / numIbgdaPeers) * numIbgdaPeers
                 : 1 * numIbgdaPeers)
          : 0;
      int ibgdaRecvW = numIbgdaPeers > 0
          ? (mergedCfg.ibgdaRecvWarps > 0
                 ? (mergedCfg.ibgdaRecvWarps / numIbgdaPeers) * numIbgdaPeers
                 : 1 * numIbgdaPeers)
          : 0;

      resolvedWarpReserve.selfEnd = static_cast<uint32_t>(selfW);
      resolvedWarpReserve.nvlSendEnd =
          resolvedWarpReserve.selfEnd + static_cast<uint32_t>(nvlSendW);
      resolvedWarpReserve.nvlRecvEnd =
          resolvedWarpReserve.nvlSendEnd + static_cast<uint32_t>(nvlRecvW);

      // Block-align IBGDA category starts so all warps for a given IBGDA
      // peer are in the same block (required for named barrier sync).
      int warpsPerBlock = static_cast<int>(blockDim.x / 32);
      uint32_t ibgdaSendBase = (ibgdaSendW > 0)
          ? ((resolvedWarpReserve.nvlRecvEnd + warpsPerBlock - 1) /
             warpsPerBlock) *
              warpsPerBlock
          : resolvedWarpReserve.nvlRecvEnd;
      resolvedWarpReserve.ibgdaSendBase = ibgdaSendBase;
      resolvedWarpReserve.ibgdaSendEnd =
          ibgdaSendBase + static_cast<uint32_t>(ibgdaSendW);

      uint32_t ibgdaRecvBase = (ibgdaRecvW > 0)
          ? ((resolvedWarpReserve.ibgdaSendEnd + warpsPerBlock - 1) /
             warpsPerBlock) *
              warpsPerBlock
          : resolvedWarpReserve.ibgdaSendEnd;
      resolvedWarpReserve.ibgdaRecvBase = ibgdaRecvBase;
      resolvedWarpReserve.ibgdaRecvEnd =
          ibgdaRecvBase + static_cast<uint32_t>(ibgdaRecvW);

      resolvedWarpReserve.nvlPeerRanks = args.warpReserve.nvlPeerRanks;
      resolvedWarpReserve.numNvlPeers = args.warpReserve.numNvlPeers;
      resolvedWarpReserve.ibgdaPeerRanks = args.warpReserve.ibgdaPeerRanks;
      resolvedWarpReserve.numIbgdaPeers = args.warpReserve.numIbgdaPeers;
      resolvedWarpReserve.maxChannelsPerPeer =
          args.warpReserve.maxChannelsPerPeer;

      effectiveWarps = resolvedWarpReserve.ibgdaRecvEnd;
    } else {
      // No warp reserve — use autotune table dimensions
      resolvedWarpReserve = {};
      resolvedWarpReserve.nvlPeerRanks = args.warpReserve.nvlPeerRanks;
      resolvedWarpReserve.numNvlPeers = args.warpReserve.numNvlPeers;
      resolvedWarpReserve.ibgdaPeerRanks = args.warpReserve.ibgdaPeerRanks;
      resolvedWarpReserve.numIbgdaPeers = args.warpReserve.numIbgdaPeers;
      effectiveWarps = static_cast<uint32_t>(
          effectiveBlocks * (effectiveThreadsPerBlock / 32));
    }

    // Safety clamp: ensure effectiveWarps never exceeds the physical grid.
    // Protects against future autotune table updates that exceed
    // kMaxAutotuneBlocks without a corresponding grid increase.
    uint32_t physicalWarps =
        static_cast<uint32_t>(gridDim.x * (blockDim.x / 32));
    if (effectiveWarps > physicalWarps) {
      effectiveWarps = physicalWarps;
    }
  }
  __syncthreads();

  // Early-return excess warps launched beyond effective count
  uint32_t globalWarpId = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (globalWarpId >= effectiveWarps) {
    return;
  }

  comms::pipes::Timeout timeout{};
  comms::pipes::all_to_allv(
      args.recvbuff,
      args.sendbuff,
      args.myRank,
      comms::pipes::DeviceSpan<comms::pipes::Transport>(
          args.transports, static_cast<uint32_t>(args.nRanks)),
      comms::pipes::DeviceSpan<comms::pipes::ChunkInfo>(
          sendChunks, static_cast<uint32_t>(args.nRanks)),
      comms::pipes::DeviceSpan<comms::pipes::ChunkInfo>(
          recvChunks, static_cast<uint32_t>(args.nRanks)),
      timeout,
      resolvedWarpReserve,
      effectiveWarps);
}

template <PipeProtocol Proto>
__global__ void ncclKernelDeviceAllToAllvPipes(
    int* /* flag */,
    CtranAlgoDeviceState* /* devState */,
    ctran::device_alltoallv_pipes::KernArgs args) {
  const int nLocalRanks = args.nLocalRanks;
  const int myRank = args.myRank;
  const size_t elementSize = args.elementSize;
  const int64_t sendMultiplier = args.sendcountsMultiplier;
  const int64_t recvMultiplier = args.recvcountsMultiplier;
  auto* transports = args.transports;

  // LL128 requires warp-level scheduling; Simple supports both.
  auto group = [&]() {
    if constexpr (Proto == PipeProtocol::LL128) {
      return comms::pipes::make_warp_group();
    } else {
      return args.useBlockGroup ? comms::pipes::make_block_group()
                                : comms::pipes::make_warp_group();
    }
  }();

  // Timeout for LL128 path (default = no timeout / infinite wait).
  // Harmless for Simple path (send/recv accept optional Timeout with same
  // default).
  comms::pipes::Timeout timeout{};

  if (nLocalRanks == 1) {
    // Single local rank — self-copy only
    int globalRank = args.localRankToGlobalRank[0];
    size_t sendBytes =
        args.sendcounts_d[globalRank] * sendMultiplier * elementSize;
    size_t sendOffset = computeDisplacement(args.sendcounts_d, globalRank) *
        sendMultiplier * elementSize;
    size_t recvOffset = computeDisplacement(args.recvcounts_d, globalRank) *
        recvMultiplier * elementSize;

    transports[globalRank].self.put(
        group,
        static_cast<char*>(args.recvbuff) + recvOffset,
        static_cast<const char*>(args.sendbuff) + sendOffset,
        sendBytes);
  } else {
    // Split into send and recv halves
    auto [partition_id, send_recv_group] = group.partition_interleaved(2);
    // Distribute across local peers
    auto [local_peer_idx, group_per_peer] =
        send_recv_group.partition_interleaved(nLocalRanks);

    int peerGlobalRank = args.localRankToGlobalRank[local_peer_idx];

    // Read counts from device memory (indexed by global rank)
    size_t sendBytes =
        args.sendcounts_d[peerGlobalRank] * sendMultiplier * elementSize;
    size_t recvBytes =
        args.recvcounts_d[peerGlobalRank] * recvMultiplier * elementSize;
    // Compute displacements as exclusive prefix sums of counts
    size_t sendOffset = computeDisplacement(args.sendcounts_d, peerGlobalRank) *
        sendMultiplier * elementSize;
    size_t recvOffset = computeDisplacement(args.recvcounts_d, peerGlobalRank) *
        recvMultiplier * elementSize;

    const char* src_ptr = static_cast<const char*>(args.sendbuff) + sendOffset;
    char* dst_ptr = static_cast<char*>(args.recvbuff) + recvOffset;

    if (peerGlobalRank == myRank) {
      if (partition_id == 0) {
        transports[peerGlobalRank].self.put(
            group_per_peer, dst_ptr, src_ptr, sendBytes);
      }
    } else if (partition_id == 0) {
      send_peer<Proto>(
          transports[peerGlobalRank],
          group_per_peer,
          src_ptr,
          sendBytes,
          args.ll128ThresholdBytes,
          timeout);
    } else {
      recv_peer<Proto>(
          transports[peerGlobalRank],
          group_per_peer,
          dst_ptr,
          recvBytes,
          args.ll128ThresholdBytes,
          timeout);
    }
  }
}

// Explicit template instantiations for both protocols.
template __global__ void ncclKernelDeviceAllToAllvPipes<PipeProtocol::Simple>(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::device_alltoallv_pipes::KernArgs args);

template __global__ void ncclKernelDeviceAllToAllvPipes<PipeProtocol::LL128>(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::device_alltoallv_pipes::KernArgs args);

#endif // ENABLE_PIPES
