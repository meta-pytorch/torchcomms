// Copyright (c) Meta Platforms, Inc. and affiliates.

#if defined(ENABLE_PRIMS)

#include <cstddef>
#include "comms/ctran/algos/AllToAll/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeRing.h"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/Transport.cuh"
#include "comms/prims/transport/ll128/Ll128Packet.cuh"

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
    comms::prims::Transport& transport,
    comms::prims::ThreadGroup& group,
    const char* src,
    size_t bytes,
    size_t ll128ThresholdBytes,
    comms::prims::Timeout timeout) {
  if constexpr (Proto == PipeProtocol::LL128) {
    bool use_ll128 = (bytes <= ll128ThresholdBytes) &&
        comms::prims::can_use_ll128(src, bytes);
    if (use_ll128) {
      transport.p2p_nvl.ll128_send_group(
          group, const_cast<char*>(src), bytes, timeout);
    } else {
      comms::prims::TiledBuffer<char> tiles(
          const_cast<char*>(src), bytes, group);
      transport.p2p_nvl.send(
          group,
          tiles.tile_data(group.group_id),
          tiles.tile_bytes(group.group_id),
          /*max_signal_bytes=*/0,
          timeout);
    }
  } else {
    comms::prims::TiledBuffer<char> tiles(const_cast<char*>(src), bytes, group);
    transport.p2p_nvl.send(
        group,
        tiles.tile_data(group.group_id),
        tiles.tile_bytes(group.group_id),
        /*max_signal_bytes=*/0,
        timeout);
  }
}

// Select LL128 or Simple protocol and receive data from peer via NVLink.
template <PipeProtocol Proto>
__device__ __forceinline__ void recv_peer(
    comms::prims::Transport& transport,
    comms::prims::ThreadGroup& group,
    char* dst,
    size_t bytes,
    size_t ll128ThresholdBytes,
    comms::prims::Timeout timeout) {
  if constexpr (Proto == PipeProtocol::LL128) {
    bool use_ll128 = (bytes <= ll128ThresholdBytes) &&
        comms::prims::can_use_ll128(dst, bytes);
    if (use_ll128) {
      transport.p2p_nvl.ll128_recv_group(group, dst, bytes, timeout);
    } else {
      comms::prims::TiledBuffer<char> tiles(dst, bytes, group);
      transport.p2p_nvl.recv(
          group,
          tiles.tile_data(group.group_id),
          tiles.tile_bytes(group.group_id),
          /*max_signal_bytes=*/0,
          timeout);
    }
  } else {
    comms::prims::TiledBuffer<char> tiles(dst, bytes, group);
    transport.p2p_nvl.recv(
        group,
        tiles.tile_data(group.group_id),
        tiles.tile_bytes(group.group_id),
        /*max_signal_bytes=*/0,
        timeout);
  }
}

template <PipeProtocol Proto>
__global__ void ncclKernelDeviceAllToAllvPipes(
    ctran::gpe::KernelFlagDev* /* flag */,
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
      return comms::prims::make_warp_group();
    } else {
      return args.useBlockGroup ? comms::prims::make_block_group()
                                : comms::prims::make_warp_group();
    }
  }();

  // Timeout for LL128 path (default = no timeout / infinite wait).
  // Harmless for Simple path (send/recv accept optional Timeout with same
  // default).
  comms::prims::Timeout timeout{};

  if (nLocalRanks == 1) {
    // Single local rank — self-copy only
    int globalRank = args.localRankToGlobalRank[0];
    size_t sendBytes =
        args.sendcounts_d[globalRank] * sendMultiplier * elementSize;
    size_t sendOffset = computeDisplacement(args.sendcounts_d, globalRank) *
        sendMultiplier * elementSize;
    size_t recvOffset = computeDisplacement(args.recvcounts_d, globalRank) *
        recvMultiplier * elementSize;

    transports[globalRank].self.put_group(
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
        transports[peerGlobalRank].self.put_group(
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
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState,
    ctran::device_alltoallv_pipes::KernArgs args);

template __global__ void ncclKernelDeviceAllToAllvPipes<PipeProtocol::LL128>(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState,
    ctran::device_alltoallv_pipes::KernArgs args);

#endif // ENABLE_PRIMS
