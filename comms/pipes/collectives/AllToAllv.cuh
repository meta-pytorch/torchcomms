// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/DeviceCheck.cuh"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/Transport.cuh"

// P2pIbgdaTransportDevice.cuh includes DOCA device headers with CUDA-only
// intrinsics (atomicCAS, __ldg, etc.) that cannot compile in .cc translation
// units. Guard with __CUDA_ARCH__ so the header is only included during device
// compilation. Transport.cuh provides the forward declaration of
// P2pIbgdaTransportDevice for the pointer type.
#ifdef __CUDA_ARCH__
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#endif

namespace comms::pipes {

/**
 * Chunk metadata for all_to_allv operation.
 * Describes a contiguous chunk of data to send or receive for a specific peer.
 */
struct ChunkInfo {
  std::size_t offset; // offset in bytes from buffer base address
  std::size_t nbytes; // number of bytes to send or recv

  __host__ __device__ __forceinline__
  ChunkInfo(std::size_t offset, std::size_t nbytes)
      : offset(offset), nbytes(nbytes) {}
};

/**
 * Host-side warp reservation config for hybrid NVLink+IBGDA AllToAllv.
 *
 * Controls exact warp allocation across five categories:
 * NVL-send, NVL-recv, IBGDA-send, IBGDA-recv, and self-copy.
 * A value of 0 means "auto-compute from peer counts" (backward compatible):
 *   - selfWarps:     default 1
 *   - nvlSendWarps:  default 2 per NVL peer
 *   - nvlRecvWarps:  default 2 per NVL peer
 *   - ibgdaSendWarps: default 1 per IBGDA peer
 *   - ibgdaRecvWarps: default 1 per IBGDA peer
 */
struct WarpReserveConfig {
  int nvlSendWarps = 0;
  int nvlRecvWarps = 0;
  int ibgdaSendWarps = 0;
  int ibgdaRecvWarps = 0;
  int selfWarps = 0;
};

/**
 * Device-side warp reserve config with precomputed cumulative boundaries.
 *
 * Warp categories are ordered: [self | nvlSend | nvlRecv | ibgdaSend |
 * ibgdaRecv]. Each *End field is the exclusive boundary (first warp_id NOT in
 * that category). Warps beyond ibgdaRecvEnd are excess from grid oversizing
 * and must early-return.
 *
 * When isConfigured() returns false, the kernel falls back to the existing
 * uniform partition_interleaved logic for full backward compatibility.
 */
struct WarpReserveDeviceConfig {
  uint32_t selfEnd = 0;
  uint32_t nvlSendEnd = 0;
  uint32_t nvlRecvEnd = 0;
  uint32_t ibgdaSendBase = 0; // block-aligned start of IBGDA send category
  uint32_t ibgdaSendEnd = 0;
  uint32_t ibgdaRecvBase = 0; // block-aligned start of IBGDA recv category
  uint32_t ibgdaRecvEnd = 0;

  const int* nvlPeerRanks = nullptr;
  uint32_t numNvlPeers = 0;
  const int* ibgdaPeerRanks = nullptr;
  uint32_t numIbgdaPeers = 0;

  // Maximum channels per IBGDA peer (derived from autotune table at setup).
  // Used to cap numActiveChannels = min(warpsPerPeer, maxChannelsPerPeer).
  uint32_t maxChannelsPerPeer = 1;

  __host__ __device__ bool isConfigured() const {
    return nvlSendEnd > 0 || ibgdaSendEnd > 0 || selfEnd > 0;
  }
};
namespace {
/**
 * Debug helper to print all_to_allv communication information.
 * Automatically detects self-copy vs peer communication based on my_rank ==
 * peer_rank.
 */
__device__ __forceinline__ void printPerPeerOperation(
    int my_rank_id,
    int peer_rank_id,
    int partition_id,
    uint32_t total_groups,
    size_t send_offset,
    size_t recv_offset,
    size_t send_nbytes,
    size_t recv_nbytes) {
  if (my_rank_id == peer_rank_id) {
    // Self-copy
    printf(
        "Rank=%d pid=%d total-groups=%d: self-copy send-offset=%lu, recv-offset=%lu nbytes=%lu\n",
        my_rank_id,
        partition_id,
        total_groups,
        send_offset,
        recv_offset,
        send_nbytes);
  } else {
    // Peer communication
    bool is_send = (partition_id == 0);
    size_t offset = is_send ? send_offset : recv_offset;
    size_t nbytes = is_send ? send_nbytes : recv_nbytes;
    printf(
        "Rank=%d pid=%d total-groups=%d: %s rank=%d offset=%lu nbytes=%lu\n",
        my_rank_id,
        partition_id,
        total_groups,
        is_send ? "send to" : "recv from",
        peer_rank_id,
        offset,
        nbytes);
  }
}
} // namespace

/**
 * IBGDA send helper with multi-channel dispatch.
 *
 * Supports two scheduling modes based on the incoming ThreadGroup's scope:
 *
 * BLOCK scope (from make_block_group uniform path):
 *   - Blocks are partitioned into numActiveBlocks (up to maxChannelsPerPeer /
 *     channelsPerBlock). Excess blocks per peer early-return.
 *   - Each active block owns a disjoint channel range [baseChannelId,
 *     baseChannelId + channelsPerBlock) and a disjoint data slice.
 *   - Within each block, threads are split into per-channel sub-groups
 *     with independent named barriers (barrier_id = threadChannelId).
 *   - Zero redundancy, proper cross-warp sync via named barriers.
 *   - Backwards compatible: when total_groups==1 behaves identically to today.
 *
 * WARP scope (legacy/fallback):
 *   - Channels are assigned via partition_interleaved across warps
 *   - Each warp independently handles its channel (redundant if multi-warp)
 *
 * @param maxChannelsPerPeer  Max channels (from setup/autotune). 1 = single
 *                            channel, backward compatible.
 */
__device__ __forceinline__ void ibgda_send_helper(
    [[maybe_unused]] ThreadGroup group_per_peer,
    [[maybe_unused]] Transport& transport,
    [[maybe_unused]] const void* sendbuff_d,
    [[maybe_unused]] const ChunkInfo& send_info,
    [[maybe_unused]] Timeout timeout,
    [[maybe_unused]] uint32_t maxChannelsPerPeer = 1) {
#ifdef __CUDA_ARCH__
  if (group_per_peer.scope == SyncScope::BLOCK ||
      group_per_peer.scope == SyncScope::MULTIWARP) {
    // Two-level partitioning: each block gets a disjoint channel range and
    // a disjoint data slice so all active blocks do meaningful work in
    // parallel.
    // When scope == MULTIWARP (warp reserve path), total_groups is always 1,
    // so numActiveBlocks = 1 and behavior is identical to single-block mode.
    uint32_t warpsPerGroup = group_per_peer.group_size / 32;
    uint32_t channelsPerBlock = (warpsPerGroup < maxChannelsPerPeer)
        ? warpsPerGroup
        : maxChannelsPerPeer;
    if (channelsPerBlock == 0)
      channelsPerBlock = 1;
    uint32_t maxBlocks = maxChannelsPerPeer / channelsPerBlock;
    uint32_t numActiveBlocks = (group_per_peer.total_groups < maxBlocks)
        ? group_per_peer.total_groups
        : maxBlocks;
    if (numActiveBlocks == 0)
      numActiveBlocks = 1;

    // Excess blocks (can't contribute disjoint channels) early-return.
    if (group_per_peer.group_id >= numActiveBlocks)
      return;

    // This block's channel range starts at baseChannelId.
    // This block's data slice starts at blockOffset.
    uint32_t baseChannelId = group_per_peer.group_id * channelsPerBlock;
    size_t blockBytes = send_info.nbytes / numActiveBlocks;
    size_t blockOffset = group_per_peer.group_id * blockBytes;
    size_t myBlockBytes = (group_per_peer.group_id == numActiveBlocks - 1)
        ? (send_info.nbytes - blockOffset)
        : blockBytes;

    if (channelsPerBlock <= 1) {
      // Single channel per block: all threads cooperate via named barrier.
      ThreadGroup channel_group = {
          .thread_id_in_group = group_per_peer.thread_id_in_group,
          .group_size = group_per_peer.group_size,
          .group_id = 0,
          .total_groups = 1,
          .scope = SyncScope::MULTIWARP,
          .barrier_id = group_per_peer.barrier_id};
      transport.p2p_ibgda->send(
          channel_group,
          static_cast<char*>(const_cast<void*>(sendbuff_d)) + send_info.offset +
              blockOffset,
          myBlockBytes,
          timeout,
          baseChannelId);
      return;
    }

    // Multi-channel within block: split threads into per-channel sub-groups.
    // barrier_id = group_per_peer.barrier_id + threadChannelId: base peer
    // barrier offset (from warp reserve path) plus within-block channel index.
    uint32_t threadsPerChannel = group_per_peer.group_size / channelsPerBlock;
    uint32_t threadChannelId =
        group_per_peer.thread_id_in_group / threadsPerChannel;
    uint32_t threadInChannel =
        group_per_peer.thread_id_in_group % threadsPerChannel;

    // Threads beyond the last channel's range are idle (remainder threads).
    if (threadChannelId >= channelsPerBlock)
      return;

    ThreadGroup channel_group = {
        .thread_id_in_group = threadInChannel,
        .group_size = threadsPerChannel,
        .group_id = 0,
        .total_groups = 1,
        .scope = SyncScope::MULTIWARP,
        .barrier_id = group_per_peer.barrier_id + threadChannelId};

    size_t bytesPerChannel = myBlockBytes / channelsPerBlock;
    size_t channelOffset = threadChannelId * bytesPerChannel;
    size_t myBytes = (threadChannelId == channelsPerBlock - 1)
        ? (myBlockBytes - channelOffset)
        : bytesPerChannel;

    transport.p2p_ibgda->send(
        channel_group,
        static_cast<char*>(const_cast<void*>(sendbuff_d)) + send_info.offset +
            blockOffset + channelOffset,
        myBytes,
        timeout,
        baseChannelId + threadChannelId);
    return;
  }

  // WARP-scope path: existing partition_interleaved behavior.
  uint32_t numActiveChannels =
      (group_per_peer.total_groups < maxChannelsPerPeer)
      ? group_per_peer.total_groups
      : maxChannelsPerPeer;

  if (numActiveChannels <= 1) {
    transport.p2p_ibgda->send(
        group_per_peer,
        static_cast<char*>(const_cast<void*>(sendbuff_d)) + send_info.offset,
        send_info.nbytes,
        timeout,
        0);
    return;
  }

  auto [channelId, channel_group] =
      group_per_peer.partition_interleaved(numActiveChannels);

  size_t bytesPerChannel = send_info.nbytes / numActiveChannels;
  size_t channelOffset = channelId * bytesPerChannel;
  size_t myBytes = (channelId == numActiveChannels - 1)
      ? (send_info.nbytes - channelOffset)
      : bytesPerChannel;

  transport.p2p_ibgda->send(
      channel_group,
      static_cast<char*>(const_cast<void*>(sendbuff_d)) + send_info.offset +
          channelOffset,
      myBytes,
      timeout,
      channelId);
#endif
}

/**
 * IBGDA recv helper with multi-channel dispatch (symmetric to send).
 * Supports both BLOCK scope (cooperative) and WARP scope (legacy).
 */
__device__ __forceinline__ void ibgda_recv_helper(
    [[maybe_unused]] ThreadGroup group_per_peer,
    [[maybe_unused]] Transport& transport,
    [[maybe_unused]] void* recvbuff_d,
    [[maybe_unused]] const ChunkInfo& recv_info,
    [[maybe_unused]] Timeout timeout,
    [[maybe_unused]] uint32_t maxChannelsPerPeer = 1) {
#ifdef __CUDA_ARCH__
  if (group_per_peer.scope == SyncScope::BLOCK ||
      group_per_peer.scope == SyncScope::MULTIWARP) {
    // Symmetric to ibgda_send_helper: two-level block+thread partitioning.
    uint32_t warpsPerGroup = group_per_peer.group_size / 32;
    uint32_t channelsPerBlock = (warpsPerGroup < maxChannelsPerPeer)
        ? warpsPerGroup
        : maxChannelsPerPeer;
    if (channelsPerBlock == 0)
      channelsPerBlock = 1;
    uint32_t maxBlocks = maxChannelsPerPeer / channelsPerBlock;
    uint32_t numActiveBlocks = (group_per_peer.total_groups < maxBlocks)
        ? group_per_peer.total_groups
        : maxBlocks;
    if (numActiveBlocks == 0)
      numActiveBlocks = 1;

    if (group_per_peer.group_id >= numActiveBlocks)
      return;

    uint32_t baseChannelId = group_per_peer.group_id * channelsPerBlock;
    size_t blockBytes = recv_info.nbytes / numActiveBlocks;
    size_t blockOffset = group_per_peer.group_id * blockBytes;
    size_t myBlockBytes = (group_per_peer.group_id == numActiveBlocks - 1)
        ? (recv_info.nbytes - blockOffset)
        : blockBytes;

    if (channelsPerBlock <= 1) {
      ThreadGroup channel_group = {
          .thread_id_in_group = group_per_peer.thread_id_in_group,
          .group_size = group_per_peer.group_size,
          .group_id = 0,
          .total_groups = 1,
          .scope = SyncScope::MULTIWARP,
          .barrier_id = group_per_peer.barrier_id};
      transport.p2p_ibgda->recv(
          channel_group,
          static_cast<char*>(recvbuff_d) + recv_info.offset + blockOffset,
          myBlockBytes,
          timeout,
          baseChannelId);
      return;
    }

    uint32_t threadsPerChannel = group_per_peer.group_size / channelsPerBlock;
    uint32_t threadChannelId =
        group_per_peer.thread_id_in_group / threadsPerChannel;
    uint32_t threadInChannel =
        group_per_peer.thread_id_in_group % threadsPerChannel;

    if (threadChannelId >= channelsPerBlock)
      return;

    ThreadGroup channel_group = {
        .thread_id_in_group = threadInChannel,
        .group_size = threadsPerChannel,
        .group_id = 0,
        .total_groups = 1,
        .scope = SyncScope::MULTIWARP,
        .barrier_id = group_per_peer.barrier_id + threadChannelId};

    size_t bytesPerChannel = myBlockBytes / channelsPerBlock;
    size_t channelOffset = threadChannelId * bytesPerChannel;
    size_t myBytes = (threadChannelId == channelsPerBlock - 1)
        ? (myBlockBytes - channelOffset)
        : bytesPerChannel;

    transport.p2p_ibgda->recv(
        channel_group,
        static_cast<char*>(recvbuff_d) + recv_info.offset + blockOffset +
            channelOffset,
        myBytes,
        timeout,
        baseChannelId + threadChannelId);
    return;
  }

  // WARP-scope path: existing partition_interleaved behavior.
  uint32_t numActiveChannels =
      (group_per_peer.total_groups < maxChannelsPerPeer)
      ? group_per_peer.total_groups
      : maxChannelsPerPeer;

  if (numActiveChannels <= 1) {
    transport.p2p_ibgda->recv(
        group_per_peer,
        static_cast<char*>(recvbuff_d) + recv_info.offset,
        recv_info.nbytes,
        timeout,
        0);
    return;
  }

  auto [channelId, channel_group] =
      group_per_peer.partition_interleaved(numActiveChannels);

  size_t bytesPerChannel = recv_info.nbytes / numActiveChannels;
  size_t channelOffset = channelId * bytesPerChannel;
  size_t myBytes = (channelId == numActiveChannels - 1)
      ? (recv_info.nbytes - channelOffset)
      : bytesPerChannel;

  transport.p2p_ibgda->recv(
      channel_group,
      static_cast<char*>(recvbuff_d) + recv_info.offset + channelOffset,
      myBytes,
      timeout,
      channelId);
#endif
}

// =============================================================================
// Per-transport-type dispatch helpers
// =============================================================================
// These are called from both the existing uniform path and the new reserve
// path to avoid code duplication.

#ifdef __CUDA_ARCH__

__device__ __forceinline__ void self_copy_helper(
    ThreadGroup group,
    Transport& transport,
    const void* sendbuff_d,
    void* recvbuff_d,
    const ChunkInfo& send_info,
    const ChunkInfo& recv_info) {
  PIPES_DEVICE_CHECK(transport.type == TransportType::SELF);
  PIPES_DEVICE_CHECK(send_info.nbytes == recv_info.nbytes);
  const char* src = static_cast<const char*>(sendbuff_d) + send_info.offset;
  char* dst = static_cast<char*>(recvbuff_d) + recv_info.offset;
  transport.self.put(group, dst, src, send_info.nbytes);
}

__device__ __forceinline__ void nvl_send_helper(
    ThreadGroup group_per_peer,
    Transport& transport,
    const void* sendbuff_d,
    const ChunkInfo& send_info,
    Timeout timeout) {
  transport.p2p_nvl.send(
      group_per_peer,
      static_cast<char*>(const_cast<void*>(sendbuff_d)) + send_info.offset,
      send_info.nbytes,
      timeout);
}

__device__ __forceinline__ void nvl_recv_helper(
    ThreadGroup group_per_peer,
    Transport& transport,
    void* recvbuff_d,
    const ChunkInfo& recv_info,
    Timeout timeout) {
  transport.p2p_nvl.recv(
      group_per_peer,
      static_cast<char*>(recvbuff_d) + recv_info.offset,
      recv_info.nbytes,
      timeout);
}

#endif // __CUDA_ARCH__
/**
 * AllToAllv collective
 *
 * Performs variable-sized all-to-all data exchange among multiple ranks.
 * Each rank sends a potentially different amount of data to every other rank,
 * and receives a potentially different amount of data from every other rank.
 *
 * When reserveConfig.isConfigured() is true, uses a 5-category warp partition
 * (self, NVL-send, NVL-recv, IBGDA-send, IBGDA-recv) with exact warp counts.
 * Otherwise, falls back to the existing uniform partition_interleaved logic.
 *
 * Parameters:
 *   @param recvbuff_d: Device pointer to receive buffer
 *   @param sendbuff_d: Device pointer to send buffer
 *   @param my_rank_id: Current rank ID
 *   @param transports_per_rank: Array of transport objects, one per rank
 *                        (self-transport for my_rank, P2P for others)
 *   @param send_chunk_infos: Array of send chunk metadata, one per destination
 * rank
 *   @param recv_chunk_infos: Array of recv chunk metadata, one per source rank
 *   @param timeout: Timeout configuration
 *   @param reserveConfig: Optional warp reserve configuration for
 *                         category-based warp partition
 */
__device__ __forceinline__ void all_to_allv(
    [[maybe_unused]] void* recvbuff_d,
    [[maybe_unused]] const void* sendbuff_d,
    [[maybe_unused]] int my_rank_id,
    [[maybe_unused]] DeviceSpan<Transport> transports_per_rank,
    [[maybe_unused]] DeviceSpan<ChunkInfo> send_chunk_infos,
    [[maybe_unused]] DeviceSpan<ChunkInfo> recv_chunk_infos,
    [[maybe_unused]] Timeout timeout,
    [[maybe_unused]] WarpReserveDeviceConfig reserveConfig = {},
    uint32_t effectiveTotalWarps = 0) {
#ifdef __CUDA_ARCH__
  const auto nranks = transports_per_rank.size();
  PIPES_DEVICE_CHECK(nranks == send_chunk_infos.size());
  PIPES_DEVICE_CHECK(nranks == recv_chunk_infos.size());

  // Detect whether IBGDA peers are present to choose scheduling mode.
  // IBGDA transport benefits from block-scope scheduling where all threads
  // in a block cooperatively memcpy to staging buffers via named barriers.
  // NVL-only can use warp-scope (each warp handles independent chunks).
  bool hasIbgdaPeers = false;
  for (uint32_t r = 0; r < nranks; ++r) {
    if (transports_per_rank[r].type == TransportType::P2P_IBGDA) {
      hasIbgdaPeers = true;
      break;
    }
  }

  auto group = hasIbgdaPeers ? make_block_group() : make_warp_group();
  if (effectiveTotalWarps > 0) {
    if (hasIbgdaPeers) {
      // Block-group: convert warp count to block count
      uint32_t warpsPerBlock = blockDim.x / 32;
      uint32_t effectiveBlocks =
          (effectiveTotalWarps + warpsPerBlock - 1) / warpsPerBlock;
      group.total_groups = effectiveBlocks;
    } else {
      group.total_groups = effectiveTotalWarps;
    }
  }

  // Single rank case - just do self-copy
  if (nranks == 1) {
    const auto& send_info = send_chunk_infos[my_rank_id];
    const auto& recv_info = recv_chunk_infos[my_rank_id];
    auto& transport = transports_per_rank[my_rank_id];
    PIPES_DEVICE_CHECK(transport.type == TransportType::SELF);
    self_copy_helper(
        group, transport, sendbuff_d, recvbuff_d, send_info, recv_info);
    return;
  }

  // Extract to local pointer to avoid aliasing (see DeviceSpan.cuh:228).
  auto transports = transports_per_rank.data();

  // =========================================================================
  // Category-based warp partition (warp reserve path)
  // =========================================================================
  if (reserveConfig.isConfigured()) {
    // Compute warp_id directly from block/thread indices — not from
    // group.group_id which is blockIdx.x (block index) under make_block_group.
    // Reserve config boundaries are in warp counts, so we need the actual
    // global warp index.
    const uint32_t warpsPerBlock = blockDim.x / 32;
    const uint32_t warp_id = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;

    if (warp_id < reserveConfig.selfEnd) {
      // --- Self-copy category ---
      ThreadGroup self_group = {
          lane, 32, warp_id, reserveConfig.selfEnd, SyncScope::WARP};
      auto& transport = transports[my_rank_id];
      const auto& send_info = send_chunk_infos[my_rank_id];
      const auto& recv_info = recv_chunk_infos[my_rank_id];
      self_copy_helper(
          self_group, transport, sendbuff_d, recvbuff_d, send_info, recv_info);

    } else if (warp_id < reserveConfig.nvlSendEnd) {
      // --- NVL send category ---
      uint32_t local_id = warp_id - reserveConfig.selfEnd;
      uint32_t cat_size = reserveConfig.nvlSendEnd - reserveConfig.selfEnd;
      ThreadGroup cat_group = {lane, 32, local_id, cat_size, SyncScope::WARP};
      auto [peer_idx, peer_group] =
          cat_group.partition_interleaved(reserveConfig.numNvlPeers);
      int peer_rank = reserveConfig.nvlPeerRanks[peer_idx];
      auto& transport = transports[peer_rank];
      const auto& send_info = send_chunk_infos[peer_rank];
      nvl_send_helper(peer_group, transport, sendbuff_d, send_info, timeout);

    } else if (warp_id < reserveConfig.nvlRecvEnd) {
      // --- NVL recv category ---
      uint32_t local_id = warp_id - reserveConfig.nvlSendEnd;
      uint32_t cat_size = reserveConfig.nvlRecvEnd - reserveConfig.nvlSendEnd;
      ThreadGroup cat_group = {lane, 32, local_id, cat_size, SyncScope::WARP};
      auto [peer_idx, peer_group] =
          cat_group.partition_interleaved(reserveConfig.numNvlPeers);
      int peer_rank = reserveConfig.nvlPeerRanks[peer_idx];
      auto& transport = transports[peer_rank];
      const auto& recv_info = recv_chunk_infos[peer_rank];
      nvl_recv_helper(peer_group, transport, recvbuff_d, recv_info, timeout);

    } else if (
        warp_id >= reserveConfig.ibgdaSendBase &&
        warp_id < reserveConfig.ibgdaSendEnd) {
      // --- IBGDA send category (block-aligned) ---
      // With block-aligned boundaries, all warps for a peer are guaranteed
      // to be in the same block, enabling MULTIWARP named barrier sync.
      uint32_t local_id = warp_id - reserveConfig.ibgdaSendBase;
      uint32_t cat_size =
          reserveConfig.ibgdaSendEnd - reserveConfig.ibgdaSendBase;
      uint32_t warpsPerPeer = cat_size / reserveConfig.numIbgdaPeers;
      uint32_t peer_idx = local_id / warpsPerPeer;
      uint32_t warpInPeer = local_id % warpsPerPeer;

      int peer_rank = reserveConfig.ibgdaPeerRanks[peer_idx];
      auto& transport = transports[peer_rank];
      const auto& send_info = send_chunk_infos[peer_rank];

      if (warpsPerPeer > 1 && warpsPerPeer <= warpsPerBlock) {
        // Multiple warps per peer, all fit in one block — MULTIWARP
        // cooperative memcpy. barrier_id = block-local peer index
        // (unique per peer within block, IBGDA category is block-aligned).
        uint32_t peerInBlock = (warp_id % warpsPerBlock) / warpsPerPeer;
        ThreadGroup peer_group = {
            .thread_id_in_group = warpInPeer * 32 + lane,
            .group_size = warpsPerPeer * 32,
            .group_id = 0,
            .total_groups = 1,
            .scope = SyncScope::MULTIWARP,
            .barrier_id = peerInBlock};
        ibgda_send_helper(
            peer_group,
            transport,
            sendbuff_d,
            send_info,
            timeout,
            reserveConfig.maxChannelsPerPeer);
      } else {
        // Single warp per peer — WARP scope, no cooperation needed.
        // (warpsPerPeer > warpsPerBlock is rejected by resolveWarpReserve.)
        ThreadGroup peer_group = {
            lane, 32, warpInPeer, warpsPerPeer, SyncScope::WARP};
        ibgda_send_helper(
            peer_group,
            transport,
            sendbuff_d,
            send_info,
            timeout,
            reserveConfig.maxChannelsPerPeer);
      }

    } else if (
        warp_id >= reserveConfig.ibgdaRecvBase &&
        warp_id < reserveConfig.ibgdaRecvEnd &&
        reserveConfig.numIbgdaPeers > 0) {
      // --- IBGDA recv category (block-aligned, symmetric to send) ---
      uint32_t local_id = warp_id - reserveConfig.ibgdaRecvBase;
      uint32_t cat_size =
          reserveConfig.ibgdaRecvEnd - reserveConfig.ibgdaRecvBase;
      uint32_t warpsPerPeer = cat_size / reserveConfig.numIbgdaPeers;
      uint32_t peer_idx = local_id / warpsPerPeer;
      uint32_t warpInPeer = local_id % warpsPerPeer;

      int peer_rank = reserveConfig.ibgdaPeerRanks[peer_idx];
      auto& transport = transports[peer_rank];
      const auto& recv_info = recv_chunk_infos[peer_rank];

      if (warpsPerPeer > 1 && warpsPerPeer <= warpsPerBlock) {
        uint32_t peerInBlock = (warp_id % warpsPerBlock) / warpsPerPeer;
        ThreadGroup peer_group = {
            .thread_id_in_group = warpInPeer * 32 + lane,
            .group_size = warpsPerPeer * 32,
            .group_id = 0,
            .total_groups = 1,
            .scope = SyncScope::MULTIWARP,
            .barrier_id = peerInBlock};
        ibgda_recv_helper(
            peer_group,
            transport,
            recvbuff_d,
            recv_info,
            timeout,
            reserveConfig.maxChannelsPerPeer);
      } else {
        // Single warp per peer — WARP scope, no cooperation needed.
        // (warpsPerPeer > warpsPerBlock is rejected by resolveWarpReserve.)
        ThreadGroup peer_group = {
            lane, 32, warpInPeer, warpsPerPeer, SyncScope::WARP};
        ibgda_recv_helper(
            peer_group,
            transport,
            recvbuff_d,
            recv_info,
            timeout,
            reserveConfig.maxChannelsPerPeer);
      }
    }
    // Padding warps (between nvlRecvEnd..ibgdaSendBase or
    // ibgdaSendEnd..ibgdaRecvBase) and excess warps beyond ibgdaRecvEnd
    // fall through all branches and return here.
    return;
  }

  // =========================================================================
  // Uniform partition path (existing behavior, backward compatible)
  // =========================================================================

  // 1. First partition by SEND/RECV using interleaved partitioning
  // partition_id: 0 = send, 1 = recv
  auto [partition_id, send_recv_group] = group.partition_interleaved(2);

  // 2. Capped partition by PEERS: cap concurrent peers at available warps
  // so partition_interleaved never exceeds total_groups. When nranks >
  // available warps, each warp iterates over multiple peers in batches.
  uint32_t avail = send_recv_group.total_groups;
  uint32_t concurrent = (nranks < avail) ? nranks : avail;
  auto [slot_id, group_per_peer] =
      send_recv_group.partition_interleaved(concurrent);

  for (uint32_t batch = 0; batch * concurrent < nranks; batch++) {
    uint32_t peer_rank_id = slot_id + batch * concurrent;
    if (peer_rank_id >= nranks)
      break;

    const auto& send_info = send_chunk_infos[peer_rank_id];
    const auto& recv_info = recv_chunk_infos[peer_rank_id];
    const bool is_send = (partition_id == 0);
    const size_t nbytes = is_send ? send_info.nbytes : recv_info.nbytes;

    // Nothing to send/recv for this peer
    if (nbytes == 0) {
      continue;
    }

    if (peer_rank_id == static_cast<uint32_t>(my_rank_id)) {
      // Self partition
      auto& transport = transports[my_rank_id];

#ifdef DEBUG_ALLTOALLV
      if (group_per_peer.is_global_leader()) {
        printPerPeerOperation(
            my_rank_id,
            peer_rank_id,
            partition_id,
            group_per_peer.total_groups,
            send_info.offset,
            recv_info.offset,
            send_info.nbytes,
            recv_info.nbytes);
      }
#endif

      // Only one partition is active for self-copy
      if (partition_id == 0) {
        self_copy_helper(
            group_per_peer,
            transport,
            sendbuff_d,
            recvbuff_d,
            send_info,
            recv_info);
      }
      continue;
    }

    auto& transport = transports[peer_rank_id];

#ifdef DEBUG_ALLTOALLV
    if (group_per_peer.is_global_leader()) {
      printPerPeerOperation(
          my_rank_id,
          peer_rank_id,
          partition_id,
          group_per_peer.total_groups,
          send_info.offset,
          recv_info.offset,
          send_info.nbytes,
          recv_info.nbytes);
    }
#endif

    // Dispatch based on transport type
    if (transport.type == TransportType::P2P_NVL) {
      if (is_send) {
        nvl_send_helper(
            group_per_peer, transport, sendbuff_d, send_info, timeout);
      } else {
        nvl_recv_helper(
            group_per_peer, transport, recvbuff_d, recv_info, timeout);
      }
    } else if (transport.type == TransportType::P2P_IBGDA) {
      uint32_t maxCh = transport.p2p_ibgda->getMaxChannelsPerPeer();
      if (is_send) {
        ibgda_send_helper(
            group_per_peer, transport, sendbuff_d, send_info, timeout, maxCh);
      } else {
        ibgda_recv_helper(
            group_per_peer, transport, recvbuff_d, recv_info, timeout, maxCh);
      }
    }
  }

#endif
}

} // namespace comms::pipes
