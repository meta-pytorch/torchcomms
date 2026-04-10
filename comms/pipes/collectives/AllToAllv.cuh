// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

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
  if (group_per_peer.scope == SyncScope::BLOCK) {
    // Two-level partitioning: each block gets a disjoint channel range and
    // a disjoint data slice so all active blocks do meaningful work in
    // parallel.
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
      // Single channel per block: all threads cooperate via named barrier 0.
      ThreadGroup channel_group = {
          .thread_id_in_group = group_per_peer.thread_id_in_group,
          .group_size = group_per_peer.group_size,
          .group_id = 0,
          .total_groups = 1,
          .scope = SyncScope::MULTIWARP,
          .barrier_id = 0};
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
    // barrier_id = threadChannelId (named barriers are block-local, no
    // collision).
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
        .barrier_id = threadChannelId};

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
  if (group_per_peer.scope == SyncScope::BLOCK) {
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
          .barrier_id = 0};
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
        .barrier_id = threadChannelId};

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

/**
 * AllToAllv collective communication primitive.
 *
 * Performs variable-sized all-to-all data exchange among multiple ranks.
 * Each rank sends a potentially different amount of data to every other rank,
 * and receives a potentially different amount of data from every other rank.
 *
 * Algorithm:
 * 1. First weighted partition: Distribute warps across ranks based on total
 *    communication workload (send+recv bytes for peers, send only for self)
 * 2. For self-rank: Perform local memory copy within the same GPU
 * 3. For peer ranks: Second weighted partition to split warps between send
 *    and recv operations based on their respective data sizes
 * 4. Execute send or recv using P2P NVL or P2P IBGDA transport
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
 *
 * Requirements:
 * - Must be called from device code with sufficient threads
 * - transports_per_rank.size() == send_chunk_infos.size() ==
 *   recv_chunk_infos.size()
 * - send_chunk_infos[i].nbytes == recv_chunk_infos[i].nbytes for i ==
 *   my_rank_id
 * - Max 8 ranks supported (stack-allocated weights)
 */
__device__ __forceinline__ void all_to_allv(
    [[maybe_unused]] void* recvbuff_d,
    [[maybe_unused]] const void* sendbuff_d,
    [[maybe_unused]] int my_rank_id,
    [[maybe_unused]] DeviceSpan<Transport> transports_per_rank,
    [[maybe_unused]] DeviceSpan<ChunkInfo> send_chunk_infos,
    [[maybe_unused]] DeviceSpan<ChunkInfo> recv_chunk_infos,
    [[maybe_unused]] Timeout timeout
    // all arguments below will eventually come from communicator
) {
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

  // Single rank case - just do self-copy
  if (nranks == 1) {
    const auto& send_info = send_chunk_infos[my_rank_id];
    const auto& recv_info = recv_chunk_infos[my_rank_id];
    const char* src = static_cast<const char*>(sendbuff_d) + send_info.offset;
    char* dst = static_cast<char*>(recvbuff_d) + recv_info.offset;

    auto& transport = transports_per_rank[my_rank_id];
    PIPES_DEVICE_CHECK(transport.type == TransportType::SELF);
    transport.self.put(group, dst, src, send_info.nbytes);
    return;
  }

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

  // Extract to local pointer to avoid aliasing (see DeviceSpan.cuh:228).
  auto transports = transports_per_rank.data();

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
      PIPES_DEVICE_CHECK(transport.type == TransportType::SELF);
      PIPES_DEVICE_CHECK(send_info.nbytes == recv_info.nbytes);

      const char* src = static_cast<const char*>(sendbuff_d) + send_info.offset;
      char* dst = static_cast<char*>(recvbuff_d) + recv_info.offset;

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
        transport.self.put(group_per_peer, dst, src, send_info.nbytes);
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
        transport.p2p_nvl.send(
            group_per_peer,
            static_cast<char*>(const_cast<void*>(sendbuff_d)) +
                send_info.offset,
            send_info.nbytes,
            timeout);
      } else {
        transport.p2p_nvl.recv(
            group_per_peer,
            static_cast<char*>(recvbuff_d) + recv_info.offset,
            recv_info.nbytes,
            timeout);
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
