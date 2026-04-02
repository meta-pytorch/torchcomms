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
 * that category). ibgdaRecvEnd is implicit (= total warps from thread count).
 *
 * When isConfigured() returns false, the kernel falls back to the existing
 * uniform partition_interleaved logic for full backward compatibility.
 */
struct WarpReserveDeviceConfig {
  uint32_t selfEnd = 0;
  uint32_t nvlSendEnd = 0;
  uint32_t nvlRecvEnd = 0;
  uint32_t ibgdaSendEnd = 0;

  const int* nvlPeerRanks = nullptr;
  uint32_t numNvlPeers = 0;
  const int* ibgdaPeerRanks = nullptr;
  uint32_t numIbgdaPeers = 0;

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

__device__ __forceinline__ void ibgda_send_helper(
    ThreadGroup group_per_peer,
    Transport& transport,
    const void* sendbuff_d,
    const ChunkInfo& send_info,
    Timeout timeout) {
  transport.p2p_ibgda->send(
      group_per_peer,
      static_cast<char*>(const_cast<void*>(sendbuff_d)) + send_info.offset,
      send_info.nbytes,
      timeout);
}

__device__ __forceinline__ void ibgda_recv_helper(
    ThreadGroup group_per_peer,
    Transport& transport,
    void* recvbuff_d,
    const ChunkInfo& recv_info,
    Timeout timeout) {
  transport.p2p_ibgda->recv(
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
    [[maybe_unused]] WarpReserveDeviceConfig reserveConfig = {}
    // all arguments below will eventually come from communicator
) {
#ifdef __CUDA_ARCH__
  auto group = make_warp_group();
  const auto nranks = transports_per_rank.size();
  PIPES_DEVICE_CHECK(nranks == send_chunk_infos.size());
  PIPES_DEVICE_CHECK(nranks == recv_chunk_infos.size());

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

  // =========================================================================
  // Category-based warp partition (warp reserve path)
  // =========================================================================
  if (reserveConfig.isConfigured()) {
    const uint32_t warp_id = group.group_id;
    const uint32_t lane = threadIdx.x % 32;
    auto transports = transports_per_rank.data();

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

    } else if (warp_id < reserveConfig.ibgdaSendEnd) {
      // --- IBGDA send category ---
      uint32_t local_id = warp_id - reserveConfig.nvlRecvEnd;
      uint32_t cat_size = reserveConfig.ibgdaSendEnd - reserveConfig.nvlRecvEnd;
      ThreadGroup cat_group = {lane, 32, local_id, cat_size, SyncScope::WARP};
      auto [peer_idx, peer_group] =
          cat_group.partition_interleaved(reserveConfig.numIbgdaPeers);
      int peer_rank = reserveConfig.ibgdaPeerRanks[peer_idx];
      auto& transport = transports[peer_rank];
      const auto& send_info = send_chunk_infos[peer_rank];
      ibgda_send_helper(peer_group, transport, sendbuff_d, send_info, timeout);

    } else if (reserveConfig.numIbgdaPeers > 0) {
      // --- IBGDA recv category (remaining warps) ---
      uint32_t local_id = warp_id - reserveConfig.ibgdaSendEnd;
      uint32_t cat_size = group.total_groups - reserveConfig.ibgdaSendEnd;
      ThreadGroup cat_group = {lane, 32, local_id, cat_size, SyncScope::WARP};
      auto [peer_idx, peer_group] =
          cat_group.partition_interleaved(reserveConfig.numIbgdaPeers);
      int peer_rank = reserveConfig.ibgdaPeerRanks[peer_idx];
      auto& transport = transports[peer_rank];
      const auto& recv_info = recv_chunk_infos[peer_rank];
      ibgda_recv_helper(peer_group, transport, recvbuff_d, recv_info, timeout);
    }
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
      if (is_send) {
        ibgda_send_helper(
            group_per_peer, transport, sendbuff_d, send_info, timeout);
      } else {
        ibgda_recv_helper(
            group_per_peer, transport, recvbuff_d, recv_info, timeout);
      }
    }
  }

#endif
}

} // namespace comms::pipes
