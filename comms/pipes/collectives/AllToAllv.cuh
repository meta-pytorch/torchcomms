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
  auto group = make_warp_group();
  const auto nranks = transports_per_rank.size();
  PIPES_DEVICE_CHECK(nranks == send_chunk_infos.size());
  PIPES_DEVICE_CHECK(nranks == recv_chunk_infos.size());

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
      if (is_send) {
        transport.p2p_ibgda->send(
            group_per_peer,
            static_cast<char*>(const_cast<void*>(sendbuff_d)) +
                send_info.offset,
            send_info.nbytes,
            timeout);
      } else {
        transport.p2p_ibgda->recv(
            group_per_peer,
            static_cast<char*>(recvbuff_d) + recv_info.offset,
            recv_info.nbytes,
            timeout);
      }
    }
  }

#endif
}

} // namespace comms::pipes
