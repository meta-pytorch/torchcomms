// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes {

namespace {
/**
 * Debug helper to print all_gather communication information.
 * Automatically detects self-copy vs peer communication based on my_rank ==
 * peer_rank.
 */
__device__ __forceinline__ void printAllGatherOperation(
    int my_rank_id,
    int peer_rank_id,
    int partition_id,
    uint32_t total_groups,
    size_t sendcount,
    size_t recv_offset) {
  if (my_rank_id == peer_rank_id) {
    // Self-copy
    printf(
        "Rank=%d pid=%d total-groups=%u: self-copy sendcount=%llu recv-offset=%llu\n",
        my_rank_id,
        partition_id,
        total_groups,
        static_cast<unsigned long long>(sendcount),
        static_cast<unsigned long long>(recv_offset));
  } else {
    // Peer communication
    bool is_send = (partition_id == 0);
    printf(
        "Rank=%d pid=%d total-groups=%u: %s rank=%d sendcount=%llu recv-offset=%llu\n",
        my_rank_id,
        partition_id,
        total_groups,
        is_send ? "send to" : "recv from",
        peer_rank_id,
        static_cast<unsigned long long>(sendcount),
        static_cast<unsigned long long>(recv_offset));
  }
}
} // namespace

/**
 * AllGather collective communication primitive.
 *
 * Gathers data from all ranks, each rank contributes sendcount bytes, and after
 * the operation, each rank has nranks * sendcount bytes containing all ranks'
 * data.
 *
 * Algorithm (Direct Pattern):
 * 1. First partition: Distribute warps into send (0) and recv (1) groups
 *    using interleaved partitioning for better load balancing.
 * 2. Second partition: Further distribute each group across peer ranks
 *    using interleaved partitioning.
 * 3. For self-rank: Perform local memory copy (sendbuff -> recvbuff at
 *    my_rank * sendcount offset)
 * 4. For peer ranks:
 *    - Send partition: Send my sendbuff to peer
 *    - Recv partition: Receive from peer into recvbuff at peer_rank *
 *      sendcount offset
 *
 * @param recvbuff_d Device pointer to receive buffer (nranks * sendcount
 * bytes)
 * @param sendbuff_d Device pointer to send buffer (sendcount bytes)
 * @param sendcount Number of bytes each rank contributes
 * @param my_rank_id Current rank ID
 * @param transports_per_rank Array of transport objects, one per rank
 *                            (self-transport for my_rank, P2P for others)
 * @param timeout Optional timeout for wait operations
 *
 * Buffer Layout:
 *   sendbuff_d: [my_data]
 *               sendcount bytes
 *   recvbuff_d: [data_from_rank0 | data_from_rank1 | ... | data_from_rankN]
 *               Each segment is exactly sendcount bytes.
 *
 * Requirements:
 * - Must be called from device code with sufficient threads
 * - Max 8 ranks supported (stack-allocated arrays)
 */
__device__ __forceinline__ void all_gather(
    void* recvbuff_d,
    const void* sendbuff_d,
    std::size_t sendcount,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    Timeout timeout = Timeout()) {
#ifdef __CUDA_ARCH__
  // Start the timeout timer - must be called once before any wait operations
  timeout.start();

  auto group = make_warp_group();
  const auto nranks = transports_per_rank.size();

  // Single rank case - just do self-copy
  if (nranks == 1) {
    char* dst = static_cast<char*>(recvbuff_d);
    const char* src = static_cast<const char*>(sendbuff_d);

    auto& transport = transports_per_rank[my_rank_id];
    assert(transport.type == TransportType::SELF);
    transport.self.put(group, dst, src, sendcount);
    return;
  }

  // 1. First partition by SEND/RECV using interleaved partitioning
  // partition_id: 0 = send, 1 = recv
  auto [partition_id, send_recv_group] = group.partition_interleaved(2);

  // 2. Then partition by PEERS using interleaved partitioning
  // Spreads blocks for same peer across SM space for better load balancing
  auto [peer_rank_id, group_per_peer] =
      send_recv_group.partition_interleaved(nranks);

  // Calculate receive offset for this peer's data
  const std::size_t recv_offset = peer_rank_id * sendcount;

  if (peer_rank_id == my_rank_id) {
    // Self partition - both send and recv groups participate in copying
    auto& transport = transports_per_rank[my_rank_id];
    assert(transport.type == TransportType::SELF);

    const char* src = static_cast<const char*>(sendbuff_d);
    char* dst = static_cast<char*>(recvbuff_d) + recv_offset;

#ifdef DEBUG_ALLGATHER
    if (group_per_peer.is_global_leader()) {
      printAllGatherOperation(
          my_rank_id,
          peer_rank_id,
          partition_id,
          group_per_peer.total_groups,
          sendcount,
          recv_offset);
    }
#endif

    transport.self.put(group_per_peer, dst, src, sendcount);
    return;
  }

  // Peer communication
  // Extract to local pointer to avoid aliasing: compiler can't prove that
  // operations on transport won't modify transports_per_rank.data_, forcing
  // reloads. Local variable is provably independent. See DeviceSpan.cuh:228.
  auto transports = transports_per_rank.data();
  auto& transport = transports[peer_rank_id];
  assert(transport.type == TransportType::P2P_NVL);

#ifdef DEBUG_ALLGATHER
  if (group_per_peer.is_global_leader()) {
    printAllGatherOperation(
        my_rank_id,
        peer_rank_id,
        partition_id,
        group_per_peer.total_groups,
        sendcount,
        recv_offset);
  }
#endif

  // Perform peer send/recv based on partition_id from first partition
  // Key difference from AllToAllv:
  // - Send: All peers receive MY sendbuff (same source for all peers)
  // - Recv: I receive peer's sendbuff at offset peer_rank_id * sendcount
  bool is_send = (partition_id == 0);
  if (is_send) {
    // Send my local data to peer
    // Note: All sends use the same source buffer (sendbuff_d)
    transport.p2p_nvl.send(
        group_per_peer,
        static_cast<char*>(const_cast<void*>(sendbuff_d)),
        sendcount,
        0, // call_index
        timeout);
  } else {
    // Receive peer's data into my recvbuff at appropriate offset
    transport.p2p_nvl.recv(
        group_per_peer,
        static_cast<char*>(recvbuff_d) + recv_offset,
        sendcount,
        0, // call_index
        timeout);
  }

#endif
}

} // namespace comms::pipes
