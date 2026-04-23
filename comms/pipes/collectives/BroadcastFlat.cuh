// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "comms/pipes/DeviceCheck.cuh"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes::collectives {

namespace {
/**
 * Debug helper to print broadcast operation information.
 */
__device__ __forceinline__ void printBroadcastOperation(
    int my_rank_id,
    int root_rank_id,
    int peer_rank_id,
    bool is_send,
    uint32_t total_groups,
    std::size_t nbytes) {
  printf(
      "Rank=%d root=%d total-groups=%d: %s rank=%d nbytes=%lu\n",
      my_rank_id,
      root_rank_id,
      total_groups,
      is_send ? "send to" : "recv from",
      peer_rank_id,
      nbytes);
}
} // namespace

/**
 * Broadcast collective communication primitive.
 *
 * Broadcasts data from root rank to all other ranks using a flat-tree (star)
 * algorithm where the root sends directly to each non-root rank in parallel.
 *
 * Algorithm:
 * 1. Single rank case: No-op (root already has data)
 * 2. Zero bytes case: No-op
 * 3. Root rank: Partitions warps across (nranks-1) peers using contiguous
 *    partitioning for better cache locality
 * 4. Non-root ranks: All warps collaborate on receiving data from root
 *
 * Parameters:
 *   @param buff_d: Device buffer pointer
 *                  - For root: contains source data to broadcast
 *                  - For non-root: receives broadcast data
 *   @param my_rank_id: Current rank ID
 *   @param root_rank_id: Rank ID of the broadcast source
 *   @param transports_per_rank: Array of transport objects, one per rank
 *                               (self-transport for my_rank, P2P for others)
 *   @param nbytes: Number of bytes to broadcast
 *
 * Requirements:
 * - Must be called from device code with sufficient threads
 * - All ranks must call with same root_rank_id and nbytes
 * - Max 8 ranks supported
 * - transports_per_rank[my_rank_id].type must be SELF
 * - transports_per_rank[i].type must be P2P_NVL for i != my_rank_id
 */
__device__ __forceinline__ void broadcast_flat(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes) {
#ifdef __CUDA_ARCH__
  const auto nranks = transports_per_rank.size();

  // Single rank case: no-op (root already has data)
  if (nranks == 1) {
    return;
  }

  // Zero-byte broadcast: no-op
  if (nbytes == 0) {
    return;
  }

  auto group = make_warp_group();

  // Extract raw pointer to avoid aliasing issues.
  // See "PERFORMANCE NOTE: Lambda Capture and Aliasing"
  // in DeviceSpan.cuh for explanation.
  auto transports = transports_per_rank.data();

  bool is_root = (my_rank_id == root_rank_id);

  if (is_root) {
    // Root: partition warps across (nranks - 1) peers for parallel sends
    // Use contiguous partitioning: adjacent blocks handle the same peer.
    // Profiling shows this provides better cache locality and 39% better
    // performance than interleaved partitioning for 1MB messages.
    auto [peer_idx, peer_group] = group.partition(nranks - 1);

    // Map peer_idx [0, nranks-2] to actual peer rank, skipping root
    // For root_rank_id = 2 and nranks = 4:
    //   peer_idx 0 → rank 0
    //   peer_idx 1 → rank 1
    //   peer_idx 2 → rank 3 (skip root at index 2)
    int peer_rank = static_cast<int>(
        peer_idx < static_cast<uint32_t>(root_rank_id) ? peer_idx
                                                       : peer_idx + 1);

    auto& transport = transports[peer_rank];
    PIPES_DEVICE_CHECK(transport.type == TransportType::P2P_NVL);

#ifdef DEBUG_BROADCAST
    if (peer_group.is_global_leader()) {
      printBroadcastOperation(
          my_rank_id,
          root_rank_id,
          peer_rank,
          true, // is_send
          peer_group.total_groups,
          nbytes);
    }
#endif

    transport.p2p_nvl.send(peer_group, static_cast<char*>(buff_d), nbytes);
  } else {
    // Non-root: all warps collaborate on receiving from root
    auto& transport = transports[root_rank_id];
    PIPES_DEVICE_CHECK(transport.type == TransportType::P2P_NVL);

#ifdef DEBUG_BROADCAST
    if (group.is_global_leader()) {
      printBroadcastOperation(
          my_rank_id,
          root_rank_id,
          root_rank_id,
          false, // is_send
          group.total_groups,
          nbytes);
    }
#endif

    transport.p2p_nvl.recv(group, static_cast<char*>(buff_d), nbytes);
  }
#endif
}

} // namespace comms::pipes::collectives
