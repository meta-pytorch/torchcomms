// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdio>

#include "comms/pipes/DeviceCheck.cuh"
#include "comms/pipes/collectives/broadcast/BroadcastContext.cuh"

namespace comms::pipes::collectives {

// ============================================================================
// Debug Helpers (compiled only with DEBUG_BROADCAST defined)
// ============================================================================

namespace detail {

#ifdef DEBUG_BROADCAST
__device__ __forceinline__ void printBroadcastOp(
    const char* topology,
    int my_rank,
    int root_rank,
    int peer_rank,
    bool is_send,
    std::size_t nbytes) {
  printf(
      "[%s] Rank=%d root=%d: %s rank=%d nbytes=%llu\n",
      topology,
      my_rank,
      root_rank,
      is_send ? "send to" : "recv from",
      peer_rank,
      static_cast<unsigned long long>(nbytes));
}
#endif

} // namespace detail

// ============================================================================
// FlatTag Implementation
// ============================================================================

/**
 * Flat-tree (star) topology: root sends directly to all non-root ranks.
 *
 * Algorithm:
 * - Root: Partitions warps across (nranks-1) peers for parallel sends.
 *   Uses contiguous partitioning for better cache locality.
 * - Non-root: All warps collaborate on receiving from root.
 *
 * Performance characteristics:
 * - Latency: O(1) - single hop from root to all peers
 * - Root bandwidth: O(N) - must send to all peers
 * - Best for: Small/medium messages where latency matters
 */
__device__ __forceinline__ void TopologyTraits<FlatTag>::execute(
    BroadcastContext& ctx) {
  if (ctx.is_root()) {
    // Partition warps across (nranks - 1) peers
    auto [peer_idx, peer_group] = ctx.group.partition(ctx.nranks - 1);

    // Map peer_idx [0, nranks-2] to actual peer rank, skipping root
    // Example for root=2, nranks=4:
    //   peer_idx 0 -> rank 0
    //   peer_idx 1 -> rank 1
    //   peer_idx 2 -> rank 3 (skip root at 2)
    int peer_rank = static_cast<int>(
        peer_idx < static_cast<uint32_t>(ctx.root_rank_id) ? peer_idx
                                                           : peer_idx + 1);

    auto& transport = ctx.transports[peer_rank];
    PIPES_DEVICE_CHECK(transport.type == TransportType::P2P_NVL);

#ifdef DEBUG_BROADCAST
    if (peer_group.is_global_leader()) {
      detail::printBroadcastOp(
          "Flat",
          ctx.my_rank_id,
          ctx.root_rank_id,
          peer_rank,
          true,
          ctx.nbytes);
    }
#endif

    transport.p2p_nvl.send(peer_group, ctx.buff, ctx.nbytes);
  } else {
    auto& transport = ctx.transports[ctx.root_rank_id];
    PIPES_DEVICE_CHECK(transport.type == TransportType::P2P_NVL);

#ifdef DEBUG_BROADCAST
    if (ctx.group.is_global_leader()) {
      detail::printBroadcastOp(
          "Flat",
          ctx.my_rank_id,
          ctx.root_rank_id,
          ctx.root_rank_id,
          false,
          ctx.nbytes);
    }
#endif

    transport.p2p_nvl.recv(ctx.group, ctx.buff, ctx.nbytes);
  }
}

// ============================================================================
// RingTag Implementation
// ============================================================================

/**
 * Ring topology: data flows sequentially around a ring.
 *   root -> rank1 -> rank2 -> ... -> rank(N-1)
 *
 * Algorithm:
 * - Each rank receives from predecessor (if not root)
 * - Then sends to successor (if not last rank)
 * - Uses virtual rank mapping so root is always virtual_rank=0
 *
 * Performance characteristics:
 * - Latency: O(N) - N-1 sequential hops
 * - Bandwidth per link: O(1) - each link transfers full message once
 * - Transport's internal pipelining handles efficiency within each transfer
 * - Best for: Large messages where bandwidth > latency
 *
 * Note: Ring is NOT optimal for pure broadcast (use Flat or BinomialTree).
 * It's included for completeness and as a building block for other collectives.
 */
__device__ __forceinline__ void TopologyTraits<RingTag>::execute(
    BroadcastContext& ctx) {
  // Compute ring neighbors using virtual ranks
  int next_virtual = (ctx.virtual_rank + 1) % ctx.nranks;
  int prev_virtual = (ctx.virtual_rank - 1 + ctx.nranks) % ctx.nranks;
  int next_rank = ctx.actual_rank(next_virtual);
  int prev_rank = ctx.actual_rank(prev_virtual);

  // Validate transport types for neighbors we'll use
  if (ctx.virtual_rank < ctx.nranks - 1) {
    PIPES_DEVICE_CHECK(
        ctx.transports[next_rank].type == TransportType::P2P_NVL);
  }
  if (ctx.virtual_rank > 0) {
    PIPES_DEVICE_CHECK(
        ctx.transports[prev_rank].type == TransportType::P2P_NVL);
  }

  // Phase 1: Receive from predecessor (if not root)
  if (ctx.virtual_rank > 0) {
#ifdef DEBUG_BROADCAST
    if (ctx.group.is_leader()) {
      detail::printBroadcastOp(
          "Ring",
          ctx.my_rank_id,
          ctx.root_rank_id,
          prev_rank,
          false,
          ctx.nbytes);
    }
#endif
    ctx.transports[prev_rank].p2p_nvl.recv(ctx.group, ctx.buff, ctx.nbytes);
  }

  // Phase 2: Send to successor (if not last rank)
  if (ctx.virtual_rank < ctx.nranks - 1) {
#ifdef DEBUG_BROADCAST
    if (ctx.group.is_leader()) {
      detail::printBroadcastOp(
          "Ring",
          ctx.my_rank_id,
          ctx.root_rank_id,
          next_rank,
          true,
          ctx.nbytes);
    }
#endif
    ctx.transports[next_rank].p2p_nvl.send(ctx.group, ctx.buff, ctx.nbytes);
  }
}

// ============================================================================
// BinomialTreeTag Implementation
// ============================================================================

/**
 * Binomial tree topology: O(log N) rounds of parallel transfers.
 *
 * Algorithm:
 * - Round r: ranks with virtual_rank < 2^r have data
 * - Each such rank sends to virtual_rank + 2^r (if that rank exists)
 * - Parallelism doubles each round
 *
 * Example for 8 ranks:
 *   Round 0: 0->1                     (1 transfer)
 *   Round 1: 0->2, 1->3               (2 transfers in parallel)
 *   Round 2: 0->4, 1->5, 2->6, 3->7   (4 transfers in parallel)
 *
 * Performance characteristics:
 * - Latency: O(log N) rounds
 * - Root bandwidth: O(log N) - only sends log(N) times
 * - Total bandwidth: distributed across all nodes
 * - Best for: Large messages where root bandwidth is bottleneck
 */
__device__ __forceinline__ void TopologyTraits<BinomialTreeTag>::execute(
    BroadcastContext& ctx) {
  // Calculate number of rounds = ceil(log2(nranks))
  int num_rounds = 0;
  for (int n = 1; n < ctx.nranks; n <<= 1) {
    num_rounds++;
  }

  // Round-major ordering: transfer entire message each round
  // This allows transport layer to use full pipeline depth
  for (int round = 0; round < num_rounds; ++round) {
    int distance = 1 << round;

    if (ctx.virtual_rank < distance) {
      // I have data - check if I need to send
      int send_to_virtual = ctx.virtual_rank + distance;

      if (send_to_virtual < ctx.nranks) {
        int send_to_actual = ctx.actual_rank(send_to_virtual);

        auto& transport = ctx.transports[send_to_actual];
        PIPES_DEVICE_CHECK(transport.type == TransportType::P2P_NVL);

#ifdef DEBUG_BROADCAST
        if (ctx.group.is_leader()) {
          detail::printBroadcastOp(
              "BinomialTree",
              ctx.my_rank_id,
              ctx.root_rank_id,
              send_to_actual,
              true,
              ctx.nbytes);
        }
#endif

        transport.p2p_nvl.send(ctx.group, ctx.buff, ctx.nbytes);
      }
    } else if (ctx.virtual_rank < 2 * distance) {
      // I need to receive data this round
      int recv_from_virtual = ctx.virtual_rank - distance;
      int recv_from_actual = ctx.actual_rank(recv_from_virtual);

      auto& transport = ctx.transports[recv_from_actual];
      PIPES_DEVICE_CHECK(transport.type == TransportType::P2P_NVL);

#ifdef DEBUG_BROADCAST
      if (ctx.group.is_leader()) {
        detail::printBroadcastOp(
            "BinomialTree",
            ctx.my_rank_id,
            ctx.root_rank_id,
            recv_from_actual,
            false,
            ctx.nbytes);
      }
#endif

      transport.p2p_nvl.recv(ctx.group, ctx.buff, ctx.nbytes);
    }
    // Ranks >= 2*distance don't participate in this round
  }
}

} // namespace comms::pipes::collectives
