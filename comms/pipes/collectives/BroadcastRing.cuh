// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes::collectives {

namespace {

// Chunk size for pipelined ring broadcast (128KB)
// Smaller chunks enable better inter-hop pipelining
constexpr std::size_t kRingChunkSize = 128 * 1024;

// Memory alignment for optimal NVLink transfers (256 bytes)
constexpr std::size_t kRingNVLinkAlignment = 256;

/**
 * Debug helper to print ring broadcast operation information.
 */
__device__ __forceinline__ void printRingBroadcastOperation(
    int my_rank_id,
    int root_rank_id,
    int step,
    int chunk,
    int peer_rank,
    bool is_send,
    std::size_t nbytes) {
  printf(
      "Rank=%d root=%d step=%d: %s chunk=%d %s rank=%d nbytes=%lu\n",
      my_rank_id,
      root_rank_id,
      step,
      is_send ? "SEND" : "RECV",
      chunk,
      is_send ? "to" : "from",
      peer_rank,
      nbytes);
}

/**
 * Align a size to the specified alignment boundary.
 */
__device__ __forceinline__ std::size_t alignUpRing(
    std::size_t size,
    std::size_t alignment) {
  return ((size + alignment - 1) / alignment) * alignment;
}

} // namespace

/**
 * Ring Broadcast collective communication primitive.
 *
 * Broadcasts data from root rank to all other ranks using a sequential ring
 * algorithm. Data flows around the ring one hop at a time:
 *   root -> rank1 -> rank2 -> ... -> rank(N-1)
 *
 * Algorithm Overview:
 * ===================
 * Each rank first receives the entire message from its predecessor (if not
 * root), then sends the entire message to its successor (if not last rank).
 *
 * This is a simple sequential hop algorithm:
 *   if not root: recv entire message from prev_rank
 *   if not last_rank: send entire message to next_rank
 *
 * The pipelining benefit comes from the transport layer's internal
 * pipelining within each send/recv operation. The transport automatically
 * divides each transfer into steps and uses pipeline buffer slots to
 * overlap data movement with synchronization.
 *
 * Virtual Rank Mapping:
 * =====================
 * To handle arbitrary root ranks, we use virtual ranks:
 *   virtual_rank = (actual_rank - root_rank + nranks) % nranks
 * This way, root always has virtual_rank=0.
 *
 * Why Sequential Hops (Not Chunk-Pipelined)?
 * ==========================================
 * The P2P NVLink transport's internal pipelining is designed for efficiency
 * within a single large transfer, not for coordinating multiple independent
 * transfers. Each send()/recv() call:
 *   - Uses internal stepId 0, 1, 2, ... which resets for each call
 *   - Maps stepId to pipeline buffer slots: slot = stepId % pipelineDepth
 *
 * If we tried to do chunk-level pipelining across hops:
 *   - Different chunks would use the SAME buffer slots (both start at step 0)
 *   - No actual overlap would occur between chunks on different hops
 *   - We'd just add loop overhead without performance benefit
 *
 * The transport's internal pipelining already maximizes NVLink utilization
 * for each hop, so sequential hops with large transfers is optimal.
 *
 * Performance Characteristics:
 * ============================
 * - Total latency = (N-1) × message_transfer_time
 * - Each hop achieves near-peak NVLink bandwidth due to internal pipelining
 * - For N ranks, this is O(N) slower than flat-tree O(1) for broadcast
 * - Ring is designed for reduce-scatter/all-gather, not pure broadcast
 *
 * When to Use Ring Broadcast:
 * ===========================
 * Ring broadcast is NOT recommended for pure broadcast operations.
 * Use flat-tree (Broadcast.cuh) or binomial tree (BroadcastBinomialTree.cuh)
 * instead. Ring is included for completeness and testing purposes.
 *
 * Parameters:
 *   @param buff_d: Device buffer pointer (same for source and destination)
 *   @param my_rank_id: Current rank ID
 *   @param root_rank_id: Rank ID of the broadcast source
 *   @param transports_per_rank: Array of transport objects, one per rank
 *   @param nbytes: Number of bytes to broadcast
 */
__device__ __forceinline__ void broadcast_ring(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes) {
#ifdef __CUDA_ARCH__
  const int nranks = static_cast<int>(transports_per_rank.size());

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
  auto transports = transports_per_rank.data();

  // Compute virtual rank relative to root (root has virtual_rank=0)
  int virtual_rank = (my_rank_id - root_rank_id + nranks) % nranks;

  // Ring neighbors (based on virtual rank ordering)
  int next_virtual = (virtual_rank + 1) % nranks;
  int prev_virtual = (virtual_rank - 1 + nranks) % nranks;
  int next_rank = (next_virtual + root_rank_id) % nranks;
  int prev_rank = (prev_virtual + root_rank_id) % nranks;

  char* buff = static_cast<char*>(buff_d);

  // Get transport handles for neighbors
  auto& next_transport = transports[next_rank];
  auto& prev_transport = transports[prev_rank];

  // Validate transport types for neighbors we'll actually use
  if (virtual_rank < nranks - 1) {
    assert(next_transport.type == TransportType::P2P_NVL);
  }
  if (virtual_rank > 0) {
    assert(prev_transport.type == TransportType::P2P_NVL);
  }

  // Sequential ring: first receive from predecessor, then send to successor.
  //
  // The transport's internal pipelining (pipelineDepth and chunking) handles
  // efficiency within each transfer. We transfer the entire message in one
  // send()/recv() call to maximize this internal pipelining benefit.

  // Phase 1: Receive from predecessor (if not root)
  if (virtual_rank > 0) {
#ifdef DEBUG_BROADCAST
    if (group.is_leader()) {
      printRingBroadcastOperation(
          my_rank_id, root_rank_id, 0, 0, prev_rank, false, nbytes);
    }
#endif
    prev_transport.p2p_nvl.recv(group, buff, nbytes);
  }

  // Phase 2: Send to successor (if not last rank)
  if (virtual_rank < nranks - 1) {
#ifdef DEBUG_BROADCAST
    if (group.is_leader()) {
      printRingBroadcastOperation(
          my_rank_id, root_rank_id, 1, 0, next_rank, true, nbytes);
    }
#endif
    next_transport.p2p_nvl.send(group, buff, nbytes);
  }
#endif
}

} // namespace comms::pipes::collectives
