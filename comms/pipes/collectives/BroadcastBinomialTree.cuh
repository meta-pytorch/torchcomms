// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/Broadcast.cuh"
#include "comms/pipes/collectives/BroadcastRing.cuh"

namespace comms::pipes::collectives {

namespace {

/**
 * Debug helper to print binomial tree broadcast operation information.
 */
__device__ __forceinline__ void printBinomialTreeOperation(
    int my_rank_id,
    int root_rank_id,
    int virtual_rank,
    int round,
    int peer_rank_id,
    bool is_send,
    std::size_t nbytes) {
  printf(
      "Rank=%d (virtual=%d) root=%d round=%d: %s rank=%d nbytes=%lu\n",
      my_rank_id,
      virtual_rank,
      root_rank_id,
      round,
      is_send ? "send to" : "recv from",
      peer_rank_id,
      nbytes);
}

} // namespace

/**
 * Binomial Tree Broadcast collective communication primitive.
 *
 * Broadcasts data from root rank to all other ranks using a binomial tree
 * algorithm with O(log N) rounds. This is more bandwidth-efficient than
 * the flat-tree (star) algorithm for large messages because:
 *
 *   1. Root bandwidth = 1×messageSize (sends to one peer per round)
 *   2. Parallelism increases each round (2^round senders in round r)
 *   3. Total bandwidth distributed across all nodes
 *
 * ROUND-MAJOR PIPELINING
 * ======================
 * This implementation uses round-major ordering: in each round, the entire
 * message is transferred (not chunk-by-chunk). The transport layer handles
 * internal chunking and pipelining, which allows:
 *   - Maximum utilization of the transport's pipeline depth
 *   - Concurrent chunk transfers within each round
 *   - Minimal synchronization overhead between chunks
 *
 * This is much faster than chunk-major ordering (which was used previously)
 * because the transport's internal pipelining can work across the full message
 * rather than being limited to small individual chunks.
 *
 * Algorithm:
 * ==========
 * For N ranks, the broadcast completes in ceil(log2(N)) rounds.
 *
 * In each round r (0-indexed):
 *   - Ranks with virtual_rank < 2^r have the data
 *   - Each such rank sends to virtual_rank + 2^r (if that rank exists)
 *   - Receiving ranks wait for data before proceeding to next round
 *
 * Example for 4 ranks (root=0):
 *   Round 0: Rank 0 → Rank 1 (virtual_rank 0 sends to 0+1=1)
 *   Round 1: Rank 0 → Rank 2, Rank 1 → Rank 3
 *
 * Example for 8 ranks (root=0):
 *   Round 0: Rank 0 → Rank 1
 *   Round 1: Rank 0 → Rank 2, Rank 1 → Rank 3
 *   Round 2: Ranks 0→4, 1→5, 2→6, 3→7
 *
 * Virtual Rank Mapping:
 * =====================
 * To support arbitrary root ranks, we use virtual ranks:
 *   virtual_rank = (actual_rank - root_rank + nranks) % nranks
 *   actual_rank = (virtual_rank + root_rank) % nranks
 *
 * This way, the root always has virtual_rank=0, and the algorithm
 * proceeds identically regardless of which physical rank is the root.
 *
 * Parameters:
 *   @param buff_d: Device buffer pointer
 *                  - For root: contains source data to broadcast
 *                  - For non-root: receives broadcast data
 *   @param my_rank_id: Current rank ID
 *   @param root_rank_id: Rank ID of the broadcast source
 *   @param transports_per_rank: Array of transport objects, one per rank
 *   @param nbytes: Number of bytes to broadcast
 *
 * Requirements:
 * - Must be called from device code with sufficient threads
 * - All ranks must call with same root_rank_id and nbytes
 * - Max 8 ranks supported
 * - transports_per_rank[my_rank_id].type must be SELF
 * - transports_per_rank[i].type must be P2P_NVL for i != my_rank_id
 */
__device__ __forceinline__ void broadcast_binomial_tree(
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

  // Calculate number of rounds = ceil(log2(nranks))
  int num_rounds = 0;
  for (int n = 1; n < nranks; n <<= 1) {
    num_rounds++;
  }

  char* buff = static_cast<char*>(buff_d);

  // ROUND-MAJOR ordering: Transfer entire message each round.
  // This allows the transport layer to use its full pipeline depth
  // for maximum bandwidth utilization.
  for (int round = 0; round < num_rounds; ++round) {
    int distance = 1 << round; // 2^round

    if (virtual_rank < distance) {
      // I have the data - check if I need to send to a peer
      int send_to_virtual = virtual_rank + distance;

      if (send_to_virtual < nranks) {
        // Convert virtual rank back to actual rank
        int send_to_actual = (send_to_virtual + root_rank_id) % nranks;

        auto& transport = transports[send_to_actual];
        assert(transport.type == TransportType::P2P_NVL);

#ifdef DEBUG_BROADCAST
        if (group.is_leader()) {
          printBinomialTreeOperation(
              my_rank_id,
              root_rank_id,
              virtual_rank,
              round,
              send_to_actual,
              true, // is_send
              nbytes);
        }
#endif

        // Send entire message - transport handles internal chunking/pipelining
        transport.p2p_nvl.send(group, buff, nbytes);
      }
      // If no peer to send to in this round, just continue to next round
    } else if (virtual_rank < 2 * distance) {
      // I don't have data yet - receive from peer
      int recv_from_virtual = virtual_rank - distance;
      int recv_from_actual = (recv_from_virtual + root_rank_id) % nranks;

      auto& transport = transports[recv_from_actual];
      assert(transport.type == TransportType::P2P_NVL);

#ifdef DEBUG_BROADCAST
      if (group.is_leader()) {
        printBinomialTreeOperation(
            my_rank_id,
            root_rank_id,
            virtual_rank,
            round,
            recv_from_actual,
            false, // is_send
            nbytes);
      }
#endif

      // Receive entire message - transport handles internal chunking/pipelining
      transport.p2p_nvl.recv(group, buff, nbytes);
    }
    // Ranks >= 2*distance don't participate in this round

    // Implicit synchronization: recv completes before proceeding,
    // which acts as a barrier for that rank to have data before
    // potentially sending in the next round.
  }
#endif
}

/**
 * Wrapper that selects between flat-tree and ring broadcast
 * based on message size and rank count.
 *
 * Selection criteria (based on empirical profiling on 8-rank NVLink):
 * - For small/medium messages (< 8MB): Use flat-tree (lower latency)
 * - For large messages (>= 8MB) with multiple ranks: Use ring
 *   (1.77x faster than flat-tree at 8MB, 5.19x faster at 64MB)
 *
 * Benchmark results for 64MB messages:
 * - Ring: 253.36 GB/s (0.77x NCCL)
 * - Binomial: 116.65 GB/s (0.35x NCCL)
 * - Flat: 48.80 GB/s (0.15x NCCL)
 *
 * The 8MB threshold was determined through benchmarking where ring starts
 * to significantly outperform both flat-tree and binomial tree.
 */
__device__ __forceinline__ void broadcast_adaptive(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes) {
#ifdef __CUDA_ARCH__
  // Threshold for switching to ring algorithm (8MB)
  // Empirically determined: ring is 1.77x faster than flat at 8MB,
  // and becomes dominant for all larger message sizes
  // IMPORTANT: Must match kRingThreshold in
  // comms/pipes/benchmarks/BroadcastTuning.h
  constexpr std::size_t kRingThreshold = 8 * 1024 * 1024;

  const auto nranks = transports_per_rank.size();

  if (nbytes >= kRingThreshold && nranks > 2) {
    // Large messages: use ring for best bandwidth
    // Ring achieves 77% of NCCL at 64MB (253.36 GB/s)
    broadcast_ring(
        buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
  } else {
    // Small/medium messages or small rank count: use flat-tree
    // Flat-tree has lower latency overhead for smaller transfers
    broadcast(buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
  }
#endif
}

} // namespace comms::pipes::collectives
