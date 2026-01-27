// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/Broadcast.cuh"

namespace comms::pipes::collectives {

namespace {

// Chunk size for pipelined binomial tree broadcast (128KB)
// Optimized based on empirical profiling - smaller chunks enable
// better warp parallelism across chunks within each round
constexpr std::size_t kBinomialChunkSize = 128 * 1024;

// Memory alignment for optimal NVLink transfers (256 bytes)
constexpr std::size_t kNVLinkAlignment = 256;

// Maximum number of chunks that can be processed concurrently by warps
// This limits memory usage for synchronization flags
constexpr std::size_t kMaxConcurrentChunks = 256;

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

/**
 * Align a size to the specified alignment boundary.
 */
__device__ __forceinline__ std::size_t alignUp(
    std::size_t size,
    std::size_t alignment) {
  return ((size + alignment - 1) / alignment) * alignment;
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
 * Optimization: Chunk-Based Pipelining
 * =====================================
 * For large messages, the data is broken into chunks and processed in a
 * pipelined manner. This allows:
 *   - Overlapping communication across rounds for different chunks
 *   - Better utilization of available bandwidth
 *   - Warps can be partitioned to work on different chunks concurrently
 *
 * The pipelining follows a "chunk-major" order:
 *   for each chunk:
 *     for each round:
 *       process chunk in this round
 *
 * This ensures that once a rank receives a chunk, it can immediately forward
 * it in the next round while the previous rank moves on to the next chunk.
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

  // Determine chunk size (aligned for optimal NVLink transfers)
  const std::size_t chunk_size = alignUp(kBinomialChunkSize, kNVLinkAlignment);
  const std::size_t num_chunks = (nbytes + chunk_size - 1) / chunk_size;

  char* buff = static_cast<char*>(buff_d);

  // Chunk-major pipelining: process chunks sequentially, allowing
  // pipelining across rounds for different chunks.
  //
  // NOTE: We cannot partition warps across chunks like the flat-tree does
  // across peers, because in the binomial tree algorithm, send/recv are
  // paired operations that require ALL warps on both sender and receiver
  // to participate. If we partition chunks across warps, we'd have a
  // subset of warps trying to send while another subset tries to receive
  // a different chunk, causing a deadlock.
  //
  // The pipelining benefit comes from the transport layer's internal
  // pipelining, not from warp-level parallelism across chunks.
  for (std::size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const std::size_t chunk_offset = chunk_idx * chunk_size;
    const std::size_t chunk_bytes = (chunk_offset + chunk_size <= nbytes)
        ? chunk_size
        : (nbytes - chunk_offset);

    char* chunk_ptr = buff + chunk_offset;

    // Process each round of the binomial tree for this chunk
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
                chunk_bytes);
          }
#endif

          transport.p2p_nvl.send(group, chunk_ptr, chunk_bytes);
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
              chunk_bytes);
        }
#endif

        transport.p2p_nvl.recv(group, chunk_ptr, chunk_bytes);
      }
      // Ranks >= 2*distance don't participate in this round

      // Implicit synchronization: recv completes before proceeding,
      // which acts as a barrier for that rank to have data before
      // potentially sending in the next round.
    }
  }
#endif
}

/**
 * Wrapper that selects between flat-tree and binomial tree broadcast
 * based on message size and rank count.
 *
 * Selection criteria:
 * - For small messages (< 64KB): Use flat-tree (lower latency)
 * - For large messages (>= 64KB) with many ranks: Use binomial tree
 *   (better bandwidth utilization)
 *
 * The threshold can be tuned based on profiling results.
 */
__device__ __forceinline__ void broadcast_adaptive(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes) {
#ifdef __CUDA_ARCH__
  // Threshold for switching to binomial tree (64KB)
  constexpr std::size_t kBinomialThreshold = 64 * 1024;

  const auto nranks = transports_per_rank.size();

  // Use binomial tree for large messages with multiple ranks
  // The benefit of binomial tree increases with message size and rank count
  if (nbytes >= kBinomialThreshold && nranks > 2) {
    broadcast_binomial_tree(
        buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
  } else {
    // Fall back to flat-tree for small messages or small rank counts
    // Import the flat-tree broadcast from Broadcast.cuh
    broadcast(buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
  }
#endif
}

} // namespace comms::pipes::collectives
