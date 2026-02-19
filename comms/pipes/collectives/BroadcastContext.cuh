// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes::collectives {

// ============================================================================
// BroadcastContext - Common state for broadcast operations
// ============================================================================

/**
 * BroadcastContext - Shared state for broadcast operations.
 *
 * Encapsulates the common parameters needed by broadcast algorithms,
 * providing a unified interface.
 *
 * Virtual Rank Mapping:
 *   virtual_rank = (actual_rank - root_rank + nranks) % nranks
 *   actual_rank = (virtual_rank + root_rank) % nranks
 *
 * This ensures root always has virtual_rank=0 regardless of which
 * physical rank is the root.
 */
struct BroadcastContext {
  // User buffer pointer (source for root, destination for non-root)
  char* buff;

  // Rank information
  int my_rank_id;
  int root_rank_id;
  int nranks;
  int virtual_rank;

  // Message size
  std::size_t nbytes;

  // Transport array (indexed by actual rank)
  Transport* transports;

  // ThreadGroup for cooperative operations
  ThreadGroup group;

  // Signal base value (for pipelined topologies)
  uint64_t signal_base{0};

  /**
   * Factory method to create BroadcastContext.
   * Must be called within __CUDA_ARCH__ guard.
   */
  __device__ __forceinline__ static BroadcastContext create(
      void* buff_d,
      int my_rank_id,
      int root_rank_id,
      DeviceSpan<Transport> transports_per_rank,
      std::size_t nbytes,
      uint64_t signal_base = 0) {
    int nranks = static_cast<int>(transports_per_rank.size());
    int virtual_rank = (my_rank_id - root_rank_id + nranks) % nranks;

    return BroadcastContext{
        .buff = static_cast<char*>(buff_d),
        .my_rank_id = my_rank_id,
        .root_rank_id = root_rank_id,
        .nranks = nranks,
        .virtual_rank = virtual_rank,
        .nbytes = nbytes,
        .transports = transports_per_rank.data(),
        .group = make_warp_group(),
        .signal_base = signal_base};
  }

  /**
   * Factory method that accepts an externally-provided ThreadGroup.
   *
   * Use this when the caller needs block-local warp groups (e.g.,
   * multi-channel kernels where each block is an independent channel).
   * The caller creates the group via make_block_warp_group() and passes
   * it here instead of using the global make_warp_group() default.
   */
  __device__ __forceinline__ static BroadcastContext create_with_group(
      void* buff_d,
      int my_rank_id,
      int root_rank_id,
      DeviceSpan<Transport> transports_per_rank,
      std::size_t nbytes,
      ThreadGroup group,
      uint64_t signal_base = 0) {
    int nranks = static_cast<int>(transports_per_rank.size());
    int virtual_rank = (my_rank_id - root_rank_id + nranks) % nranks;

    return BroadcastContext{
        .buff = static_cast<char*>(buff_d),
        .my_rank_id = my_rank_id,
        .root_rank_id = root_rank_id,
        .nranks = nranks,
        .virtual_rank = virtual_rank,
        .nbytes = nbytes,
        .transports = transports_per_rank.data(),
        .group = group,
        .signal_base = signal_base};
  }

  /**
   * Check if broadcast should be skipped (single rank or zero bytes).
   */
  __device__ __forceinline__ bool should_skip() const {
    return nranks == 1 || nbytes == 0;
  }

  /**
   * Check if this rank is the broadcast root.
   */
  __device__ __forceinline__ bool is_root() const {
    return my_rank_id == root_rank_id;
  }

  /**
   * Convert virtual rank back to actual rank.
   * virtual_rank 0 = root, virtual_rank 1 = next in ring, etc.
   */
  __device__ __forceinline__ int actual_rank(int vrank) const {
    return (vrank + root_rank_id) % nranks;
  }
};

} // namespace comms::pipes::collectives

// Include implementation after BroadcastContext is fully defined
#include "comms/pipes/collectives/BroadcastImpl.cuh"

namespace comms::pipes::collectives {

// ============================================================================
// Broadcast Public API
// ============================================================================

/**
 * Broadcast data from root rank to all other ranks.
 *
 * Uses pipelined ring topology with fused dual-destination copies at
 * intermediate ranks for optimal bandwidth utilization.
 *
 * @param buff_d Device buffer (source for root, destination for non-root)
 * @param my_rank_id This rank's ID
 * @param root_rank_id The rank that originates the broadcast
 * @param transports_per_rank Array of Transport handles (index = peer rank)
 * @param nbytes Number of bytes to broadcast
 * @param signal_base Optional base value for signal operations
 */
__device__ __forceinline__ void broadcast(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    uint64_t signal_base = 0) {
#ifdef __CUDA_ARCH__
  auto ctx = BroadcastContext::create(
      buff_d,
      my_rank_id,
      root_rank_id,
      transports_per_rank,
      nbytes,
      signal_base);
  if (ctx.should_skip()) {
    return;
  }
  detail::broadcast_pipelined_ring(ctx);
#endif
}

/**
 * Broadcast data from root rank to all other ranks.
 *
 * Overload accepting externally-provided ThreadGroup for multi-channel kernels
 * where each block handles one channel and needs block-local warp groups.
 *
 * @param buff_d Device buffer (source for root, destination for non-root)
 * @param my_rank_id This rank's ID
 * @param root_rank_id The rank that originates the broadcast
 * @param transports_per_rank Array of Transport handles (index = peer rank)
 * @param nbytes Number of bytes to broadcast
 * @param group ThreadGroup for cooperative operations
 * @param signal_base Optional base value for signal operations
 */
__device__ __forceinline__ void broadcast(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    ThreadGroup group,
    uint64_t signal_base = 0) {
#ifdef __CUDA_ARCH__
  auto ctx = BroadcastContext::create_with_group(
      buff_d,
      my_rank_id,
      root_rank_id,
      transports_per_rank,
      nbytes,
      group,
      signal_base);
  if (ctx.should_skip()) {
    return;
  }
  detail::broadcast_pipelined_ring(ctx);
#endif
}

} // namespace comms::pipes::collectives
