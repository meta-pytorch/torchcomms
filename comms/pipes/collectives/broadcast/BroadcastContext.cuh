// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes::collectives {

// ============================================================================
// Topology Tag Types (for compile-time dispatch)
// ============================================================================

struct FlatTag {};
struct RingTag {};
struct BinomialTreeTag {};

// ============================================================================
// BroadcastContext - Common state for all broadcast topologies
// ============================================================================

/**
 * Encapsulates common broadcast state shared by all topologies.
 *
 * This struct consolidates the setup code for all broadcast algorithms:
 * - Buffer pointer (as char* to avoid repeated casts)
 * - Rank information (my_rank, root_rank, nranks, virtual_rank)
 * - Message size
 * - Transport array pointer
 * - ThreadGroup for parallel execution
 */
struct BroadcastContext {
  char* buff;
  int my_rank_id;
  int root_rank_id;
  int nranks;
  int virtual_rank;
  std::size_t nbytes;
  Transport* transports;
  ThreadGroup group;

  /**
   * Factory method to create BroadcastContext.
   * Must be called within __CUDA_ARCH__ guard.
   */
  __device__ __forceinline__ static BroadcastContext create(
      void* buff_d,
      int my_rank_id,
      int root_rank_id,
      DeviceSpan<Transport> transports_per_rank,
      std::size_t nbytes) {
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
        .group = make_warp_group()};
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

// ============================================================================
// TopologyTraits - Template for topology-specific execute() implementations
// ============================================================================

/**
 * Primary template (undefined) - specializations provide execute().
 * This causes compile errors if an unsupported tag is used.
 */
template <typename TopologyTag>
struct TopologyTraits;

// Specialization declarations - implementations in BroadcastTopologies.cuh
template <>
struct TopologyTraits<FlatTag> {
  __device__ __forceinline__ static void execute(BroadcastContext& ctx);
};

template <>
struct TopologyTraits<RingTag> {
  __device__ __forceinline__ static void execute(BroadcastContext& ctx);
};

template <>
struct TopologyTraits<BinomialTreeTag> {
  __device__ __forceinline__ static void execute(BroadcastContext& ctx);
};

// ============================================================================
// Unified Broadcast Entry Point
// ============================================================================

/**
 * broadcast<TopologyTag>() - Primary broadcast API with compile-time topology
 * selection.
 *
 * This is the preferred API for new code. It provides zero-cost abstraction
 * through template dispatch to topology-specific implementations.
 *
 * Usage:
 *   broadcast<FlatTag>(buff, my_rank, root_rank, transports, nbytes);
 *   broadcast<RingTag>(buff, my_rank, root_rank, transports, nbytes);
 *   broadcast<BinomialTreeTag>(buff, my_rank, root_rank, transports, nbytes);
 *
 * @tparam TopologyTag One of: FlatTag, RingTag, BinomialTreeTag
 * @param buff_d Device buffer (source for root, destination for non-root)
 * @param my_rank_id This rank's ID
 * @param root_rank_id The rank that originates the broadcast
 * @param transports_per_rank Array of Transport handles (index = peer rank)
 * @param nbytes Number of bytes to broadcast
 */
template <typename TopologyTag>
__device__ __forceinline__ void broadcast(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes) {
#ifdef __CUDA_ARCH__
  auto ctx = BroadcastContext::create(
      buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);

  if (ctx.should_skip()) {
    return;
  }

  TopologyTraits<TopologyTag>::execute(ctx);
#endif
}

} // namespace comms::pipes::collectives
