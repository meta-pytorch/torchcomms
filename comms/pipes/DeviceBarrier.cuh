// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>
#include <cstdint>
#include "comms/pipes/BarrierState.cuh"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

// Debug-only bounds check helper for device code
// TODO(D91689639): Replace with PIPES_DEVICE_CHECK_MSG once D91689639 lands
#ifdef __CUDA_ARCH__
#define DEVICE_BARRIER_CHECK_PEER_INDEX(peer_index, nPeers)              \
  do {                                                                   \
    if (!((peer_index) >= 0 && (peer_index) < (nPeers))) {               \
      printf(                                                            \
          "DeviceBarrier: peer_index %d out of range [0, %d) at "        \
          "%s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",                  \
          (int)(peer_index),                                             \
          (int)(nPeers),                                                 \
          __FILE__,                                                      \
          __LINE__,                                                      \
          blockIdx.x,                                                    \
          blockIdx.y,                                                    \
          blockIdx.z,                                                    \
          threadIdx.x,                                                   \
          threadIdx.y,                                                   \
          threadIdx.z);                                                  \
      __trap();                                                          \
    }                                                                    \
  } while (0)
#else
#define DEVICE_BARRIER_CHECK_PEER_INDEX(peer_index, nPeers) \
  assert((peer_index) >= 0 && (peer_index) < (nPeers))
#endif

/**
 * DeviceBarrier - Device-side barrier for multi-peer synchronization
 *
 * Provides barrier synchronization across multiple peers using NVLink.
 * Implements a distributed counting barrier where each rank signals all peers
 * and waits for arrivals from all peers.
 *
 * BARRIER SEMANTICS:
 * - Each rank signals N-1 peers (excludes self - you know you've arrived)
 * - Each rank waits for N-1 arrivals from peers
 * - Uses BarrierState arrive/wait primitives internally
 *
 * BARRIER ID:
 * Multiple barriers can be used concurrently by specifying different
 * barrier_id values. Each barrier_id has independent state.
 *
 * v1: Unicast arrive (O(N) writes per barrier)
 * v2 (future): Multi-mem multicast optimization
 *
 * BOUNDS CHECKING:
 * ================
 * In device builds (__CUDA_ARCH__ defined), invalid peer values in
 * barrier_peer() trigger __trap() with a descriptive error message showing
 * the invalid value, valid range, file/line, and block/thread indices.
 * In host builds, standard assert() is used for validation.
 * Users should validate inputs before calling device functions in release
 * builds.
 *
 * RESET OPERATIONS:
 * =================
 * Barrier counters accumulate across uses. To reuse a barrier_id:
 * 1. Ensure all ranks have completed the barrier (e.g., host MPI_Barrier)
 * 2. Call reset_barrier(group, barrier_id) on all ranks
 * 3. Proceed with the next phase
 * Alternatively, use different barrier_id values for each phase.
 *
 * USAGE:
 *   // All ranks call barrier with same barrier_id
 *   barrier.barrier(group, 0);  // Synchronizes all ranks
 *
 *   // Two-sided barrier with specific peer
 *   barrier.barrier_peer(peer_index, group, 0);
 */
class DeviceBarrier {
 public:
  __host__ __device__ DeviceBarrier() = default;

  /**
   * Construct a DeviceBarrier for multi-peer synchronization
   *
   * @param myRank This rank's ID
   * @param nRanks Total number of ranks
   * @param localBarriers This rank's barrier states (local memory)
   * @param peerBarrierPtrs Pointers to peers' barrier states, peer-indexed
   *   (size = nRanks - 1, excludes self)
   */
  __host__ __device__ DeviceBarrier(
      int myRank,
      int nRanks,
      DeviceSpan<BarrierState> localBarriers,
      DeviceSpan<BarrierState*> peerBarrierPtrs)
      : myRank_(myRank),
        nRanks_(nRanks),
        localBarriers_(localBarriers),
        peerBarrierPtrs_(peerBarrierPtrs) {}

  // ===========================================================================
  // Barrier Operations
  // ===========================================================================

  /**
   * barrier - N-way barrier: synchronize with all ranks
   *
   * All ranks must call this to complete the barrier.
   * Algorithm (v1 - unicast):
   * 1. Signal arrival to all N-1 peers
   * 2. Wait for N-1 arrivals from peers
   *
   * @param group ThreadGroup for cooperative processing
   * @param barrier_id Index of the barrier to use
   * @param timeout_ns Optional timeout in nanoseconds (default: infinite)
   */
  __device__ __forceinline__ void barrier(
      ThreadGroup& group,
      int barrier_id,
      [[maybe_unused]] uint64_t timeout_ns = UINT64_MAX) {
    // Step 1: Signal arrival to all peers (peer-indexed, excludes self)
    for (uint32_t i = 0; i < peerBarrierPtrs_.size(); ++i) {
      peerBarrierPtrs_[i][barrier_id].arrive(group);
    }

    // Step 2: Wait for N-1 arrivals from peers
    // BarrierState uses increment-then-wait semantics:
    // - 1st wait: expects current_counter >= 1 (1 peer arrived)
    // - 2nd wait: expects current_counter >= 2 (2 peers arrived)
    // - ...
    // - (N-1)th wait: expects current_counter >= N-1 (all peers arrived)
    //
    // NOTE: BarrierState counters accumulate across barrier() calls.
    // Use reset_barrier() between phases if reusing the same barrier_id,
    // or use different barrier_id values for each phase.
    // TODO: Implement timeout logic
    for (uint32_t i = 0; i < peerBarrierPtrs_.size(); ++i) {
      localBarriers_[barrier_id].wait(group);
    }
  }

  /**
   * barrier_peer - Two-sided barrier with a specific peer
   *
   * Synchronizes with a single peer. Both ranks must call barrier_peer
   * with each other's peer_index to complete.
   *
   * @param peer_index Peer index in [0, num_peers)
   * @param group ThreadGroup for cooperative processing
   * @param barrier_id Index of the barrier to use
   * @param timeout_ns Optional timeout in nanoseconds (default: infinite)
   */
  __device__ __forceinline__ void barrier_peer(
      int peer_index,
      ThreadGroup& group,
      int barrier_id,
      [[maybe_unused]] uint64_t timeout_ns = UINT64_MAX) {
    DEVICE_BARRIER_CHECK_PEER_INDEX(
        peer_index, static_cast<int>(peerBarrierPtrs_.size()));

    // Signal arrival to peer
    peerBarrierPtrs_[peer_index][barrier_id].arrive(group);

    // Wait for peer's arrival
    // TODO: Implement timeout logic
    localBarriers_[barrier_id].wait(group);
  }

  // ===========================================================================
  // Reset Operations
  // ===========================================================================

  /**
   * reset_barrier - Reset a specific barrier slot to initial state
   *
   * Resets the barrier's counters to 0. Only safe when all ranks have
   * completed prior barrier operations on this slot.
   *
   * @param group ThreadGroup for cooperative processing
   * @param barrier_id Index of the barrier to reset
   *
   * SAFETY: Must be called collectively by all ranks after ensuring no
   * barrier operations are in flight. Typically done after a host-side
   * MPI_Barrier to guarantee synchronization.
   */
  __device__ __forceinline__ void reset_barrier(
      ThreadGroup& group,
      int barrier_id) {
    localBarriers_[barrier_id].reset(group);
  }

  /**
   * reset_all_barriers - Reset all barrier slots
   *
   * Convenience method to reset all barriers. Same safety requirements
   * as reset_barrier().
   *
   * @param group ThreadGroup for cooperative processing
   */
  __device__ __forceinline__ void reset_all_barriers(ThreadGroup& group) {
    if (group.is_leader()) {
      for (uint32_t i = 0; i < localBarriers_.size(); ++i) {
        localBarriers_[i].reset(group);
      }

      group.sync();
    }
  }

  // ===========================================================================
  // Accessors
  // ===========================================================================

  __device__ __forceinline__ int rank() const {
    return myRank_;
  }

  __device__ __forceinline__ int n_ranks() const {
    return nRanks_;
  }

 private:
  int myRank_{-1};
  int nRanks_{0};
  DeviceSpan<BarrierState> localBarriers_;
  DeviceSpan<BarrierState*> peerBarrierPtrs_;
};

/**
 * getMultiPeerBarrierBufferSize - Calculate buffer size for multi-peer barriers
 *
 * @param barrierCount Number of barrier slots
 * @return Size in bytes, aligned to 128-byte boundary
 */
__host__ __device__ __forceinline__ std::size_t getMultiPeerBarrierBufferSize(
    int barrierCount) {
  return getBarrierBufferSize(barrierCount);
}

} // namespace comms::pipes
