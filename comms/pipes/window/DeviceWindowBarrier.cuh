// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>
#include <cstdint>
#include "comms/pipes/BarrierState.cuh"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes {

// Debug-only bounds check helper for device code
// TODO(D91689639): Replace with PIPES_DEVICE_CHECK_MSG once D91689639 lands
#ifdef __CUDA_ARCH__
#define DEVICE_WINDOW_BARRIER_CHECK_PEER_INDEX(peer_index, nPeers)      \
  do {                                                                  \
    if (!((peer_index) >= 0 && (peer_index) < (nPeers))) {              \
      printf(                                                           \
          "DeviceWindowBarrier: peer_index %d out of range [0, %d) at " \
          "%s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",                 \
          (int)(peer_index),                                            \
          (int)(nPeers),                                                \
          __FILE__,                                                     \
          __LINE__,                                                     \
          blockIdx.x,                                                   \
          blockIdx.y,                                                   \
          blockIdx.z,                                                   \
          threadIdx.x,                                                  \
          threadIdx.y,                                                  \
          threadIdx.z);                                                 \
      __trap();                                                         \
    }                                                                   \
  } while (0)
#else
#define DEVICE_WINDOW_BARRIER_CHECK_PEER_INDEX(peer_index, nPeers) \
  assert((peer_index) >= 0 && (peer_index) < (nPeers))
#endif

/**
 * DeviceWindowBarrier - Device-side barrier for multi-peer synchronization
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
 * COUNTER BEHAVIOR:
 * =================
 * Barrier counters grow monotonically and are never reset from the device
 * side (device-side reset has an inherent race condition — see BarrierState
 * docs). Two recommended patterns for multi-phase usage:
 * 1. Reuse the same barrier_id: counters accumulate across barrier() calls
 *    and the increment-then-wait semantics handle this automatically.
 * 2. Use different barrier_id values for each phase.
 * To reinitialize counters between kernel launches, use host-side
 * cudaMemset when the GPU is idle.
 *
 * USAGE:
 *   // All ranks call barrier with same barrier_id
 *   barrier.barrier(group, 0);  // Synchronizes all ranks
 *
 *   // Two-sided barrier with specific peer
 *   barrier.barrier_peer(peer_index, group, 0);
 */
class DeviceWindowBarrier {
 public:
  __host__ __device__ DeviceWindowBarrier() = default;

  /**
   * Construct a DeviceWindowBarrier for multi-peer synchronization
   *
   * @param myRank This rank's ID
   * @param nRanks Total number of ranks
   * @param localBarriers This rank's barrier states (local memory)
   * @param peerBarrierPtrs Pointers to peers' barrier states, peer-indexed
   *   (size = nRanks - 1, excludes self)
   */
  __host__ __device__ DeviceWindowBarrier(
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
   * 1. Signal arrival to all N-1 peers (parallelized across threads)
   * 2. Wait for N-1 arrivals from peers
   *
   * @param group ThreadGroup for cooperative processing
   * @param barrier_id Index of the barrier to use
   * @param timeout Optional timeout (default: no timeout, infinite wait)
   */
  __device__ __forceinline__ void barrier(
      ThreadGroup& group,
      int barrier_id,
      const Timeout& timeout = Timeout()) {
    // Step 1: Signal arrival to all peers (peer-indexed, excludes self)
    // Thread-level parallelism: each thread signals different peers
    group.sync();
    int nPeers = static_cast<int>(peerBarrierPtrs_.size());
    for (int i = static_cast<int>(group.thread_id_in_group); i < nPeers;
         i += static_cast<int>(group.group_size)) {
      peerBarrierPtrs_[i][barrier_id].arrive();
    }

    // Ensure all threads have finished signaling before any thread starts
    // waiting. Without this, a fast leader could enter the wait loop while
    // other threads are still signaling their assigned peers.
    group.sync();

    // Step 2: Wait for N-1 arrivals from peers
    // BarrierState uses increment-then-wait semantics:
    // - 1st wait: expects current_counter >= 1 (1 peer arrived)
    // - 2nd wait: expects current_counter >= 2 (2 peers arrived)
    // - ...
    // - (N-1)th wait: expects current_counter >= N-1 (all peers arrived)
    //
    // NOTE: BarrierState counters accumulate across barrier() calls.
    // This is by design; the increment-then-wait semantics handle
    // reuse automatically without any reset.
    for (uint32_t i = 0; i < peerBarrierPtrs_.size(); ++i) {
      localBarriers_[barrier_id].wait(group, timeout);
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
   * @param timeout Optional timeout (default: no timeout, infinite wait)
   */
  __device__ __forceinline__ void barrier_peer(
      int peer_index,
      ThreadGroup& group,
      int barrier_id,
      const Timeout& timeout = Timeout()) {
    DEVICE_WINDOW_BARRIER_CHECK_PEER_INDEX(
        peer_index, static_cast<int>(peerBarrierPtrs_.size()));

    // Signal arrival to peer
    peerBarrierPtrs_[peer_index][barrier_id].arrive(group);

    // Wait for peer's arrival
    localBarriers_[barrier_id].wait(group, timeout);
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
