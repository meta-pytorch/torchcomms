// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include "comms/common/AtomicUtils.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

/**
 * SignalState - Synchronization primitive for P2P NVLink barrier operations
 *
 * A 128-byte aligned synchronization primitive with two counters that enable
 * independent signal/wait operations between peers.
 *
 * MEMORY LAYOUT:
 * ==============
 * - signal_: 64-bit counter incremented by signal() (receives signals)
 * - localState_: 64-bit counter for tracking expected signal count locally
 * - Remaining bytes: padding for cache line isolation
 * - Total size: 128 bytes (cache line aligned)
 *
 * TWO-COUNTER DESIGN:
 * ===================
 *
 * The two counters serve different purposes:
 * - signal_: Incremented when a peer signals this state (via NVLink remote
 * write)
 * - localState_: Incremented locally by wait() to track expected signal count
 *
 * This design allows wait() to be called without passing an explicit expected
 * value - each wait() automatically expects the next signal count.
 *
 * PROTOCOL (Pairwise Barrier via P2pNvlTransportDevice):
 * ======================================================
 *
 * Each PE has a local SignalState and access to peer's SignalState via NVLink.
 * The barrier() operation calls:
 *   - remoteState.signalState->signal() to notify peer
 *   - localState.signalState->wait() to wait for peer's notification
 *
 *   PE 0                              PE 1
 *   ────                              ────
 *   signal_=0, localState_=0          signal_=0, localState_=0
 *       │                                  │
 *       │ signal() on PE1's state ────────▶│ PE1.signal_++
 *       │                                  │
 *       │ PE0.signal_++ ◀──────────────────│ signal() on PE0's state
 *       │                                  │
 *   wait() on own state               wait() on own state
 *     localState_++ → expected=1        localState_++ → expected=1
 *     poll signal_ >= 1                 poll signal_ >= 1
 *       │                                  │
 *   barrier complete                  barrier complete
 *
 * MEMORY SEMANTICS:
 * =================
 * - All reads use acquire ordering (visible after peer's release)
 * - All writes use release ordering (visible to peer after their acquire)
 * - Uses .sys scope for cross-GPU NVLink coherence
 *
 */
struct alignas(128) SignalState {
  volatile uint64_t current_counter_;
  volatile uint64_t expected_counter_;
  char padding_[128 - 2 * sizeof(uint64_t)]{};

  __host__ __device__ SignalState()
      : current_counter_(0), expected_counter_(0) {}

  // ===========================================================================
  // Core Operations
  // ===========================================================================

  /**
   * atomic_inc - Atomically increment counter
   *
   * @return The value before incrementing
   */
  __device__ __forceinline__ uint64_t atomic_inc(uint64_t* ptr) {
#ifdef __CUDA_ARCH__
    return comms::device::atomic_fetch_add_sys_global(ptr, 1);
#else
    return (*ptr)++;
#endif
  }

  /**
   * signal - Increment the signal counter to notify a waiting peer
   *
   * Atomically increments the signal_ counter. When called on a peer's
   * SignalState (via NVLink remote pointer), this notifies the peer that
   * we have reached a synchronization point.
   */
  __device__ __forceinline__ void signal() {
    atomic_inc(const_cast<uint64_t*>(&current_counter_));
  }

  /**
   * wait - Wait for the next expected signal from a peer
   *
   * Atomically increments localState_ to determine the expected signal count,
   * then spins until signal_ reaches that value. This allows multiple
   * sequential wait() calls without explicit counter tracking by the caller.
   *
   * Uses acquire semantics to ensure subsequent reads see peer's writes.
   */
  __device__ __forceinline__ void wait() {
    uint64_t expected =
        atomic_inc(const_cast<uint64_t*>(&expected_counter_)) + 1;
    while (current_counter_ < expected) {
    }
  }

  // ===========================================================================
  // Thread-Group-Safe Operations
  // ===========================================================================
  // These methods coordinate signal operations across all threads in a group.
  //
  // Note: Please do not use them concurrently across different threadgroups.

  /**
   * signal - Thread-group-safe signal operation with atomic increment
   *
   * Synchronizes all threads in the group, then the leader atomically
   * increments the counter. This ensures all prior memory operations from all
   * threads in the group are complete before signaling.
   *
   * @param group ThreadGroup for cooperative processing
   */
  __device__ __forceinline__ void signal(ThreadGroup& group) {
    // Sync to ensure all prior memory operations from all threads are complete
    group.sync();

    // Only leader performs the atomic increment
    if (group.is_leader()) {
      comms::device::threadfence_system();
      signal();
    }
  }

  /**
   * wait - Thread-group-safe wait operation
   *
   * Only the group leader polls until the signal counter reaches the expected
   * value.
   *
   * @param group ThreadGroup for cooperative processing
   */
  __device__ __forceinline__ void wait(ThreadGroup& group) {
    if (group.is_leader()) {
      wait();
    }
    group.sync();
  }
};

static_assert(
    alignof(SignalState) == 128,
    "SignalState must be 128-byte aligned");

} // namespace comms::pipes
