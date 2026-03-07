// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// @lint-ignore-every CLANGTIDY facebook-modularize-issue-check

#pragma once

#include <cstddef>
#include <cstdint>
#include "comms/common/AtomicUtils.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes {

// Forward declaration for thread-safe overloads
struct ThreadGroup;

/**
 * ChunkState - State machine for P2P NVLink chunk synchronization
 *
 * A 128-byte aligned synchronization primitive that manages the lifecycle
 * of a data chunk in the P2P transfer pipeline.
 *
 * STATES:
 * =======
 *   READY_TO_SEND (-1) : Buffer is empty, sender can write
 *   READY_TO_RECV (N)  : Buffer has data from step N, receiver can read
 *
 * STATE MACHINE:
 * ==============
 *                      ready_to_recv(stepId)
 *    ┌───────────────┐ ─────────────────────▶ ┌───────────────┐
 *    │ READY_TO_SEND │                        │ READY_TO_RECV │
 *    │     (-1)      │ ◀───────────────────── │   (stepId)    │
 *    └───────────────┘      ready_to_send()   └───────────────┘
 *
 * SENDER WORKFLOW:
 *   1. wait_ready_to_send()      - Block until state == READY_TO_SEND
 *   2. [copy data to buffer]
 *   3. ready_to_recv(stepId)    - Transition to READY_TO_RECV
 *
 * RECEIVER WORKFLOW:
 *   1. wait_ready_to_recv(stepId) - Block until state == stepId
 *   2. [copy data from buffer]
 *   3. ready_to_send()           - Transition to READY_TO_SEND
 *
 * MEMORY LAYOUT:
 * - Bytes 0-3: value_ (int) - sync state
 * - Bytes 4-127: padding for cache line isolation
 * - Total size: 128 bytes (cache line aligned)
 *
 * MEMORY SEMANTICS:
 * - All reads use acquire ordering (visible after peer's release)
 * - All writes use release ordering (visible to peer after their acquire)
 * - Uses .sys scope for cross-GPU NVLink coherence
 */
struct alignas(128) ChunkState {
  static constexpr int32_t READY_TO_SEND = -1;

  int32_t value_; // 4 bytes - sync state (stepId or READY_TO_SEND)
  char padding_[128 - sizeof(int32_t)]{};

  __host__ __device__ ChunkState() : value_(READY_TO_SEND) {}

  // ===========================================================================
  // Core Operations (Thread-Group-Safe)
  // ===========================================================================
  //
  // All operations require a ThreadGroup for proper synchronization:
  // - For signals (ready_to_recv, ready_to_send): sync before leader writes
  // - For waits (wait_ready_to_send, wait_ready_to_recv): all threads poll for
  //   better latency

  /**
   * wait_ready_to_send - Block until buffer is available for writing
   *
   * Spins until receiver has consumed previous data and marked buffer ready.
   * All threads poll for lower latency.
   *
   * @param group ThreadGroup for cooperative processing
   * @param timeout Timeout config (default: no timeout)
   */
  __device__ __forceinline__ void wait_ready_to_send(
      ThreadGroup& group,
      const Timeout& timeout = Timeout()) const;

  /**
   * wait_ready_to_recv - Block until data for specific step is ready
   *
   * Spins until sender has written data and signaled ready.
   * All threads poll for lower latency.
   *
   * @param group ThreadGroup for cooperative processing
   * @param stepId The step identifier to wait for
   * @param timeout Timeout config (default: no timeout)
   */
  __device__ __forceinline__ void wait_ready_to_recv(
      ThreadGroup& group,
      std::size_t stepId,
      const Timeout& timeout = Timeout()) const;

  /**
   * ready_to_recv - Signal that data is ready for receiver
   *
   * Transitions state from READY_TO_SEND to READY_TO_RECV.
   * Syncs all threads, then leader writes stepId.
   *
   * @param group ThreadGroup for cooperative processing
   * @param stepId The step identifier for this data
   */
  __device__ __forceinline__ void ready_to_recv(
      ThreadGroup& group,
      std::size_t stepId);

  /**
   * ready_to_send - Signal that buffer can be reused by sender
   *
   * Transitions state from READY_TO_RECV to READY_TO_SEND.
   * Syncs all threads, then leader writes READY_TO_SEND.
   *
   * @param group ThreadGroup for cooperative processing
   */
  __device__ __forceinline__ void ready_to_send(ThreadGroup& group);

 private:
  __device__ __forceinline__ int32_t load() const {
    return comms::device::ld_acquire_sys_global(const_cast<int32_t*>(&value_));
  }

  __device__ __forceinline__ void store(int32_t v) {
    comms::device::st_release_sys_global(&value_, v);
  }
};

static_assert(
    alignof(ChunkState) == 128,
    "ChunkState must be 128-byte aligned");

// =============================================================================
// Thread-Group-Safe ChunkState Implementation
// =============================================================================
//
// These implementations require ThreadGroup to be fully defined, so they
// are placed after the namespace closes and ThreadGroup.cuh is included.
// This avoids a circular dependency (ThreadGroup doesn't need ChunkState).

__device__ __forceinline__ void ChunkState::wait_ready_to_send(
    ThreadGroup& group,
    const Timeout& timeout) const {
  // All threads poll: slightly lower latency for small messages
  // (avoids sync barrier overhead after leader-only poll)
  while (load() != READY_TO_SEND) {
    TIMEOUT_TRAP_IF_EXPIRED(
        timeout,
        group,
        "ChunkState::wait_ready_to_send waiting for READY_TO_SEND (current=%d)",
        load());
  }
}

__device__ __forceinline__ void ChunkState::wait_ready_to_recv(
    ThreadGroup& group,
    std::size_t stepId,
    const Timeout& timeout) const {
  // All threads poll for better latency.
  while (true) {
    int current_value = load();
    if (current_value == static_cast<int32_t>(stepId)) {
      return;
    }
    TIMEOUT_TRAP_IF_EXPIRED(
        timeout,
        group,
        "ChunkState::wait_ready_to_recv waiting for stepId=%zu "
        "(current=%d)",
        stepId,
        current_value);
  }
}

__device__ __forceinline__ void ChunkState::ready_to_recv(
    ThreadGroup& group,
    std::size_t stepId) {
  group.sync();
  if (group.is_leader()) {
    store(static_cast<int32_t>(stepId));
  }
}

__device__ __forceinline__ void ChunkState::ready_to_send(ThreadGroup& group) {
  group.sync();
  if (group.is_leader()) {
    store(READY_TO_SEND);
  }
}

} // namespace comms::pipes
