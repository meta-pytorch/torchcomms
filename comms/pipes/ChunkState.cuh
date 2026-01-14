// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// @lint-ignore-every CLANGTIDY facebook-modularize-issue-check

#pragma once

#include <cstddef>
#include <cstdint>
#include "comms/common/AtomicUtils.cuh"

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
 *                      readyToRecv(stepId)
 *    ┌───────────────┐ ─────────────────────▶ ┌───────────────┐
 *    │ READY_TO_SEND │                        │ READY_TO_RECV │
 *    │     (-1)      │ ◀───────────────────── │   (stepId)    │
 *    └───────────────┘      readyToSend()     └───────────────┘
 *
 * SENDER WORKFLOW:
 *   1. waitReadyToSend()      - Block until state == READY_TO_SEND
 *   2. [copy data to buffer]
 *   3. readyToRecv(stepId)    - Transition to READY_TO_RECV
 *
 * RECEIVER WORKFLOW:
 *   1. waitReadyToRecv(stepId) - Block until state == stepId
 *   2. [copy data from buffer]
 *   3. readyToSend()           - Transition to READY_TO_SEND
 *
 * MEMORY LAYOUT:
 * - First 4 bytes: state value (int32_t)
 * - Remaining 124 bytes: padding for cache line isolation
 * - Total size: 128 bytes (cache line aligned)
 *
 * MEMORY SEMANTICS:
 * - All reads use acquire ordering (visible after peer's release)
 * - All writes use release ordering (visible to peer after their acquire)
 * - Uses .sys scope for cross-GPU NVLink coherence
 */
struct alignas(128) ChunkState {
  static constexpr int32_t READY_TO_SEND = -1;

  int32_t value_;
  char padding_[128 - sizeof(int32_t)]{};

  __host__ __device__ ChunkState() : value_(READY_TO_SEND) {}

  // ===========================================================================
  // Sender Operations
  // ===========================================================================

  /**
   * waitReadyToSend - Block until buffer is available for writing
   *
   * Spins until receiver has consumed previous data and marked buffer ready.
   * Uses acquire semantics to ensure receiver's reads completed.
   */
  __device__ __forceinline__ void waitReadyToSend() const {
    while (load() != READY_TO_SEND) {
    }
  }

  /**
   * isReadyToSend - Non-blocking check if buffer is available
   *
   * @return true if buffer can be written, false otherwise
   */
  __device__ __forceinline__ bool isReadyToSend() const {
    return load() == READY_TO_SEND;
  }

  /**
   * readyToRecv - Signal that data is ready for receiver
   *
   * Transitions state from READY_TO_SEND to READY_TO_RECV.
   * Uses release semantics to ensure data writes are visible to receiver.
   *
   * @param stepId The step identifier for this data
   */
  __device__ __forceinline__ void readyToRecv(std::size_t stepId) {
    store(static_cast<int32_t>(stepId));
  }

  // ===========================================================================
  // Receiver Operations
  // ===========================================================================

  /**
   * waitReadyToRecv - Block until data for specific step is ready
   *
   * Spins until sender has written data and signaled ready.
   * Uses acquire semantics to ensure sender's writes are visible.
   *
   * @param stepId The step identifier to wait for
   */
  __device__ __forceinline__ void waitReadyToRecv(std::size_t stepId) const {
    while (load() != static_cast<int32_t>(stepId)) {
    }
  }

  /**
   * isReadyToRecv - Non-blocking check if data for step is ready
   *
   * @param stepId The step identifier to check
   * @return true if data is ready, false otherwise
   */
  __device__ __forceinline__ bool isReadyToRecv(std::size_t stepId) const {
    return load() == static_cast<int32_t>(stepId);
  }

  /**
   * readyToSend - Signal that buffer can be reused by sender
   *
   * Transitions state from READY_TO_RECV to READY_TO_SEND.
   * Uses release semantics to ensure data reads completed before signal.
   */
  __device__ __forceinline__ void readyToSend() {
    store(READY_TO_SEND);
  }

  // ===========================================================================
  // Thread-Group-Safe Operations (preferred for multi-threaded use)
  // ===========================================================================
  //
  // These overloads accept a ThreadGroup and handle synchronization correctly:
  // - For signals (readyToRecv, readyToSend): sync before leader writes
  // - For waits (waitReadyToSend, waitReadyToRecv): all threads poll for better
  // latency
  //
  // This ensures all threads in the group see consistent state.

  __device__ __forceinline__ void waitReadyToSend(ThreadGroup& group) const;
  __device__ __forceinline__ void waitReadyToRecv(
      ThreadGroup& group,
      std::size_t stepId) const;
  __device__ __forceinline__ void readyToRecv(
      ThreadGroup& group,
      std::size_t stepId);
  __device__ __forceinline__ void readyToSend(ThreadGroup& group);

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

} // namespace comms::pipes

// =============================================================================
// Thread-Group-Safe ChunkState Implementation
// =============================================================================
//
// These implementations require ThreadGroup to be fully defined, so they
// are placed after the namespace closes and ThreadGroup.cuh is included.
// This avoids a circular dependency (ThreadGroup doesn't need ChunkState).

#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

__device__ __forceinline__ void ChunkState::waitReadyToSend(
    ThreadGroup& group) const {
  // All threads poll: slightly lower latency for small messages
  // (avoids sync barrier overhead after leader-only poll)
  while (load() != READY_TO_SEND) {
  }
}

__device__ __forceinline__ void ChunkState::waitReadyToRecv(
    ThreadGroup& group,
    std::size_t stepId) const {
  // All threads poll: slightly lower latency for small messages
  // (avoids sync barrier overhead after leader-only poll)
  while (load() != static_cast<int32_t>(stepId)) {
  }
}

__device__ __forceinline__ void ChunkState::readyToRecv(
    ThreadGroup& group,
    std::size_t stepId) {
  group.sync();
  if (group.is_leader()) {
    store(static_cast<int32_t>(stepId));
  }
}

__device__ __forceinline__ void ChunkState::readyToSend(ThreadGroup& group) {
  group.sync();
  if (group.is_leader()) {
    store(READY_TO_SEND);
  }
}

} // namespace comms::pipes
