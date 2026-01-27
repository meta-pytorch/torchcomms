// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// @lint-ignore-every CLANGTIDY facebook-modularize-issue-check

#pragma once

#include <cstddef>
#include <cstdint>
#include "comms/common/AtomicUtils.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

// Forward declaration for thread-safe overloads
struct ThreadGroup;

/**
 * Number of fast spins before exponential backoff.
 *
 * Tuning rationale:
 * - Too low: Unnecessary backoff latency for quick operations
 * - Too high: Wastes cycles and increases NVLink polling pressure
 * - 16 provides good balance for NVLink transfers (empirically validated)
 *
 * This can be tuned based on workload characteristics:
 * - Increase for latency-sensitive small messages
 * - Decrease for bandwidth-bound large transfers
 */
constexpr int kFastSpinCount = 16;

namespace detail {

/**
 * Poll with exponential backoff until condition is satisfied.
 *
 * Implements a three-phase polling strategy:
 * 1. Fast path: immediate check (for conditions often true on first check)
 * 2. Fast spin: kFastSpinCount iterations without backoff
 * 3. Slow path: exponential backoff with nanosleep (or threadfence fallback)
 *
 * @tparam Condition Callable returning bool - true when ready
 * @param cond The condition to poll for
 */
template <typename Condition>
__device__ __forceinline__ void poll_with_backoff(Condition&& cond) {
  // Fast path: check immediately without backoff (common for small messages)
  if (cond()) {
    return;
  }

  // Fast spin without backoff
  for (int i = 0; i < kFastSpinCount; ++i) {
    if (cond()) {
      return;
    }
  }

  // Slow path: exponential backoff for longer waits
  // This reduces NVLink polling pressure for large transfers
  int backoff = 1;
  while (!cond()) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(backoff * 100);
#else
    for (int j = 0; j < backoff; ++j) {
      comms::device::threadfence();
    }
#endif
    backoff = (backoff * 2 < 64) ? backoff * 2 : 64; // Cap at 6.4us
  }
}

} // namespace detail

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
 * CALL INDEX FOR MULTI-CALL SAFETY:
 * ==================================
 * In send_one/recv_one, ChunkState[0] is used to transfer metadata (nbytes,
 * offset, has_more). ALL thread-groups synchronize on ChunkState[0] to
 * read this metadata before processing the data transfer.
 *
 * Problem: When send_one/recv_one is called multiple times in the same kernel,
 * idle thread groups (those not assigned to any chunk) may race ahead and see
 * stale stepId values from previous calls. Since all thread-groups share
 * ChunkState[0], a fast thread-group completing call N could see the stepId
 * from call N+1 before the sender has updated the metadata for call N+1.
 *
 * Solution: The call_index_ field disambiguates calls:
 * - ready_to_recv(group, stepId, call_index) - writes call_index before stepId
 * - wait_ready_to_recv(group, stepId, call_index) - waits for stepId, then
 *   verifies call_index matches; if not, retries until correct call arrives
 *
 * MEMORY LAYOUT:
 * - Bytes 0-3: value_ (int) - sync state
 * - Bytes 4-7: call_index_ (uint32_t) - call disambiguation
 * - Bytes 8-8: has_more (bool) - whether more chunks are coming
 * - Bytes 9-15: padding
 * - Bytes 16-23: nbytes (size_t) - current chunk size
 * - Bytes 24-31: offset (size_t) - current chunk offset
 * - Bytes 32-127: padding for cache line isolation
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
  uint32_t call_index_; // 4 bytes - call disambiguation for multi-call safety
  bool has_more; // 1 byte - whether more chunks are coming after this one
  size_t nbytes; // 8 bytes - current chunk size (for send_multiple)
  size_t offset; // 8 bytes - current chunk offset (for send_multiple)
  char
      padding_[128 - 2 * sizeof(int32_t) - sizeof(bool) - 2 * sizeof(size_t)]{};

  __host__ __device__ ChunkState()
      : value_(READY_TO_SEND),
        call_index_(0),
        has_more(false),
        nbytes(0),
        offset(0) {}

  // ===========================================================================
  // Core Operations (Thread-Group-Safe)
  // ===========================================================================
  //
  // All operations require a ThreadGroup for proper synchronization:
  // - For signals (ready_to_recv, ready_to_send): sync before leader writes
  // - For waits (wait_ready_to_send, wait_ready_to_recv): all threads poll for
  //   better latency
  //
  // The call_index parameter disambiguates multiple calls to send_one/recv_one
  // in the same kernel.

  /**
   * wait_ready_to_send - Block until buffer is available for writing
   *
   * Spins until receiver has consumed previous data and marked buffer ready.
   * All threads poll for lower latency.
   *
   * Optimization: Fast-path for small messages where the condition is often
   * immediately true. Only applies exponential backoff after initial spins
   * fail, to avoid adding latency for quick operations.
   *
   * @param group ThreadGroup for cooperative processing
   * Uses acquire semantics to ensure receiver's reads completed.
   */
  __device__ __forceinline__ void wait_ready_to_send(ThreadGroup& group) const;
  __device__ __forceinline__ void waitReadyToSend() const {
    detail::poll_with_backoff([this]() { return load() == READY_TO_SEND; });
  }

  /**
   * wait_ready_to_recv - Block until data for specific step is ready
   *
   * Spins until sender has written data and signaled ready.
   * All threads poll for lower latency.
   *
   * @param group ThreadGroup for cooperative processing
   * @param stepId The step identifier to wait for
   * @param call_index Call index for multi-call disambiguation
   */
  __device__ __forceinline__ void wait_ready_to_recv(
      ThreadGroup& group,
      std::size_t stepId,
      uint32_t call_index) const;

  /**
   * ready_to_recv - Signal that data is ready for receiver
   *
   * Transitions state from READY_TO_SEND to READY_TO_RECV.
   * Syncs all threads, then leader writes call_index and stepId.
   *
   * @param group ThreadGroup for cooperative processing
   * @param stepId The step identifier for this data
   * @param call_index Call index for multi-call disambiguation
   */
  __device__ __forceinline__ void
  ready_to_recv(ThreadGroup& group, std::size_t stepId, uint32_t call_index);

  /**
   * ready_to_send - Signal that buffer can be reused by sender
   *
   * Transitions state from READY_TO_RECV to READY_TO_SEND.
   * Syncs all threads, then leader writes READY_TO_SEND.
   * Spins until sender has written data and signaled ready.
   * Uses acquire semantics to ensure sender's writes are visible.
   *
   * Optimization: Fast-path for small messages where the condition is often
   * immediately true. Only applies exponential backoff after initial spins
   * fail, to avoid adding latency for quick operations.
   *
   * @param stepId The step identifier to wait for
   */
  __device__ __forceinline__ void waitReadyToRecv(std::size_t stepId) const {
    const int32_t targetStepId = static_cast<int32_t>(stepId);
    detail::poll_with_backoff(
        [this, targetStepId]() { return load() == targetStepId; });
  }

  /**
   * isReadyToRecv - Non-blocking check if data for step is ready
   *
   * @param group ThreadGroup for cooperative processing
   */
  __device__ __forceinline__ void ready_to_send(ThreadGroup& group);

  /**
   * write_metadata - Write metadata fields (leader only)
   *
   * @param group ThreadGroup for cooperative processing
   * @param nbytes_val Number of bytes in this chunk
   * @param offset_val Offset in output buffer
   * @param has_more_val Whether more chunks are coming
   */
  __device__ __forceinline__ void write_metadata(
      ThreadGroup& group,
      std::size_t nbytes_val,
      std::size_t offset_val,
      bool has_more_val);

  /**
   * read_metadata - Read metadata fields
   *
   * @param group ThreadGroup for cooperative processing
   * @param nbytes_val Output: number of bytes in this chunk
   * @param offset_val Output: offset in output buffer
   * @param has_more_val Output: whether more chunks are coming
   */
  __device__ __forceinline__ void read_metadata(
      ThreadGroup& group,
      std::size_t& nbytes_val,
      std::size_t& offset_val,
      bool& has_more_val) const;

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
    ThreadGroup& group) const {
  detail::poll_with_backoff([this]() { return load() == READY_TO_SEND; });
}

__device__ __forceinline__ void ChunkState::wait_ready_to_recv(
    ThreadGroup& group,
    std::size_t stepId,
    uint32_t call_index) const {
  const int32_t targetStepId = static_cast<int32_t>(stepId);
  detail::poll_with_backoff([this, targetStepId, call_index]() {
    return load() == targetStepId && call_index_ == call_index;
  });
}

__device__ __forceinline__ void ChunkState::ready_to_recv(
    ThreadGroup& group,
    std::size_t stepId,
    uint32_t call_index) {
  group.sync();
  if (group.is_leader()) {
    // Write call_index BEFORE release-store of value_.
    // This ensures receiver sees call_index after acquire-load of value_.
    call_index_ = call_index;
    store(static_cast<int32_t>(stepId));
  }
}

__device__ __forceinline__ void ChunkState::ready_to_send(ThreadGroup& group) {
  group.sync();
  if (group.is_leader()) {
    store(READY_TO_SEND);
  }
}

__device__ __forceinline__ void ChunkState::write_metadata(
    ThreadGroup& group,
    std::size_t nbytes_val,
    std::size_t offset_val,
    bool has_more_val) {
  if (group.is_leader()) {
    nbytes = nbytes_val;
    offset = offset_val;
    has_more = has_more_val;
  }
}

__device__ __forceinline__ void ChunkState::read_metadata(
    ThreadGroup& group,
    std::size_t& nbytes_val,
    std::size_t& offset_val,
    bool& has_more_val) const {
  nbytes_val = nbytes;
  offset_val = offset;
  has_more_val = has_more;
}

} // namespace comms::pipes
