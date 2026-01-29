// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include "comms/common/AtomicUtils.cuh"
#include "comms/common/BitOps.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes {

enum class SignalOp {
  SIGNAL_SET,
  SIGNAL_ADD,
};

enum class CmpOp {
  CMP_EQ,
  CMP_GT,
  CMP_LT,
  CMP_GE,
  CMP_LE,
  CMP_NE,
};

// Helper to get string representation of CmpOp for error messages
__device__ inline const char* cmpOpToString(CmpOp op) {
  switch (op) {
    case CmpOp::CMP_EQ:
      return "==";
    case CmpOp::CMP_GT:
      return ">";
    case CmpOp::CMP_LT:
      return "<";
    case CmpOp::CMP_GE:
      return ">=";
    case CmpOp::CMP_LE:
      return "<=";
    case CmpOp::CMP_NE:
      return "!=";
    default:
      return "?";
  }
}

/**
 * SignalState - Synchronization primitive for P2P NVLink signaling operations
 *
 * A lightweight signaling primitive with a single 64-bit counter for
 * signal/wait operations between peers.
 *
 * MEMORY LAYOUT:
 * ==============
 * - signal_: 64-bit counter modified by signal() and polled by wait_until()
 *
 * DESIGN:
 * =======
 *
 * The signal_ counter can be set or incremented by the signaling peer (via
 * NVLink remote write), and the waiting peer polls until a condition is met.
 * The caller is responsible for tracking expected values externally.
 *
 * MEMORY SEMANTICS:
 * =================
 * - All reads use acquire ordering (visible after peer's release)
 * - All writes use release ordering (visible to peer after their acquire)
 * - Uses .sys scope for cross-GPU NVLink coherence
 *
 */
struct alignas(128) SignalState {
  uint64_t signal_;

  __host__ __device__ SignalState() : signal_(0) {}

  // ===========================================================================
  // Core Operations
  // ===========================================================================

  __device__ __forceinline__ uint64_t load() const {
    return comms::device::ld_acquire_sys_global(&signal_);
  }

  __device__ __forceinline__ void store(uint64_t value) {
    comms::device::st_release_sys_global(&signal_, value);
  }

  __device__ __forceinline__ uint64_t atomic_fetch_add(uint64_t value) {
    return comms::device::atomic_fetch_add_release_sys_global(&signal_, value);
  }

  /**
   * signal - Modify the signal counter to notify a waiting peer
   *
   * Updates the signal_ counter using the specified operation.
   * When called on a peer's Signal (via NVLink remote pointer), this notifies
   * the peer that a synchronization point has been reached.
   *
   * @param op SIGNAL_SET to store value, SIGNAL_ADD to atomically add value
   * @param value The value to set or add
   */
  __device__ __forceinline__ void signal(SignalOp op, uint64_t value) {
    switch (op) {
      case SignalOp::SIGNAL_SET:
        store(value);
        break;
      case SignalOp::SIGNAL_ADD:
        atomic_fetch_add(value);
        break;
    }
  }

 private:
  /**
   * checkTimeoutAndTrap - Helper to check timeout and trap with error message
   *
   * Used internally by wait_until to avoid code duplication.
   */
  __device__ __forceinline__ void checkTimeoutAndTrap(
      const Timeout& timeout,
      CmpOp op,
      uint64_t expected) const {
    TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
        timeout,
        "SignalState::wait_until waiting for signal %s %llu (current=%llu)",
        cmpOpToString(op),
        static_cast<unsigned long long>(expected),
        static_cast<unsigned long long>(load()));
  }

 public:
  /**
   * wait_until - Wait until the signal counter satisfies a condition
   *
   * Polls the signal_ counter until the specified comparison with the given
   * expected value evaluates to true. The caller is responsible for tracking
   * the expected value externally.
   *
   * Uses acquire semantics to ensure subsequent reads see peer's writes.
   *
   * @param op The comparison operation (CMP_EQ, CMP_GT, CMP_LT, CMP_GE, etc.)
   * @param expected The expected value to compare against
   * @param timeout Timeout config (default: no timeout)
   */
  __device__ __forceinline__ void
  wait_until(CmpOp op, uint64_t expected, const Timeout& timeout = Timeout()) {
    switch (op) {
      case CmpOp::CMP_EQ:
        while (load() != expected) {
          checkTimeoutAndTrap(timeout, op, expected);
        }
        break;
      case CmpOp::CMP_GT:
        while (load() <= expected) {
          checkTimeoutAndTrap(timeout, op, expected);
        }
        break;
      case CmpOp::CMP_LT:
        while (load() >= expected) {
          checkTimeoutAndTrap(timeout, op, expected);
        }
        break;
      case CmpOp::CMP_GE:
        while (load() < expected) {
          checkTimeoutAndTrap(timeout, op, expected);
        }
        break;
      case CmpOp::CMP_LE:
        while (load() > expected) {
          checkTimeoutAndTrap(timeout, op, expected);
        }
        break;
      case CmpOp::CMP_NE:
        while (load() == expected) {
          checkTimeoutAndTrap(timeout, op, expected);
        }
        break;
    }
  }

  // ===========================================================================
  // Thread-Group-Safe Operations
  // ===========================================================================
  // These methods coordinate signal operations across all threads in a group.
  //

  /**
   * signal - Thread-group-safe signal operation
   *
   * Synchronizes all threads in the group, then the leader updates the signal
   * counter. This ensures all prior memory operations from all threads in the
   * group are complete before signaling.
   *
   * @param group ThreadGroup for cooperative processing
   * @param op SIGNAL_SET to store value, SIGNAL_ADD to atomically add value
   * @param value The value to set or add
   */
  __device__ __forceinline__ void
  signal(ThreadGroup& group, SignalOp op, uint64_t value) {
    // Sync to ensure all prior memory operations from all threads are complete
    group.sync();

    // Only leader performs the signal operation
    if (group.is_leader()) {
      signal(op, value);
    }
  }

  /**
   * wait_until - Thread-group-safe wait operation
   *
   * All threads in the group poll until the signal counter satisfies the
   * condition. This provides lower latency than leader-only polling by
   * avoiding a sync barrier after the wait completes.
   *
   * @param group ThreadGroup for cooperative processing
   * @param op The comparison operation (CMP_EQ, CMP_GE, etc.)
   * @param expected The expected value to compare against
   * @param timeout Timeout config (default: no timeout)
   */
  __device__ __forceinline__ void wait_until(
      ThreadGroup& group,
      CmpOp op,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    wait_until(op, expected, timeout);
  }
};

static_assert(alignof(SignalState) == 128, "Signal must be 128-byte aligned");

__host__ __device__ __forceinline__ std::size_t getSignalBufferSize(int count) {
  return bitops::alignUp(count * sizeof(SignalState), 128);
}

} // namespace comms::pipes
