// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/SignalState.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

/**
 * DeviceCounter - Local completion counter for tracking operation completion
 *
 * Written to LOCAL memory when operations complete (e.g., source buffer
 * consumed, safe to reuse). This is distinct from Signal which is written
 * to REMOTE peer's memory.
 *
 * SIGNAL VS COUNTER (NCCL semantics):
 * - Signal: Written to REMOTE peer's memory (remote completion notification)
 * - Counter: Written to LOCAL memory (source buffer consumed, safe to reuse)
 *
 * WHY COUNTERS EXIST (ASYNC API SUPPORT):
 * =======================================
 * Current synchronous NVLink APIs (e.g., P2pNvlTransportDevice::send()) block
 * until the source buffer copy is complete, so the source buffer is safe to
 * reuse immediately upon return. In this model, counters aren't strictly
 * required.
 *
 * However, the Pipes library roadmap calls for fully asynchronous send/recv
 * APIs to maximize overlap of I/O and compute:
 *
 *   // Future async API pattern (ChunkIterator):
 *   ChunkIterator it = transport.send(dst, src, nbytes);
 *   while (!it.isCompleted()) {
 *     // Do useful work while transfer progresses
 *   }
 *   // NOW source buffer is safe to reuse
 *
 * With async APIs, the sender needs a mechanism to know when the source buffer
 * has been fully consumed and is safe to reuse. DeviceCounter provides this:
 *
 *   // Internally, after async copy completes:
 *   counter.increment_counter(group, counter_id, 1);
 *
 *   // Caller waits for completion:
 *   counter.wait_counter(group, counter_id, CmpOp::CMP_GE, 1);
 *   // Source buffer now safe to reuse
 *
 * This maps to the BufferState model with chunksWritten/chunksRead tracking
 * for fine-grained progress monitoring in pipelined collectives.
 *
 * USAGE:
 *   // After put completes, counter is incremented
 *   counter.increment_counter(group, counter_id, 1);
 *
 *   // Wait for N operations to complete (source buffers safe to reuse)
 *   counter.wait_counter(group, counter_id, CmpOp::CMP_GE, N);
 */
class DeviceCounter {
 public:
  __host__ __device__ DeviceCounter() = default;

  /**
   * Construct a DeviceCounter for local completion tracking
   *
   * @param counters Local counter buffer
   */
  __host__ __device__ explicit DeviceCounter(DeviceSpan<SignalState> counters)
      : counters_(counters) {}

  // ===========================================================================
  // Counter Operations
  // ===========================================================================

  /**
   * increment_counter - Increment a counter (called when operation completes)
   *
   * Atomically increments the counter to signal local completion.
   * Typically called internally when a put/send operation completes.
   *
   * @param group ThreadGroup for cooperative processing
   * @param counter_id Index of the counter
   * @param value Value to add (default: 1)
   */
  __device__ __forceinline__ void
  increment_counter(ThreadGroup& group, int counter_id, uint64_t value = 1) {
    counters_[counter_id].signal(group, SignalOp::SIGNAL_ADD, value);
  }

  /**
   * wait_counter - Wait for counter to reach a value
   *
   * Polls the counter until the condition is met.
   * Use this to wait for source buffers to be safe to reuse.
   *
   * @param group ThreadGroup for cooperative processing
   * @param counter_id Index of the counter
   * @param cmp Comparison operation (CMP_EQ, CMP_GE, etc.)
   * @param value Value to compare against
   * @param timeout_ns Optional timeout in nanoseconds (default: infinite)
   */
  __device__ __forceinline__ void wait_counter(
      ThreadGroup& group,
      int counter_id,
      CmpOp cmp,
      uint64_t value,
      [[maybe_unused]] uint64_t timeout_ns = UINT64_MAX) {
    // TODO: Implement timeout logic
    counters_[counter_id].wait_until(group, cmp, value);
  }

  // ===========================================================================
  // Non-blocking Read Operations
  // ===========================================================================

  /**
   * read_counter - Read current counter value (non-blocking)
   *
   * @param counter_id Index of the counter
   * @return Current counter value
   */
  __device__ __forceinline__ uint64_t read_counter(int counter_id) {
    return counters_[counter_id].load();
  }

  // ===========================================================================
  // Reset Operations
  // ===========================================================================
  //
  // NOTE: Reset operations are SAFE for counters because counters are
  // LOCAL-only (written by this rank, not remote peers). This differs from
  // DeviceSignal where remote peers write to the inbox, making reset unsafe.
  // See DeviceSignal.cuh for the full explanation.

  /**
   * reset_counter - Reset counter value to 0
   *
   * @param counter_id Index of the counter
   */
  __device__ __forceinline__ void reset_counter(int counter_id) {
    counters_[counter_id].store(0);
  }

  /**
   * reset_all_counters - Reset all counters to 0
   *
   * Resets the entire counter buffer. Useful between phases.
   */
  __device__ __forceinline__ void reset_all_counters() {
    for (uint32_t i = 0; i < counters_.size(); ++i) {
      counters_[i].store(0);
    }
  }

  // ===========================================================================
  // Accessors
  // ===========================================================================

  __device__ __forceinline__ uint32_t counter_count() const {
    return counters_.size();
  }

 private:
  DeviceSpan<SignalState> counters_;
};

/**
 * getCounterBufferSize - Calculate buffer size for counters
 *
 * @param counterCount Number of counter slots
 * @return Size in bytes, aligned to 128-byte boundary
 */
__host__ __device__ __forceinline__ std::size_t getCounterBufferSize(
    int counterCount) {
  return getSignalBufferSize(counterCount);
}

} // namespace comms::pipes
