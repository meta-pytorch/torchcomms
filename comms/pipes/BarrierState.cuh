// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include "comms/common/BitOps.cuh"
#include "comms/pipes/SignalState.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes {

/**
 * BarrierState - A reusable arrival-wait barrier for GPU synchronization
 *
 * This barrier implements a split-phase synchronization primitive where
 * participants first "arrive" to signal they've reached the barrier, then
 * "wait" until expected arrival have occurred.
 *
 * The barrier is reusable: each wait() call increments the expected counter,
 * allowing the barrier to be used multiple times without resetting.
 *
 * Memory layout is 128-byte aligned to avoid false sharing between barriers
 * and to ensure optimal memory access patterns on GPU.
 *
 * Typical usage pattern for two-sided synchronization:
 *   GPU A: remoteBarrierState.arrive(); localBarrierState.wait();
 *   GPU B: remoteBarrierState.arrive(); localBarrierState.wait();
 */
struct alignas(128) BarrierState {
  SignalState current_counter_; // Tracks total arrivals
  SignalState expected_counter_; // Tracks expected arrivals for waiting

  __host__ __device__ BarrierState() {}

  /**
   * arrive - Signal arrival at the barrier (single thread)
   *
   * Atomically increments the arrival counter to signal that this participant
   * has reached the barrier. This is the "signal" phase of the barrier.
   *
   * Thread-safe: Can be called concurrently from multiple threads.
   * Non-blocking: Returns immediately after incrementing the counter.
   */
  __device__ __forceinline__ void arrive() {
    current_counter_.atomic_fetch_add(1);
  }

  /**
   * wait - Wait for an arrival at the barrier (single thread)
   *
   * Atomically increments the expected counter and spins until the arrival
   * counter reaches the new expected value. This is the "wait" phase.
   *
   * The increment-then-wait pattern makes the barrier reusable:
   * - 1st wait: expects current_counter >= 1
   * - 2nd wait: expects current_counter >= 2
   * - etc.
   *
   * Blocking: Spins until the condition is met (or timeout expires).
   * Warning: Only one thread should call wait() per synchronization round.
   *
   * @param timeout Optional timeout (default: no timeout, infinite wait)
   */
  __device__ __forceinline__ void wait(const Timeout& timeout = Timeout()) {
    uint64_t expected = expected_counter_.atomic_fetch_add(1) + 1;
    while (current_counter_.load() < expected) {
      TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
          timeout,
          "BarrierState::wait timed out (expected=%llu, current=%llu)",
          static_cast<unsigned long long>(expected),
          static_cast<unsigned long long>(current_counter_.load()));
    }
  }

  /**
   * arrive - Signal arrival at the barrier (thread group)
   *
   * Synchronizes the thread group, then has the leader thread signal arrival.
   * The group.sync() ensures all threads in the group have completed their
   * work before the arrival is signaled.
   *
   * @param group ThreadGroup for cooperative synchronization
   *
   * All threads in the group must call this function (collective operation).
   */
  __device__ __forceinline__ void arrive(ThreadGroup& group) {
    group.sync();
    if (group.is_leader()) {
      arrive();
    }
  }

  /**
   * wait - Wait for an arrival at the barrier (thread group)
   *
   * The leader thread waits for the arrival, then synchronizes the group.
   * The group.sync() at the end ensures all threads observe the completed
   * barrier before proceeding.
   *
   * @param group ThreadGroup for cooperative synchronization
   * @param timeout Optional timeout (default: no timeout, infinite wait)
   *
   * All threads in the group must call this function (collective operation).
   */
  __device__ __forceinline__ void wait(
      ThreadGroup& group,
      const Timeout& timeout = Timeout()) {
    if (group.is_leader()) {
      wait(timeout);
    }
    group.sync();
  }
};

/**
 * getBarrierBufferSize - Calculate buffer size for multiple barriers
 *
 * Computes the total memory needed to store 'count' BarrierState objects,
 * aligned to 128 bytes for optimal GPU memory access.
 *
 * @param count Number of barriers to allocate
 * @return Size in bytes, aligned to 128-byte boundary
 */
__host__ __device__ __forceinline__ std::size_t getBarrierBufferSize(
    int count) {
  return bitops::alignUp(count * sizeof(BarrierState), 128);
}

} // namespace comms::pipes
