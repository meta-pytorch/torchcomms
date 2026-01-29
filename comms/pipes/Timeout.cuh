// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <cstdio>

namespace comms::pipes {

// Forward declaration - full definition in ThreadGroup.cuh
struct ThreadGroup;

/**
 * Timeout - Stateful timeout for GPU kernel wait operations
 *
 * A timeout object that captures the start time once (via start()) and can
 * then be checked multiple times throughout the kernel's lifetime. This avoids
 * calling clock64() at the start of every wait loop.
 *
 * Usage:
 *   // Host side - create timeout config using makeTimeout()
 *   Timeout timeout = makeTimeout(1000, deviceId);  // 1 second timeout
 *
 *   // Kernel side - start the timeout once at kernel entry
 *   timeout.start();
 *
 *   // Use in wait loops with ThreadGroup - only leader checks
 *   while (!ready) {
 *     timeout.check(group);  // Leader checks and traps if exceeded
 *   }
 *
 * Default constructor creates a "no timeout" config (infinite wait) with
 * zero overhead - check() is a no-op when timeout is disabled.
 *
 * Memory layout: Compact 16-byte struct (2x uint64_t) for efficient GPU usage.
 */
struct Timeout {
  uint64_t timeout_cycles; // Timeout duration in cycles (0 = infinite wait)
  uint64_t deadline_cycles; // Absolute deadline (set by start())

  /**
   * Default constructor - creates "no timeout" config (infinite wait)
   */
  __host__ __device__ Timeout() : timeout_cycles(0), deadline_cycles(0) {}

  /**
   * Constructor with pre-computed timeout in cycles
   *
   * Use makeTimeout() from TimeoutUtils.h instead of calling this directly.
   * That helper queries the GPU clock rate and computes timeout_cycles for you.
   *
   * @param cycles Total timeout duration in GPU clock cycles
   */
  __host__ __device__ explicit Timeout(uint64_t cycles)
      : timeout_cycles(cycles), deadline_cycles(0) {}

  /**
   * Check if timeout is enabled
   */
  __host__ __device__ bool isEnabled() const {
    return timeout_cycles > 0;
  }

  /**
   * Start the timeout timer
   *
   * Call this once at the beginning of the kernel (or at least before any
   * wait operations that use this timeout). Captures the current GPU cycle
   * count and computes the absolute deadline for efficient checking.
   *
   * IMPORTANT: All threads that will call check() must call start() first.
   * Since Timeout is passed by value, each thread has its own copy and
   * captures its own deadline. For ThreadGroup-based check(group), only
   * the leader checks, so only the leader's deadline_cycles is used.
   *
   * Only captures the clock if timeout is enabled (timeout_cycles > 0).
   * Traps if called more than once (programming error) - detected by
   * deadline_cycles already being non-zero.
   */
  __device__ __forceinline__ void start() {
#ifdef __CUDA_ARCH__
    if (timeout_cycles > 0) {
      if (deadline_cycles != 0) {
        printf(
            "CUDA TIMEOUT ERROR: Timeout::start() called twice "
            "(deadline_cycles=%llu)\n",
            static_cast<unsigned long long>(deadline_cycles));
        __trap(); // Double-start is a programming error
      }
      deadline_cycles = clock64() + timeout_cycles;
    }
#endif
  }

  /**
   * Check if timeout has expired (single-threaded version)
   *
   * Compares current clock against the precomputed deadline.
   *
   * IMPORTANT: The calling thread must have called start() first to compute
   * its deadline_cycles.
   *
   * Use this version for single-threaded timeout checking. For multi-threaded
   * use with ThreadGroup, prefer checkExpired(const ThreadGroup&) which only
   * has the leader check for better efficiency.
   *
   * @return true if timeout has expired, false otherwise (or if disabled)
   */
  __device__ __forceinline__ bool checkExpired() const {
#ifdef __CUDA_ARCH__
    if (timeout_cycles > 0) {
      return clock64() > deadline_cycles;
    }
#endif
    return false;
  }

  /**
   * Check if timeout has expired (ThreadGroup version)
   *
   * Only the group leader calls clock64() and checks the deadline.
   * Returns the result to all threads via warp shuffle or shared memory
   * depending on the ThreadGroup scope.
   *
   * This is more efficient than all threads checking: O(1) clock64() calls
   * per group instead of O(N).
   *
   * @param group ThreadGroup for cooperative processing
   * @return true if timeout has expired, false otherwise (or if disabled)
   */
  __device__ __forceinline__ bool checkExpired(const ThreadGroup& group) const;
};

} // namespace comms::pipes

// Include ThreadGroup for the inline implementation of check(ThreadGroup&)
// This is placed after the namespace to avoid circular dependencies
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

__device__ __forceinline__ bool Timeout::checkExpired(
    const ThreadGroup& group) const {
#ifdef __CUDA_ARCH__
  if (timeout_cycles > 0 && group.is_leader()) {
    return clock64() > deadline_cycles;
  }
#else
  (void)group;
#endif
  return false;
}

/**
 * TIMEOUT_TRAP_IF_EXPIRED - Check timeout and trap with error message if
 * expired
 *
 * This macro consolidates the common pattern of checking for timeout
 * expiration, printing a context-specific error message, and calling __trap().
 *
 * Usage:
 *   while (condition) {
 *     TIMEOUT_TRAP_IF_EXPIRED(timeout, group,
 *         "MyClass::my_method waiting for X (current=%d)", current_value);
 *   }
 *
 * @param timeout The Timeout object
 * @param group The ThreadGroup for cooperative checking
 * @param fmt Printf-style format string (without newline)
 * @param ... Format arguments
 */
#ifdef __CUDA_ARCH__
#define TIMEOUT_TRAP_IF_EXPIRED(timeout, group, fmt, ...)     \
  do {                                                        \
    if ((timeout).checkExpired(group)) {                      \
      printf("CUDA TIMEOUT ERROR: " fmt "\n", ##__VA_ARGS__); \
      __trap();                                               \
    }                                                         \
  } while (0)
#else
#define TIMEOUT_TRAP_IF_EXPIRED(timeout, group, fmt, ...) \
  do {                                                    \
    (void)(timeout);                                      \
    (void)(group);                                        \
  } while (0)
#endif

/**
 * TIMEOUT_TRAP_IF_EXPIRED_SINGLE - Single-threaded version (no ThreadGroup)
 *
 * Same as TIMEOUT_TRAP_IF_EXPIRED but for single-threaded timeout checking.
 *
 * @param timeout The Timeout object
 * @param fmt Printf-style format string (without newline)
 * @param ... Format arguments
 */
#ifdef __CUDA_ARCH__
#define TIMEOUT_TRAP_IF_EXPIRED_SINGLE(timeout, fmt, ...)     \
  do {                                                        \
    if ((timeout).checkExpired()) {                           \
      printf("CUDA TIMEOUT ERROR: " fmt "\n", ##__VA_ARGS__); \
      __trap();                                               \
    }                                                         \
  } while (0)
#else
#define TIMEOUT_TRAP_IF_EXPIRED_SINGLE(timeout, fmt, ...) \
  do {                                                    \
    (void)(timeout);                                      \
  } while (0)
#endif

} // namespace comms::pipes
