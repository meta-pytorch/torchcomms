// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#ifndef __HIP_PLATFORM_AMD__
#include <cuda/atomic>
#endif

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "comms/common/fault_tolerance/Abort.h"

namespace comms::fault_tolerance {

namespace detail {

inline int hostCurrentDevice() {
  int device = 0;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        "cudaGetDevice failed for Abort::getDeviceHandle(): " +
        std::string(cudaGetErrorString(status)));
  }
  return device;
}

__device__ __forceinline__ uint64_t deviceClock() {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  return wall_clock64();
#elif defined(__CUDA_ARCH__)
  return clock64();
#else
  return 0;
#endif
}

inline uint64_t hostDeviceCyclesPerMs(int device) {
#ifdef __HIP_PLATFORM_AMD__
  (void)device;
  // TODO: Revisit this when AMD device-side abort timeout support is added.
  return 100000;
#else
  constexpr int kMaxCachedDevices = 64;
  thread_local std::array<uint64_t, kMaxCachedDevices> cachedCyclesPerMs{};
  const bool cacheableDevice = device >= 0 && device < kMaxCachedDevices;
  if (cacheableDevice && cachedCyclesPerMs[device] != 0) {
    return cachedCyclesPerMs[device];
  }

  int clockRateKHz = 0;
  auto status =
      cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, device);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        "cudaDeviceGetAttribute(cudaDevAttrClockRate) failed for Abort::getDeviceHandle(): " +
        std::string(cudaGetErrorString(status)));
  }
  const auto cyclesPerMs = static_cast<uint64_t>(clockRateKHz);
  if (cacheableDevice) {
    cachedCyclesPerMs[device] = cyclesPerMs;
  }
  return cyclesPerMs;
#endif
}

__device__ __forceinline__ bool deviceIsValidTerminalReason(
    AbortReason reason) {
  switch (reason) {
    case AbortReason::ABORTED:
    case AbortReason::TIMED_OUT:
      return true;
    case AbortReason::NONE:
      return false;
  }
  return false;
}

template <typename T>
__device__ __forceinline__ T deviceLoadAcquireSystem(T* value) {
#ifdef __HIP_PLATFORM_AMD__
  return __atomic_load_n(value, __ATOMIC_ACQUIRE);
#else
  return cuda::atomic_ref<T, cuda::thread_scope_system>{*value}.load(
      cuda::memory_order_acquire);
#endif
}

__device__ __forceinline__ bool
deviceCompareExchangeSystem(int* value, int* expected, int desired) {
#ifdef __HIP_PLATFORM_AMD__
  return __atomic_compare_exchange_n(
      value, expected, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
#else
  return cuda::atomic_ref<int, cuda::thread_scope_system>{*value}
      .compare_exchange_strong(
          *expected,
          desired,
          cuda::memory_order_acq_rel,
          cuda::memory_order_acquire);
#endif
}

} // namespace detail

/**
 * Non-owning CUDA device view of an `Abort` object's shared state.
 *
 * The owning `Abort` object allocates and frees the mapped pinned `AbortState`.
 * Kernels may copy this handle by value, but the handle must not be used after
 * the owning `Abort` is destroyed. All reads and writes of shared state use
 * system-scope atomics so CPU and device updates are mutually visible. Create
 * handles through `Abort::getDeviceHandle()` so the mapped device pointer and
 * timeout clock-rate conversion are initialized together.
 */
struct AbortDevice final {
  /**
   * Creates a disabled no-op device handle.
   *
   * This makes `AbortDevice` usable in aggregate kernel arguments and as a
   * temporary compatibility replacement for disabled Prims timeouts.
   */
  __host__ __device__ AbortDevice() = default;

  /**
   * Returns whether this handle references enabled shared abort state.
   *
   * Disabled handles are valid kernel arguments. They have a null state pointer
   * and all mutating/query APIs behave like the host disabled `Abort` object.
   */
  __host__ __device__ bool isEnabled() const {
    return state_ != nullptr;
  }

  /**
   * Starts this handle's device-side timeout from the shared default duration.
   *
   * Non-positive or unset default timeouts leave the device deadline inactive.
   * If converting the timeout to device cycles would overflow, the deadline is
   * set to the current device clock so the next timeout check fails fast rather
   * than silently creating an effectively infinite deadline.
   */
  __device__ void startTimeout() {
    if (!isEnabled()) {
      deadlineCycles_ = 0;
      return;
    }
    const auto timeoutMs = getTimeoutMs();
    if (timeoutMs <= 0 || cyclesPerMs_ == 0) {
      deadlineCycles_ = 0;
      return;
    }
    const auto now = detail::deviceClock();
    const auto timeoutMsU = static_cast<uint64_t>(timeoutMs);
    if (timeoutMsU > (UINT64_MAX - now) / cyclesPerMs_) {
      deadlineCycles_ = now;
      return;
    }
    deadlineCycles_ = now + timeoutMsU * cyclesPerMs_;
  }

  /**
   * Transition alias for Prims `Timeout::start()`.
   *
   * New device code should prefer `startTimeout()`. This exists so timeout
   * shaped Prims waits can migrate to `AbortDevice` without introducing a
   * separate adapter type.
   */
  __device__ void start() {
    startTimeout();
  }

  /**
   * Cancels this handle's active device-side timeout.
   *
   * The deadline is local to this copied handle. Canceling it does not clear a
   * shared abort reason that has already been recorded.
   */
  __device__ void cancelTimeout() {
    deadlineCycles_ = 0;
  }

  /**
   * Returns true when an explicit abort or this handle's expired timeout has
   * recorded a terminal abort reason.
   *
   * If the local deadline has expired, this attempts to record
   * `AbortReason::TIMED_OUT` as the shared reason. The shared reason remains
   * first-writer-wins.
   */
  __device__ bool isAborted() const {
    return checkExpired();
  }

  /**
   * Transition alias for Prims `Timeout::checkExpired()`.
   *
   * Returns true for either an explicit abort or an expired local device
   * timeout. If this handle's local deadline has expired, this records
   * `AbortReason::TIMED_OUT` in the shared state.
   */
  __device__ bool checkExpired() const {
    if (!isEnabled()) {
      return false;
    }
    if (reason() != AbortReason::NONE) {
      return true;
    }
    return markTimedOutIfExpired() || reason() != AbortReason::NONE;
  }

  /**
   * Group-scoped transition alias for Prims `Timeout::checkExpired(group)`.
   *
   * Only the group leader polls shared abort state, matching the current
   * timeout check shape used by Prims wait loops.
   */
  template <typename ThreadGroup>
  __device__ bool checkExpired(const ThreadGroup& group) const {
    return group.is_leader() && checkExpired();
  }

  /**
   * Returns the shared abort reason currently visible to device code.
   *
   * Disabled handles always report `AbortReason::NONE`.
   */
  __device__ AbortReason reason() const {
    if (!isEnabled()) {
      return AbortReason::NONE;
    }
    return static_cast<AbortReason>(
        detail::deviceLoadAcquireSystem(&state_->abort));
  }

  /**
   * Returns the shared default timeout duration in milliseconds.
   *
   * A negative value means no default timeout has been configured.
   */
  __device__ int64_t getTimeoutMs() const {
    if (!isEnabled()) {
      return -1;
    }
    return detail::deviceLoadAcquireSystem(&state_->timeoutMs);
  }

  /**
   * Records the first shared abort reason from device code.
   *
   * Valid terminal reasons are `AbortReason::ABORTED` and
   * `AbortReason::TIMED_OUT`. `AbortReason::NONE` and unknown enum values are
   * invalid; debug/device assert builds catch them, and release-compatible
   * builds return before touching shared state. The CAS only transitions the
   * shared state from `NONE`, so later writers cannot overwrite the first
   * terminal reason.
   */
  __device__ void setAbort(AbortReason newReason = AbortReason::ABORTED) const {
    if (!isEnabled()) {
      return;
    }
    const bool validReason = detail::deviceIsValidTerminalReason(newReason);
    assert(validReason);
    if (!validReason) {
      return;
    }

    int expected = static_cast<int>(AbortReason::NONE);
    detail::deviceCompareExchangeSystem(
        &state_->abort, &expected, static_cast<int>(newReason));
  }

 private:
  friend class Abort;

  /**
   * Creates a device handle for mapped pinned state owned by `Abort`.
   *
   * Passing this handle by value to a kernel passes only this small non-owning
   * view. The shared abort state remains in mapped pinned memory; device code
   * does not receive a private copy of the state. Disabled handles use
   * `state == nullptr`, which makes all public device APIs no-op/non-aborted.
   *
   * `cyclesPerMs` is sampled on the host while creating the handle. CUDA device
   * clocks can change dynamically, so long-running timeouts may be approximate.
   * AMD uses `wall_clock64()`, whose frequency is fixed at 100 MHz; NVIDIA uses
   * `clock64()` and `cudaDevAttrClockRate`.
   *
   * The mapped device pointer and clock conversion are for the CUDA device that
   * was current when `Abort::getDeviceHandle()` created this handle. Launch
   * kernels that consume the handle on that same device, or create a new handle
   * after switching devices.
   *
   * TODO: Evaluate CUPTI APIs for a more accurate timeout conversion source.
   */
  explicit AbortDevice(AbortState* state, uint64_t cyclesPerMs)
      : state_{state}, cyclesPerMs_{cyclesPerMs} {}

  __device__ bool deadlineExpired() const {
    return deadlineCycles_ != 0 && detail::deviceClock() >= deadlineCycles_;
  }

  __device__ bool markTimedOutIfExpired() const {
    if (!isEnabled() || !deadlineExpired()) {
      return false;
    }

    int expected = static_cast<int>(AbortReason::NONE);
    if (detail::deviceCompareExchangeSystem(
            &state_->abort,
            &expected,
            static_cast<int>(AbortReason::TIMED_OUT))) {
      return true;
    }
    return expected == static_cast<int>(AbortReason::TIMED_OUT);
  }

  /**
   * Device pointer to the same mapped pinned state owned by the host `Abort`.
   *
   * This pointer is non-owning and is expected to be non-null for handles
   * returned by `Abort::getDeviceHandle()`.
   */
  AbortState* state_{nullptr};

  /**
   * Device clock cycles per millisecond captured when the handle is created.
   *
   * Device timeout setup uses this to convert the shared default timeout from
   * milliseconds into device clock cycles.
   */
  uint64_t cyclesPerMs_{0};

  /**
   * Per-handle device deadline in `detail::deviceClock()` cycles.
   *
   * A value of zero means no device-side timeout is active. This is local to
   * the copied handle and is not part of the host-owned shared state.
   */
  uint64_t deadlineCycles_{0};
};

inline AbortDevice Abort::getDeviceHandle() const {
  if (state_ == nullptr) {
    return AbortDevice{/*state=*/nullptr, /*cyclesPerMs=*/0};
  }
  if (!stateMapped_) {
    throw std::runtime_error(
        "Abort::getDeviceHandle() requires CUDA host-mapped Abort state");
  }
  const auto device = detail::hostCurrentDevice();
  void* deviceState = nullptr;
  auto status = cudaHostGetDevicePointer(&deviceState, state_, 0);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        "cudaHostGetDevicePointer failed for Abort::getDeviceHandle(): " +
        std::string(cudaGetErrorString(status)));
  }
  return AbortDevice{
      static_cast<AbortState*>(deviceState),
      detail::hostDeviceCyclesPerMs(device)};
}

} // namespace comms::fault_tolerance
