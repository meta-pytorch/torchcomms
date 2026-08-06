// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace comms::fault_tolerance {

enum class AbortReason : int {
  NONE = 0,
  ABORTED = 1,
  TIMED_OUT = 2,
};

enum class AbortBehavior : int {
  SKIP = 0,
  TRAP = 1,
};

enum class AbortCheckResult : int {
  CONTINUE = 0,
  SKIP = 1,
  TRAP = 2,
};

/**
 * Host-owned state shared with CUDA device code through mapped pinned memory.
 *
 * Both fields are read and written with system-scope atomic operations. The
 * allocation is owned by `Abort`; `AbortDevice` stores only a non-owning mapped
 * pointer to this same state.
 */
struct AbortState {
  /**
   * Encoded `AbortReason`.
   *
   * Starts as `AbortReason::NONE` and may transition once to a valid terminal
   * reason. Host and device writers use compare-exchange from `NONE`, so the
   * first valid terminal reason wins.
   */
  int abort;

  /**
   * Shared default timeout duration in milliseconds.
   *
   * `-1` means unset. Host code may update this value, and device handles read
   * the latest value when starting a device-side timeout.
   */
  int64_t timeoutMs;
};

// Atomic operations require naturally aligned fields. Cacheline padding is a
// performance choice, not a correctness requirement for this shared state.
static_assert(offsetof(AbortState, abort) % alignof(int) == 0);
static_assert(offsetof(AbortState, timeoutMs) % alignof(int64_t) == 0);

struct AbortDevice;

/**
 * Shared abort state for communicator-scoped fault tolerance.
 *
 * `Abort` owns a single host-allocated, CUDA-mapped pinned state object that is
 * visible to both CPU threads and CUDA device code. The host object owns that
 * storage for its full lifetime; device handles returned by `getDeviceHandle()`
 * are non-owning views and must not outlive the `Abort` object.
 *
 * Enabled `Abort` instances use shared state when CUDA pinned allocation and
 * device mapping are available. Host-only environments fall back to ordinary
 * host memory for the same `AbortState` fields so CPU callers can keep using
 * the same API. Disabled instances do not allocate state and every
 * mutating/query operation is a no-op or a non-aborted result. The disabled
 * singleton returned by `createAbort(false)` is intended for code paths that
 * must accept an abort object without enabling fault tolerance.
 *
 * `AbortState::abort` and `AbortState::timeoutMs` are mutable shared fields.
 * Host code and device code access them with system-scope atomic operations so
 * updates from either side become visible to the other side. The abort reason
 * is first-writer-wins: an explicit abort records `AbortReason::ABORTED`; an
 * expired timeout records `AbortReason::TIMED_OUT` only if no earlier valid
 * terminal reason has been recorded. The default timeout duration is also
 * stored in shared state for graph-mode device code; host active-deadline
 * tracking remains host-only.
 */
class Abort final {
 public:
  /**
   * Constructs an abort controller.
   *
   * Enabled controllers use a single `AbortState`. CUDA-capable environments
   * allocate it as host-mapped pinned memory so host and device code can
   * observe the same abort reason. Host-only or non-mappable runtime
   * environments fall back to ordinary host memory while preserving the same
   * host-side state semantics. Disabled controllers are no-op placeholders for
   * callers that must pass an `Abort` object while fault tolerance is disabled.
   */
  explicit Abort(bool enabled, AbortBehavior behavior = AbortBehavior::SKIP);
  ~Abort();
  Abort(const Abort&) = delete;
  Abort& operator=(const Abort&) = delete;
  Abort(Abort&&) = delete;
  Abort& operator=(Abort&&) = delete;

  /**
   * Returns whether this controller is enabled.
   *
   * Disabled controllers never report an abort, active timeout, or default
   * timeout duration. Internally, disabled controllers are represented by a
   * null `AbortState` pointer.
   */
  inline bool isEnabled() const {
    return state_ != nullptr;
  }

  /**
   * Returns the device-side behavior selected for this controller.
   *
   * `SKIP` asks device waits to return without trapping when an abort is
   * observed. `TRAP` asks Prims wait helpers to preserve the legacy device
   * trap behavior. The behavior is captured into each `AbortDevice` handle.
   */
  AbortBehavior behavior() const {
    return behavior_;
  }

  /**
   * Records the first abort reason.
   *
   * The abort state starts as `AbortReason::NONE` and can transition exactly
   * once to one valid terminal reason. The first writer wins. Later calls with
   * a different reason do not override the reason already visible to other host
   * threads and device consumers. Valid terminal reasons are
   * `AbortReason::ABORTED` and `AbortReason::TIMED_OUT`; `AbortReason::NONE`
   * and unknown enum values are invalid input and are rejected before
   * attempting to update shared state.
   */
  void setAbort(AbortReason newReason = AbortReason::ABORTED);

  /**
   * Returns true when an explicit abort or expired active timeout has aborted
   * this controller.
   *
   * This also checks the active deadline and records `AbortReason::TIMED_OUT`
   * when the timeout is the first abort reason.
   */
  bool isAborted();

  /**
   * Returns whether a per-operation timeout deadline is currently active.
   *
   * This does not imply that the deadline has expired.
   */
  bool isTimeoutActive() const;

  /**
   * Returns true only when the recorded abort reason is timeout.
   *
   * If the active deadline has expired, this attempts to record
   * `AbortReason::TIMED_OUT`. If an explicit abort already won the race, this
   * returns false.
   */
  bool isTimedOut();

  /**
   * Returns the time remaining before the active timeout deadline.
   *
   * Returns `-1ms` when no timeout is active and `0ms` after the active
   * deadline has expired.
   */
  std::chrono::milliseconds getTimeRemaining();

  /**
   * Starts or replaces the active per-operation timeout deadline.
   *
   * The deadline is computed from the current steady-clock time plus
   * `duration`.
   */
  void startTimeout(std::chrono::milliseconds duration);

  /**
   * Cancels the active per-operation timeout deadline.
   *
   * This does not clear an abort reason that has already been recorded.
   */
  void cancelTimeout();

  /**
   * Stores the default timeout duration.
   *
   * GPE applies this as a per-iteration deadline when no per-operation timeout
   * is supplied. This is only a stored duration; it does not start a deadline.
   */
  void setDefaultTimeout(std::chrono::milliseconds duration);

  /**
   * Returns the default timeout duration when one has been configured.
   *
   * Returns `std::nullopt` for disabled controllers or before a default
   * timeout has been set.
   */
  std::optional<std::chrono::milliseconds> getDefaultTimeout() const;

  /**
   * Returns a non-owning device view over the shared abort state.
   *
   * The returned handle is valid only while this `Abort` instance remains
   * alive. Disabled abort objects return a disabled no-op device handle.
   * Enabled abort objects must be backed by CUDA-mapped pinned state; this
   * throws `std::runtime_error` when the shared state is host-only or when CUDA
   * cannot map the host state for the current device. The handle captures the
   * current device mapping and clock rate, so kernels must consume it on the
   * same CUDA device that was current when the handle was created. Create a new
   * handle after switching devices.
   */
  AbortDevice getDeviceHandle() const;

 private:
  static constexpr int encode(AbortReason reason) {
    return static_cast<int>(reason);
  }

  static constexpr bool isValidTerminalReason(AbortReason reason) {
    switch (reason) {
      case AbortReason::ABORTED:
      case AbortReason::TIMED_OUT:
        return true;
      case AbortReason::NONE:
        return false;
    }
    return false;
  }

  int loadAbortReason() const;
  void markAbort(AbortReason newReason);

  AbortState* state_{nullptr};
  bool stateMapped_{false};
  std::atomic<bool> hasTimeout_{false};
  std::atomic<std::chrono::steady_clock::time_point> deadline_{
      std::chrono::steady_clock::time_point{}};
  AbortBehavior behavior_{AbortBehavior::SKIP};

  static_assert(std::atomic<bool>::is_always_lock_free);
  static_assert(
      std::atomic<std::chrono::steady_clock::time_point>::is_always_lock_free);
};

/**
 * Creates an enabled abort controller or returns the shared disabled
 * controller.
 */
std::shared_ptr<Abort> createAbort(
    bool enabled,
    AbortBehavior behavior = AbortBehavior::SKIP);

} // namespace comms::fault_tolerance
