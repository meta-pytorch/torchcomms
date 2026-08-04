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

struct AbortState {
  int abort;
  int64_t timeoutMs;
};

// Atomic operations require naturally aligned fields. Cacheline padding is a
// performance choice, not a correctness requirement for this shared state.
static_assert(offsetof(AbortState, abort) % alignof(int) == 0);
static_assert(offsetof(AbortState, timeoutMs) % alignof(int64_t) == 0);

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
  explicit Abort(bool enabled);
  ~Abort();
  Abort(const Abort&) = delete;
  Abort& operator=(const Abort&) = delete;
  Abort(Abort&&) = delete;
  Abort& operator=(Abort&&) = delete;

  /**
   * Returns whether this controller is enabled.
   *
   * Disabled controllers never report an abort, active timeout, or default
   * timeout duration.
   */
  inline bool isEnabled() const {
    return enabled_;
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
  void setAbort(AbortReason reason = AbortReason::ABORTED);

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
  void markAbort(AbortReason reason);

  const bool enabled_;

  AbortState* state_{nullptr};
  bool stateMapped_{false};
  std::atomic<bool> hasTimeout_{false};
  std::atomic<std::chrono::steady_clock::time_point> deadline_{
      std::chrono::steady_clock::time_point{}};

  static_assert(std::atomic<bool>::is_always_lock_free);
  static_assert(
      std::atomic<std::chrono::steady_clock::time_point>::is_always_lock_free);
};

/**
 * Creates an enabled abort controller or returns the shared disabled
 * controller.
 */
std::shared_ptr<Abort> createAbort(bool enabled);

} // namespace comms::fault_tolerance
