// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>

namespace comms::fault_tolerance {

enum class AbortReason : int {
  NONE = 0,
  ABORTED = 1,
  TIMED_OUT = 2,
};

class Abort final {
 public:
  static constexpr int encode(AbortReason reason) {
    return static_cast<int>(reason);
  }

  /**
   * Constructs an abort controller.
   *
   * Enabled controllers honor abort and timeout operations. Disabled
   * controllers are no-op placeholders for callers that must pass an abort
   * object while fault tolerance is disabled.
   */
  explicit Abort(bool enabled) : enabled_(enabled) {}
  ~Abort() = default;

  /**
   * Returns whether this controller is enabled.
   *
   * Disabled controllers never report an abort or active timeout.
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
   * threads. Valid terminal reasons are `AbortReason::ABORTED` and
   * `AbortReason::TIMED_OUT`; `AbortReason::NONE` and unknown enum values are
   * invalid input and are rejected before attempting to update shared state.
   */
  inline void setAbort(AbortReason reason = AbortReason::ABORTED) {
    if (!enabled_) {
      return;
    }
    if (!isValidTerminalReason(reason)) {
      throw std::invalid_argument("Abort reason must be ABORTED or TIMED_OUT");
    }
    int expected = encode(AbortReason::NONE);
    abort_.compare_exchange_strong(
        expected,
        encode(reason),
        std::memory_order_acq_rel,
        std::memory_order_acquire);
  }

  /**
   * Returns true when an explicit abort or expired active timeout has aborted
   * this controller.
   *
   * This also checks the active deadline and records `AbortReason::TIMED_OUT`
   * if the timeout is the first abort reason.
   */
  inline bool isAborted() {
    if (!enabled_) {
      return false;
    }

    if (abort_.load(std::memory_order_acquire) != encode(AbortReason::NONE)) {
      return true;
    }

    if (!hasTimeout_.load(std::memory_order_acquire)) {
      return false;
    }

    return isTimedOut();
  }

  /**
   * Returns whether a per-operation timeout deadline is currently active.
   *
   * This does not imply that the deadline has expired.
   */
  inline bool isTimeoutActive() const {
    return hasTimeout_.load(std::memory_order_acquire);
  }

  /**
   * Returns true only when the recorded abort reason is timeout.
   *
   * If the active deadline has expired, this attempts to record
   * `AbortReason::TIMED_OUT`. If an explicit abort already won the race, this
   * returns false.
   */
  inline bool isTimedOut() {
    if (abort_.load(std::memory_order_acquire) ==
        encode(AbortReason::TIMED_OUT)) {
      return true;
    }

    if (!hasTimeout_.load(std::memory_order_acquire)) {
      return false;
    }

    // Check for timeout if timeout is set
    auto now = std::chrono::steady_clock::now();
    if (now >= deadline_.load(std::memory_order_acquire)) {
      int expected = encode(AbortReason::NONE);
      if (abort_.compare_exchange_strong(
              expected,
              encode(AbortReason::TIMED_OUT),
              std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        return true;
      }
      return expected == encode(AbortReason::TIMED_OUT);
    }

    return false;
  }

  /**
   * Returns the time remaining before the active timeout deadline.
   *
   * Returns `-1ms` when no timeout is active and `0ms` after the active
   * deadline has expired.
   */
  inline std::chrono::milliseconds getTimeRemaining() {
    if (!enabled_) {
      return std::chrono::milliseconds{-1};
    }

    if (!hasTimeout_.load(std::memory_order_acquire)) {
      return std::chrono::milliseconds{-1};
    }

    auto now = std::chrono::steady_clock::now();
    auto deadline = deadline_.load(std::memory_order_acquire);
    if (now >= deadline) {
      return std::chrono::milliseconds{0};
    }

    return std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - now);
  }

  /**
   * Starts or replaces the active per-operation timeout deadline.
   *
   * The deadline is computed from the current steady-clock time plus
   * `duration`.
   */
  inline void startTimeout(std::chrono::milliseconds duration) {
    if (!enabled_) {
      return;
    }

    auto deadline = std::chrono::steady_clock::now() + duration;
    deadline_.store(deadline, std::memory_order_release);
    hasTimeout_.store(true, std::memory_order_release);
  }

  /**
   * Cancels the active per-operation timeout deadline.
   *
   * This does not clear an abort reason that has already been recorded.
   */
  inline void cancelTimeout() {
    if (!enabled_) {
      return;
    }

    hasTimeout_.store(false, std::memory_order_release);
  }

  /**
   * Stores the default timeout duration.
   *
   * GPE applies this as a per-iteration deadline when no per-operation timeout
   * is supplied. This is only a stored duration; it does not start a deadline.
   */
  inline void setDefaultTimeout(std::chrono::milliseconds duration) {
    if (!enabled_) {
      return;
    }

    timeoutMs_.store(duration.count(), std::memory_order_release);
  }

  /**
   * Returns the default timeout duration when one has been configured.
   *
   * Returns `std::nullopt` for disabled controllers or before a default
   * timeout has been set.
   */
  inline std::optional<std::chrono::milliseconds> getDefaultTimeout() const {
    if (!enabled_) {
      return std::nullopt;
    }

    auto v = timeoutMs_.load(std::memory_order_acquire);
    if (v < 0) {
      return std::nullopt;
    }
    return std::chrono::milliseconds{v};
  }

 private:
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

  const bool enabled_;

  std::atomic<int> abort_{encode(AbortReason::NONE)};
  std::atomic<bool> hasTimeout_{false};
  std::atomic<std::chrono::steady_clock::time_point> deadline_{
      std::chrono::steady_clock::time_point{}};
  // -1 = unset.
  std::atomic<int64_t> timeoutMs_{-1};

  static_assert(std::atomic<bool>::is_always_lock_free);
  static_assert(std::atomic<int>::is_always_lock_free);
  static_assert(
      std::atomic<std::chrono::steady_clock::time_point>::is_always_lock_free);
  static_assert(std::atomic<int64_t>::is_always_lock_free);
};

/**
 * Creates an enabled abort controller or returns the shared disabled
 * controller.
 */
std::shared_ptr<Abort> createAbort(bool enabled);

} // namespace comms::fault_tolerance
