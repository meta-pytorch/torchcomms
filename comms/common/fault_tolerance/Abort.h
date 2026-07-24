// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>

namespace comms::fault_tolerance {

class Abort final {
 private:
  enum class AbortReason : int {
    NONE = 0,
    ABORTED = 1,
    TIMED_OUT = 2,
  };

  static constexpr int encode(AbortReason reason) {
    return static_cast<int>(reason);
  }

 public:
  explicit Abort(bool enabled) : enabled_(enabled) {}
  ~Abort() = default;

  inline bool Enabled() const {
    return enabled_;
  }

  inline void Set() {
    if (!enabled_) {
      return;
    }
    int expected = encode(AbortReason::NONE);
    abort_.compare_exchange_strong(
        expected,
        encode(AbortReason::ABORTED),
        std::memory_order_acq_rel,
        std::memory_order_acquire);
  }

  inline bool Test() {
    if (!enabled_) {
      return false;
    }

    if (abort_.load(std::memory_order_acquire) != encode(AbortReason::NONE)) {
      return true;
    }

    if (!hasTimeout_.load(std::memory_order_acquire)) {
      return false;
    }

    return TimedOut();
  }

  inline bool HasTimeout() const {
    return hasTimeout_.load(std::memory_order_acquire);
  }

  inline bool TimedOut() {
    if (!hasTimeout_.load(std::memory_order_acquire)) {
      return false;
    }

    if (abort_.load(std::memory_order_acquire) ==
        encode(AbortReason::TIMED_OUT)) {
      return true;
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

  // Returns the time remaining until the timeout is hit.
  // Returns -1 if no timeout is set.
  inline std::chrono::milliseconds TimeRemaining() {
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

  inline void SetTimeout(std::chrono::milliseconds duration) {
    if (!enabled_) {
      return;
    }

    auto deadline = std::chrono::steady_clock::now() + duration;
    deadline_.store(deadline, std::memory_order_release);
    hasTimeout_.store(true, std::memory_order_release);
  }

  // CancelTimeout resets timeout state.
  // Users can first Test(), then explicitly Set() to preserve state.
  inline void CancelTimeout() {
    if (!enabled_) {
      return;
    }

    hasTimeout_.store(false, std::memory_order_release);
  }

  // Stores a duration; GPE applies it as a per-iteration deadline when no
  // per-op timeout is supplied. Safe to update concurrently.
  inline void SetDefaultTimeoutDuration(std::chrono::milliseconds duration) {
    if (!enabled_) {
      return;
    }

    timeoutMs_.store(duration.count(), std::memory_order_release);
  }

  inline std::optional<std::chrono::milliseconds> GetDefaultTimeoutDuration()
      const {
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

std::shared_ptr<Abort> createAbort(bool enabled);

} // namespace comms::fault_tolerance
