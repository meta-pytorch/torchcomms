// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>

namespace ctran::utils {

class Abort final {
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
    abort_.store(true, std::memory_order_release);
  }

  inline bool Test() const {
    if (!enabled_) {
      return false;
    }

    // Check if abort was explicitly set
    if (abort_.load(std::memory_order_acquire)) {
      return true;
    }

    // Check for timeout if timeout is set
    if (hasTimeout_.load(std::memory_order_acquire)) {
      auto now = std::chrono::steady_clock::now();
      return now >= timeoutTime_.load(std::memory_order_acquire);
    }

    return false;
  }

  inline void SetTimeout(std::chrono::milliseconds duration) {
    if (!enabled_) {
      return;
    }

    auto timeoutTime = std::chrono::steady_clock::now() + duration;
    timeoutTime_.store(timeoutTime, std::memory_order_release);
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

 private:
  const bool enabled_;

  std::atomic<bool> abort_{false};
  std::atomic<bool> hasTimeout_{false};
  std::atomic<std::chrono::steady_clock::time_point> timeoutTime_{
      std::chrono::steady_clock::time_point{}};

  static_assert(std::atomic<bool>::is_always_lock_free);
  static_assert(
      std::atomic<std::chrono::steady_clock::time_point>::is_always_lock_free);
};

std::shared_ptr<Abort> createAbort(bool enabled);

} // namespace ctran::utils
