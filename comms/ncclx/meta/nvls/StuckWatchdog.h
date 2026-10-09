// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace ncclx::nvls {

// Observer thread for a blocking call that has no timeout of its own. Invokes
// `onStuck` once per interval until finish() is called; a non-positive interval
// disables the thread entirely and the guarded call runs unobserved.
//
// Deliberately free of NCCL and CUDA dependencies: the call this guards
// (cuMulticastBindMem) needs NVSwitch hardware to reach, so the timing
// behaviour is only testable if it is separable from the call.
class StuckWatchdog {
 public:
  using Callback = std::function<void()>;

  StuckWatchdog(
      std::chrono::nanoseconds interval,
      Callback onStuck,
      Callback onThreadStart = {})
      : interval_{interval},
        onStuck_{std::move(onStuck)},
        onThreadStart_{std::move(onThreadStart)} {
    if (interval_ <= std::chrono::nanoseconds::zero()) {
      return;
    }
    try {
      thread_ = std::thread([this] { run(); });
    } catch (const std::exception& ex) {
      launchError_ = ex.what();
    }
  }

  StuckWatchdog(const StuckWatchdog&) = delete;
  StuckWatchdog& operator=(const StuckWatchdog&) = delete;
  StuckWatchdog(StuckWatchdog&&) = delete;
  StuckWatchdog& operator=(StuckWatchdog&&) = delete;

  ~StuckWatchdog() {
    finish();
  }

  // The thread could not be started. Best-effort by design: the caller reports
  // it and proceeds with the guarded call.
  bool launchFailed() const noexcept {
    return !launchError_.empty();
  }

  const std::string& launchError() const noexcept {
    return launchError_;
  }

  bool running() const noexcept {
    return thread_.joinable();
  }

  // Signals completion and joins. Idempotent. Returns false if the join failed.
  // A failed join is fatal: the thread stays joinable (detaching it would leave
  // it running against a destroyed object), so destroying the watchdog then
  // calls std::terminate, as for any joinable std::thread. Callers should log
  // joinError() before that happens.
  bool finish() {
    if (!thread_.joinable()) {
      return joinError_.empty();
    }
    try {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
      }
      cv_.notify_one();
      thread_.join();
    } catch (const std::exception& ex) {
      joinError_ = ex.what();
      return false;
    } catch (...) {
      joinError_ = "unknown error";
      return false;
    }
    return true;
  }

  const std::string& joinError() const noexcept {
    return joinError_;
  }

 private:
  // A watchdog must never take down the process it observes: an exception
  // escaping a callback would propagate out of the thread's callable and
  // std::terminate. Each callback is guarded separately, so one that throws
  // does not stop the warnings that follow it.
  void run() {
    if (onThreadStart_) {
      try {
        onThreadStart_();
      } catch (...) {
      }
    }
    std::unique_lock<std::mutex> lock(mutex_);
    while (!cv_.wait_for(lock, interval_, [this] { return done_; })) {
      // Released across the callback so one that re-enters this object cannot
      // deadlock. finish() still waits for a running callback to return.
      lock.unlock();
      try {
        onStuck_();
      } catch (...) {
      }
      lock.lock();
    }
  }

  const std::chrono::nanoseconds interval_;
  const Callback onStuck_;
  const Callback onThreadStart_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool done_{false};
  std::string launchError_;
  std::string joinError_;
  std::thread thread_;
};

} // namespace ncclx::nvls
