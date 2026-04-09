// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <mutex>
#include <vector>

#include "comms/uniflow/core/Func.h"

namespace uniflow {

/// Queue policy using a mutex-guarded double-buffered vector.
///
/// Dispatch (push) acquires the mutex and appends. Drain swaps the queue
/// under the lock and executes outside — minimizing lock hold time.
///
/// See EpollEventBase.h for how queue policies are used.
struct MutexQueuePolicy {
  void push(Func func) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(std::move(func));
  }

  bool drain() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.swap(swapBuf_);
    }
    if (swapBuf_.empty()) {
      return false;
    }
    for (auto& f : swapBuf_) {
      f();
    }
    swapBuf_.clear();
    return true;
  }

 private:
  std::mutex mutex_;
  std::vector<Func> queue_;
  std::vector<Func> swapBuf_;
};

} // namespace uniflow
