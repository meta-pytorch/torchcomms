// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/uniflow/core/Func.h"
#include "comms/uniflow/core/MpscQueue.h"

namespace uniflow {

/// Queue policy using a lock-free MPSC (multi-producer single-consumer) queue.
///
/// Dispatch (push) is wait-free. Drain pops items one at a time — no locking.
///
/// See EpollEventBase.h for how queue policies are used.
struct LockFreeQueuePolicy {
  void push(Func func) {
    queue_.push(std::move(func));
  }

  bool drain() noexcept {
    bool drained = false;
    while (!queue_.empty()) {
      if (auto f = queue_.pop()) {
        (*f)();
        drained = true;
      }
    }
    return drained;
  }

 private:
  MpscQueue<Func> queue_;
};

} // namespace uniflow
