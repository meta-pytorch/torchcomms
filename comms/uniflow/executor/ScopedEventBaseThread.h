// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <pthread.h>

#include <exception>
#include <string>
#include <thread>

#include "comms/uniflow/executor/LockFreeEventBase.h"
#include "comms/uniflow/executor/MutexEventBase.h" // IWYU pragma: keep
#include "comms/uniflow/logging/Logger.h"

namespace uniflow {

/// A managed thread that owns and runs an EventBase.
///
/// The EventBase loop starts on construction and stops on destruction.
/// Equivalent to folly::ScopedEventBaseThread.
///
/// Usage:
///   ScopedEventBaseThread evbThread("worker");
///   evbThread.getEventBase()->dispatch([] { /* ... */ });
///
template <typename EventBaseImpl = LockFreeEventBase>
class TScopedEventBaseThread {
 public:
  /// Construct and start the thread. The EventBase begins loop()
  /// immediately.
  explicit TScopedEventBaseThread(std::string name = "") {
    thread_ = std::thread([this, n = std::move(name)] {
      if (!n.empty()) {
        pthread_setname_np(pthread_self(), n.substr(0, 15).c_str());
      }
      /*
       * loop() can throw (e.g. EpollEventBase::waitForEvents on an epoll_wait
       * failure, or a resource-allocation path run on this thread). An escaping
       * exception would otherwise reach std::terminate with no diagnostic and
       * take the whole process down silently. Log it (and re-throw to preserve
       * the fail-fast behavior) so the death is attributable.
       */
      try {
        evb_.loop();
      } catch (const std::exception& e) {
        UNIFLOW_LOG_ERROR(
            "EventBase thread '{}' terminating on uncaught exception: {}",
            n,
            e.what());
        throw;
      } catch (...) {
        UNIFLOW_LOG_ERROR(
            "EventBase thread '{}' terminating on unknown exception", n);
        throw;
      }
    });
    // Wait for the thread_ to start the loop.
    evb_.dispatchAndWait([]() noexcept {});
  }

  /// Stop the event loop and join the thread.
  ~TScopedEventBaseThread() {
    evb_.stop();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  // Non-copyable, non-movable
  TScopedEventBaseThread(const TScopedEventBaseThread&) = delete;
  TScopedEventBaseThread& operator=(const TScopedEventBaseThread&) = delete;
  TScopedEventBaseThread(TScopedEventBaseThread&&) = delete;
  TScopedEventBaseThread& operator=(TScopedEventBaseThread&&) = delete;

  /// Returns a pointer to the EventBase interface. Valid for the lifetime
  /// of this TScopedEventBaseThread.
  EventBase* getEventBase() {
    return &evb_;
  }

  /// Returns the thread ID of the managed thread.
  std::thread::id getThreadId() const {
    return thread_.get_id();
  }

 private:
  EventBaseImpl evb_;
  std::thread thread_;
};

using ScopedEventBaseThread = TScopedEventBaseThread<LockFreeEventBase>;

} // namespace uniflow
