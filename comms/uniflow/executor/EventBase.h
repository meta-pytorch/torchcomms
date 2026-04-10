// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <functional>

#include "comms/uniflow/core/Func.h"

namespace uniflow {

/// Pure virtual interface for a single-threaded event loop with cross-thread
/// dispatch.
///
/// All callbacks run on the loop thread. Other threads interact exclusively
/// via dispatch(). This single-consumer constraint enables lock-free MPSC
/// optimizations in concrete implementations.
class EventBase {
 public:
  EventBase() = default;
  virtual ~EventBase() = default;

  // Non-copyable, Non-movable
  EventBase(const EventBase&) = delete;
  EventBase& operator=(const EventBase&) = delete;
  EventBase(EventBase&&) = delete;
  EventBase& operator=(EventBase&&) = delete;

  // --- Core event loop ---

  /// Run the event loop until stop() is called.
  /// Must be called from exactly one thread — the loop thread.
  virtual void loop() = 0;

  /// Signal the event loop to exit. Thread-safe.
  ///
  /// After stop() returns, any subsequent dispatch() calls will enqueue
  /// functions that are never executed. dispatchAndWait() will block
  /// indefinitely. Callers must ensure all dispatch activity ceases
  /// before or concurrently with stop().
  virtual void stop() noexcept = 0;

  // --- Cross-thread dispatch ---

  /// Enqueue a function to run on the event loop thread. Thread-safe.
  /// Even if called from the loop thread, the function will be deferred.
  ///
  /// If called after stop(), the function is enqueued but never executed.
  virtual void dispatch(Func func) = 0;

  /// Enqueue a function to run on the event loop thread. Thread-safe.
  /// If called from the loop thread, the function runs immediately.
  virtual void dispatchInline(Func func) = 0;

  /// Enqueue a function and block until it completes. Thread-safe.
  /// If called from the loop thread, runs the function inline.
  ///
  /// Must not be called after stop() — will block indefinitely.
  virtual void dispatchAndWait(Func func) = 0;

  // --- I/O fd watching ---

  /// Callback for fd events. Receives the revents mask from epoll.
  ///
  /// std::function is used instead of Func because:
  ///   1. Func is one-shot — operator() destroys the callable. IOCallback
  ///      must fire repeatedly (e.g., async accept fires per client).
  ///   2. Func wraps void(). IOCallback needs void(uint32_t) to pass
  ///      revents so callbacks can distinguish EPOLLIN/EPOLLOUT/EPOLLERR.
  ///
  /// Callbacks must be noexcept — an uncaught exception will terminate
  /// the process. This is intentional: IO errors should be handled
  /// inside the callback, not propagated via exceptions.
  using IOCallback = std::function<void(uint32_t revents)>;

  /// Register a file descriptor for event watching. The callback fires
  /// on the loop thread whenever epoll reports matching events.
  /// If fd is already registered, replaces the previous registration.
  /// Thread-safe.
  virtual void registerFd(int fd, uint32_t events, IOCallback cb) = 0;

  /// Unregister a file descriptor. Thread-safe.
  /// The unregister is deferred to the loop thread. Callers must wait
  /// for it to complete (e.g., via dispatchAndWait) before closing the fd.
  virtual void unregisterFd(int fd) = 0;

  // --- Thread identity ---

  virtual bool inLoopThread() const noexcept = 0;
  virtual bool isLoopRunning() const noexcept = 0;
};

} // namespace uniflow
