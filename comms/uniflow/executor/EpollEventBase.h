// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <system_error>
#include <thread>
#include <unordered_map>

#include "comms/uniflow/core/Func.h"
#include "comms/uniflow/executor/EventBase.h"
#include "comms/uniflow/logging/Logger.h"

namespace uniflow {

/// Epoll-based EventBase parameterized by a queue policy.
///
/// The QueuePolicy controls how cross-thread dispatch works — it's the
/// only dimension that varies between EventBase implementations. Everything
/// else (epoll loop, fd registration, wakeup, dispatch helpers, thread
/// identity) is shared.
///
/// Policy-based design is used here instead of inheritance because:
///   1. No friend class needed — the policy is a member, called directly.
///   2. No template tricks — loop() calls queue_.drain() naturally.
///   3. No boilerplate — adding a new queue strategy = new policy struct.
///   4. 2-level hierarchy (EventBase → EpollEventBase<P>) instead of 3.
///   5. Same performance — static dispatch via template, drain() inlineable.
///
/// QueuePolicy must implement:
///   void push(Func func);     — thread-safe enqueue
///   bool drain() noexcept;    — drain all items, return true if any drained
///                                (called only from loop thread)
///
/// The event loop uses epoll for I/O multiplexing and an eventfd
/// to wake up when functions are enqueued from other threads.
template <typename QueuePolicy>
class EpollEventBase : public EventBase {
 public:
  EpollEventBase()
      : wakeupFd_(eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)),
        epollFd_(epoll_create1(EPOLL_CLOEXEC)) {
    if (wakeupFd_ < 0) {
      throw std::system_error(
          errno, std::system_category(), "eventfd creation failed");
    }
    if (epollFd_ < 0) {
      close(wakeupFd_);
      throw std::system_error(
          errno, std::system_category(), "epoll_create1 failed");
    }
    struct epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = wakeupFd_;
    if (epoll_ctl(epollFd_, EPOLL_CTL_ADD, wakeupFd_, &ev) < 0) {
      int err = errno;
      close(epollFd_);
      close(wakeupFd_);
      throw std::system_error(
          err, std::system_category(), "epoll_ctl wakeupFd failed");
    }
  }

  /// Caller must call stop() and join the loop thread before destroying.
  /// Asserts in debug builds if the loop is still running.
  ~EpollEventBase() override {
    assert(stop_.load(std::memory_order_acquire));
    if (epollFd_ >= 0) {
      close(epollFd_);
    }
    if (wakeupFd_ >= 0) {
      close(wakeupFd_);
    }
  }

  // Non-copyable, Non-movable
  EpollEventBase(const EpollEventBase&) = delete;
  EpollEventBase& operator=(const EpollEventBase&) = delete;
  EpollEventBase(EpollEventBase&&) = delete;
  EpollEventBase& operator=(EpollEventBase&&) = delete;

  /// drain() is called via static dispatch — non-virtual and inlineable,
  /// which matters because it's the hot path on every loop iteration.
  void loop() override {
    assert(stop_.load(std::memory_order_acquire));
    loopThreadId_ = std::this_thread::get_id();
    stop_.store(false, std::memory_order_release);

    while (!stop_.load(std::memory_order_acquire)) {
      wakeupPending_.store(false, std::memory_order_release);
      queue_.drain();
      waitForEvents();
    }
    while (queue_.drain()) {
    }
  }

  void dispatch(Func func) override {
    queue_.push(std::move(func));
    wakeup();
  }

  void stop() noexcept override {
    stop_.store(true, std::memory_order_release);
    wakeup();
  }

  void dispatchInline(Func func) override {
    if (inLoopThread()) {
      func();
      return;
    }
    dispatch(std::move(func));
  }

  void dispatchAndWait(Func func) override {
    if (inLoopThread()) {
      func();
      return;
    }
    std::mutex m;
    std::condition_variable cv;
    bool done = false;

    dispatch([&func, &m, &cv, &done]() noexcept {
      func();
      {
        std::lock_guard<std::mutex> lock(m);
        done = true;
      }
      cv.notify_one();
    });

    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&done] { return done; });
  }

  bool inLoopThread() const noexcept override {
    return std::this_thread::get_id() == loopThreadId_;
  }

  bool isLoopRunning() const noexcept override {
    return !stop_.load(std::memory_order_acquire);
  }

  void registerFd(int fd, uint32_t events, IOCallback cb) override {
    dispatch([this, fd, events, cb = std::move(cb)]() mutable noexcept {
      struct epoll_event ev{};
      ev.events = events;
      ev.data.fd = fd;
      auto it = ioEntries_.find(fd);
      if (it != ioEntries_.end()) {
        it->second.cb = std::move(cb);
        int ret = epoll_ctl(epollFd_, EPOLL_CTL_MOD, fd, &ev);
        if (ret != 0) {
          UNIFLOW_LOG_ERROR(
              "epoll_ctl MOD failed: fd={} errno={} ({})",
              fd,
              errno,
              std::system_category().message(errno));
        }
        assert(ret == 0);
      } else {
        ioEntries_.emplace(fd, IOEntry{std::move(cb)});
        int ret = epoll_ctl(epollFd_, EPOLL_CTL_ADD, fd, &ev);
        if (ret != 0) {
          UNIFLOW_LOG_ERROR(
              "epoll_ctl ADD failed: fd={} errno={} ({})",
              fd,
              errno,
              std::system_category().message(errno));
        }
        assert(ret == 0);
      }
    });
  }

  void unregisterFd(int fd) override {
    dispatch([this, fd]() noexcept {
      if (ioEntries_.erase(fd) > 0) {
        int ret = epoll_ctl(epollFd_, EPOLL_CTL_DEL, fd, nullptr);
        if (ret != 0) {
          UNIFLOW_LOG_ERROR(
              "epoll_ctl DEL failed: fd={} errno={} ({})",
              fd,
              errno,
              std::system_category().message(errno));
        }
        assert(ret == 0);
      }
    });
  }

 private:
  struct IOEntry {
    IOCallback cb;
  };

  /// Block until events arrive (epoll_wait with infinite timeout), then
  /// dispatch IO callbacks and drain the wakeup eventfd.
  /// Returns immediately on EINTR so the caller can re-check stop.
  __attribute__((noinline)) void waitForEvents() {
    constexpr int kMaxEvents = 64;
    struct epoll_event events[kMaxEvents];

    int n = epoll_wait(epollFd_, events, kMaxEvents, -1);
    if (n < 0) {
      if (errno == EINTR) {
        return;
      }
      throw std::system_error(
          errno, std::system_category(), "epoll_wait failed");
    }

    for (int i = 0; i < n; ++i) {
      int fd = events[i].data.fd;
      if (fd == wakeupFd_) {
        uint64_t val = 0;
        while (read(wakeupFd_, &val, sizeof(val)) > 0) {
        }
      } else {
        auto it = ioEntries_.find(fd);
        if (it != ioEntries_.end()) {
          // IO callbacks are invoked inline on the loop thread. Callbacks
          // that call registerFd/unregisterFd go through dispatch(), so
          // ioEntries_ is not modified during this iteration.
          [&]() noexcept { it->second.cb(events[i].events); }();
        } else {
          uint32_t revents = events[i].events;
          UNIFLOW_LOG_WARN(
              "epoll event for unknown fd={}, revents={:#x}", fd, revents);
        }
      }
    }
  }

  /// wakeupPending_ coalesces redundant writes.
  void wakeup() noexcept {
    if (!wakeupPending_.exchange(true, std::memory_order_acq_rel)) {
      uint64_t val = 1;
      [[maybe_unused]] auto ret = write(wakeupFd_, &val, sizeof(val));
    }
  }

  QueuePolicy queue_;
  const int wakeupFd_{-1};
  const int epollFd_{-1};
  std::atomic<bool> wakeupPending_{false};
  std::atomic<bool> stop_{true};
  std::thread::id loopThreadId_;
  std::unordered_map<int, IOEntry> ioEntries_;
};

} // namespace uniflow
