// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "comms/uniflow/executor/LockFreeEventBase.h"
#include "comms/uniflow/executor/MutexEventBase.h"

namespace uniflow {

template <typename T>
class EventBaseTypedTest : public testing::Test {
 protected:
  void SetUp() override {
    loopThread_ = std::thread([this] { evb_.loop(); });
    evb_.dispatchAndWait([]() noexcept {});
    ASSERT_TRUE(evb_.isLoopRunning());
  }

  void TearDown() override {
    evb_.stop();
    loopThread_.join();
  }

  T evb_;
  std::thread loopThread_;
};

using EventBaseTypes = ::testing::Types<MutexEventBase, LockFreeEventBase>;
TYPED_TEST_SUITE(EventBaseTypedTest, EventBaseTypes);

TYPED_TEST(EventBaseTypedTest, Dispatch) {
  std::promise<std::thread::id> p;
  auto f = p.get_future();
  this->evb_.dispatch(
      [&p]() noexcept { p.set_value(std::this_thread::get_id()); });

  auto executedOn = f.get();
  EXPECT_NE(executedOn, std::this_thread::get_id());
  EXPECT_EQ(executedOn, this->loopThread_.get_id());
}

TYPED_TEST(EventBaseTypedTest, DispatchInlineFromLoopThread) {
  std::promise<bool> p;
  auto f = p.get_future();
  this->evb_.dispatch([this, &p]() noexcept {
    bool ranInline = false;
    this->evb_.dispatchInline([&ranInline]() noexcept { ranInline = true; });
    p.set_value(ranInline);
  });

  EXPECT_TRUE(f.get());
}

TYPED_TEST(EventBaseTypedTest, DispatchInlineFromNonLoopThread) {
  std::promise<std::thread::id> p;
  auto f = p.get_future();
  this->evb_.dispatchInline(
      [&p]() noexcept { p.set_value(std::this_thread::get_id()); });

  auto executedOn = f.get();
  EXPECT_NE(executedOn, std::this_thread::get_id());
  EXPECT_EQ(executedOn, this->loopThread_.get_id());
}

TYPED_TEST(EventBaseTypedTest, InLoopThread) {
  EXPECT_FALSE(this->evb_.inLoopThread());

  std::promise<bool> p;
  auto f = p.get_future();
  this->evb_.dispatch(
      [this, &p]() noexcept { p.set_value(this->evb_.inLoopThread()); });
  EXPECT_TRUE(f.get());
}

TYPED_TEST(EventBaseTypedTest, DispatchAndWait) {
  int result = 0;
  this->evb_.dispatchAndWait([&result]() noexcept { result = 42; });
  EXPECT_EQ(result, 42);
}

TYPED_TEST(EventBaseTypedTest, DispatchAndWaitFromLoopThread) {
  std::promise<int> p;
  auto f = p.get_future();
  this->evb_.dispatch([this, &p]() noexcept {
    int result = 0;
    this->evb_.dispatchAndWait([&result]() noexcept { result = 42; });
    p.set_value(result);
  });

  EXPECT_EQ(f.get(), 42);
}

TYPED_TEST(EventBaseTypedTest, ConcurrentDispatch) {
  constexpr int kThreads = 32;
  constexpr int kPerThread = 10000;
  std::atomic<int> counter{0};

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([this, &counter] {
      for (int j = 0; j < kPerThread; ++j) {
        this->evb_.dispatch([&counter]() noexcept {
          counter.fetch_add(1, std::memory_order_relaxed);
        });
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  this->evb_.dispatchAndWait([]() noexcept {});
  EXPECT_EQ(counter.load(), kThreads * kPerThread);
}

TYPED_TEST(EventBaseTypedTest, StressDispatchAndDispatchInline) {
  constexpr int kThreads = 32;
  constexpr int kPerThread = 10000;
  std::atomic<int> counter{0};

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([this, &counter] {
      for (int j = 0; j < kPerThread; ++j) {
        if (j % 2 == 0) {
          this->evb_.dispatch([&counter]() noexcept {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
        } else {
          this->evb_.dispatchInline([&counter]() noexcept {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
        }
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  this->evb_.dispatchAndWait([]() noexcept {});
  EXPECT_EQ(counter.load(), kThreads * kPerThread);
}

TYPED_TEST(EventBaseTypedTest, StressDispatchAndWait) {
  constexpr int kThreads = 32;
  constexpr int kPerThread = 1000;
  std::atomic<int> counter{0};

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([this, &counter] {
      for (int j = 0; j < kPerThread; ++j) {
        this->evb_.dispatchAndWait([&counter]() noexcept {
          counter.fetch_add(1, std::memory_order_relaxed);
        });
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(counter.load(), kThreads * kPerThread);
}

TYPED_TEST(EventBaseTypedTest, MoveOnlyFuncDispatch) {
  auto ptr = std::make_unique<int>(42);
  std::promise<int> p;
  auto f = p.get_future();

  this->evb_.dispatch(
      [val = std::move(ptr), &p]() noexcept { p.set_value(*val); });

  EXPECT_EQ(f.get(), 42);
}

// ---------------------------------------------------------------------------
// fd-watching tests
// ---------------------------------------------------------------------------

TYPED_TEST(EventBaseTypedTest, RegisterFdPollin) {
  int efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  ASSERT_GE(efd, 0);

  std::promise<uint32_t> p;
  auto f = p.get_future();

  // Drain the eventfd in the callback so it becomes non-readable and
  // the persistent callback doesn't fire again.
  this->evb_.registerFd(efd, EPOLLIN, [&p, efd](uint32_t revents) {
    uint64_t drain = 0;
    while (read(efd, &drain, sizeof(drain)) > 0) {
    }
    p.set_value(revents);
  });

  uint64_t val = 1;
  ASSERT_EQ(write(efd, &val, sizeof(val)), static_cast<ssize_t>(sizeof(val)));

  auto revents = f.get();
  EXPECT_NE(revents & EPOLLIN, 0);

  this->evb_.unregisterFd(efd);
  this->evb_.dispatchAndWait([]() noexcept {});
  close(efd);
}

TYPED_TEST(EventBaseTypedTest, UnregisterFdPreventsCallback) {
  int efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  ASSERT_GE(efd, 0);

  std::atomic<bool> called{false};

  this->evb_.registerFd(efd, EPOLLIN, [&called](uint32_t) {
    called.store(true, std::memory_order_release);
  });

  this->evb_.unregisterFd(efd);

  uint64_t val = 1;
  ASSERT_EQ(write(efd, &val, sizeof(val)), static_cast<ssize_t>(sizeof(val)));

  this->evb_.dispatchAndWait([]() noexcept {});
  EXPECT_FALSE(called.load(std::memory_order_acquire));

  close(efd);
}

TYPED_TEST(EventBaseTypedTest, MultipleFdsRegistered) {
  int efd1 = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  int efd2 = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  ASSERT_GE(efd1, 0);
  ASSERT_GE(efd2, 0);

  std::promise<void> p1;
  auto f1 = p1.get_future();
  std::promise<void> p2;
  auto f2 = p2.get_future();

  this->evb_.registerFd(efd1, EPOLLIN, [&p1, efd1](uint32_t) {
    uint64_t drain = 0;
    while (read(efd1, &drain, sizeof(drain)) > 0) {
    }
    p1.set_value();
  });
  this->evb_.registerFd(efd2, EPOLLIN, [&p2, efd2](uint32_t) {
    uint64_t drain = 0;
    while (read(efd2, &drain, sizeof(drain)) > 0) {
    }
    p2.set_value();
  });

  this->evb_.dispatchAndWait([]() noexcept {});

  uint64_t val = 1;
  ASSERT_EQ(write(efd1, &val, sizeof(val)), static_cast<ssize_t>(sizeof(val)));
  f1.get();

  ASSERT_EQ(write(efd2, &val, sizeof(val)), static_cast<ssize_t>(sizeof(val)));
  f2.get();

  this->evb_.unregisterFd(efd1);
  this->evb_.unregisterFd(efd2);
  this->evb_.dispatchAndWait([]() noexcept {});
  close(efd1);
  close(efd2);
}

TYPED_TEST(EventBaseTypedTest, RegisterFdPollout) {
  // socketpair: connected sockets where writing is immediately possible.
  // POLLOUT fires continuously, so guard the promise with an atomic.
  int fds[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0, fds), 0);

  std::promise<uint32_t> p;
  auto f = p.get_future();
  std::atomic<bool> done{false};

  this->evb_.registerFd(fds[0], EPOLLOUT, [&p, &done](uint32_t revents) {
    if (!done.exchange(true, std::memory_order_acq_rel)) {
      p.set_value(revents);
    }
  });

  auto revents = f.get();
  EXPECT_NE(revents & EPOLLOUT, 0);

  this->evb_.unregisterFd(fds[0]);
  this->evb_.dispatchAndWait([]() noexcept {});
  close(fds[0]);
  close(fds[1]);
}

TYPED_TEST(EventBaseTypedTest, ReRegisterFdReplacesCallback) {
  int efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  ASSERT_GE(efd, 0);

  std::atomic<int> firstCalled{0};

  this->evb_.registerFd(efd, EPOLLIN, [&firstCalled](uint32_t) {
    firstCalled.fetch_add(1, std::memory_order_relaxed);
  });

  std::promise<void> p;
  auto f = p.get_future();
  this->evb_.registerFd(efd, EPOLLIN, [efd, &p](uint32_t) {
    uint64_t drain = 0;
    while (read(efd, &drain, sizeof(drain)) > 0) {
    }
    p.set_value();
  });

  this->evb_.dispatchAndWait([]() noexcept {});

  uint64_t val = 1;
  ASSERT_EQ(write(efd, &val, sizeof(val)), static_cast<ssize_t>(sizeof(val)));

  f.get();
  EXPECT_EQ(firstCalled.load(), 0);

  this->evb_.unregisterFd(efd);
  this->evb_.dispatchAndWait([]() noexcept {});
  close(efd);
}

TYPED_TEST(EventBaseTypedTest, IOCallbackFiresRepeatedly) {
  int efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  ASSERT_GE(efd, 0);

  std::atomic<int> callCount{0};
  std::promise<void> done;
  auto f = done.get_future();

  constexpr int kExpected = 3;
  this->evb_.registerFd(efd, EPOLLIN, [&](uint32_t) {
    uint64_t drain = 0;
    while (read(efd, &drain, sizeof(drain)) > 0) {
    }
    if (callCount.fetch_add(1, std::memory_order_acq_rel) + 1 >= kExpected) {
      done.set_value();
    }
  });

  this->evb_.dispatchAndWait([]() noexcept {});

  // Write one at a time, waiting for the previous callback to complete
  // before writing again (prevents eventfd counter accumulation).
  for (int i = 0; i < kExpected; ++i) {
    while (callCount.load(std::memory_order_acquire) < i) {
      std::this_thread::yield();
    }
    uint64_t val = 1;
    ASSERT_EQ(write(efd, &val, sizeof(val)), static_cast<ssize_t>(sizeof(val)));
  }

  f.get();
  EXPECT_GE(callCount.load(), kExpected);

  this->evb_.unregisterFd(efd);
  this->evb_.dispatchAndWait([]() noexcept {});
  close(efd);
}

TYPED_TEST(EventBaseTypedTest, ContinuousWriteAccumulation) {
  // Caller thread continuously writes val=1 to an eventfd without waiting.
  // The loop thread reads and accumulates values. The total must match
  // the number of writes — verifying no events are lost under pressure.
  int efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  ASSERT_GE(efd, 0);

  constexpr int kWrites = 10000;
  uint64_t totalRead = 0;
  std::promise<void> done;
  auto f = done.get_future();

  this->evb_.registerFd(efd, EPOLLIN, [&](uint32_t) {
    uint64_t val = 0;
    assert(totalRead < kWrites);
    while (read(efd, &val, sizeof(val)) > 0) {
      totalRead += val;
      val = 0;
      if (totalRead == kWrites) {
        done.set_value();
      } else if (totalRead > kWrites) {
        FAIL() << "totalRead=" << totalRead << " > kWrites=" << kWrites;
      }
    }
  });

  // Ensure registration is complete before writing.
  this->evb_.dispatchAndWait([]() noexcept {});

  // Continuously write val=1 without waiting between writes.
  for (int i = 0; i < kWrites; ++i) {
    uint64_t val = 1;
    ASSERT_EQ(write(efd, &val, sizeof(val)), static_cast<ssize_t>(sizeof(val)));
  }

  f.get();
  EXPECT_EQ(totalRead, kWrites);

  this->evb_.unregisterFd(efd);
  this->evb_.dispatchAndWait([]() noexcept {});
  close(efd);
}

TYPED_TEST(EventBaseTypedTest, RegisterFdEpollhup) {
  // Create a socketpair. Close one end to trigger EPOLLHUP on the other.
  int fds[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0, fds), 0);

  std::promise<uint32_t> p;
  auto f = p.get_future();
  std::atomic<bool> done{false};

  this->evb_.registerFd(fds[0], EPOLLIN, [&p, &done](uint32_t revents) {
    if (!done.exchange(true, std::memory_order_acq_rel)) {
      p.set_value(revents);
    }
  });

  this->evb_.dispatchAndWait([]() noexcept {});

  // Close the peer end — triggers EPOLLHUP on fds[0]
  close(fds[1]);

  auto revents = f.get();
  EXPECT_NE(revents & EPOLLHUP, 0u);

  this->evb_.unregisterFd(fds[0]);
  this->evb_.dispatchAndWait([]() noexcept {});
  close(fds[0]);
}

TYPED_TEST(EventBaseTypedTest, UnregisterClosedFdDoesNotCrash) {
  int efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  ASSERT_GE(efd, 0);

  this->evb_.registerFd(efd, EPOLLIN, [](uint32_t) {});
  this->evb_.dispatchAndWait([]() noexcept {});

  // Close the fd first, then unregister — should log a warning, not crash.
  close(efd);
  this->evb_.unregisterFd(efd);
  this->evb_.dispatchAndWait([]() noexcept {});
}

TYPED_TEST(EventBaseTypedTest, RegisterFdFromLoopThreadIsImmediate) {
  int efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  ASSERT_GE(efd, 0);

  std::promise<void> p;
  auto f = p.get_future();

  // Register from inside a dispatch lambda (on the loop thread).
  // With the inLoopThread optimization, registerFd executes inline —
  // the callback should fire on the very next epoll iteration.
  this->evb_.dispatch([&]() noexcept {
    this->evb_.registerFd(efd, EPOLLIN, [&p, efd](uint32_t) {
      uint64_t drain = 0;
      ::read(efd, &drain, sizeof(drain));
      p.set_value();
    });
    // Write from the loop thread — the callback should fire immediately
    // on the next epoll iteration (no extra dispatch round-trip).
    uint64_t val = 1;
    ::write(efd, &val, sizeof(val));
  });

  f.get();

  this->evb_.unregisterFd(efd);
  this->evb_.dispatchAndWait([]() noexcept {});
  close(efd);
}

} // namespace uniflow
