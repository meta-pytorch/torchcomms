// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>
#include <thread>

#include <c10/util/intrusive_ptr.h>
#include "comms/torchcomms/TorchWork.hpp"

namespace torch::comms::test {

class TestWork : public TorchWork {
 public:
  explicit TestWork(bool* destroyed = nullptr) : destroyed_(destroyed) {}
  ~TestWork() override {
    if (destroyed_) {
      *destroyed_ = true;
    }
  }

  void wait() override {
    runWaitPreHooks();
    runWaitPostHooks();
  }

  // expose for testing
  using TorchWork::setStatus;

 private:
  bool* destroyed_;
};

// -- Lifecycle hook tests --

TEST(TorchWorkTest, StartHookFiredOnInProgress) {
  auto work = c10::make_intrusive<TestWork>();
  int start_count = 0;
  work->registerWorkStartHook([&start_count]() { start_count++; });

  EXPECT_EQ(start_count, 0);
  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  EXPECT_EQ(start_count, 1);
}

TEST(TorchWorkTest, StartHookFiredImmediatelyIfAlreadyInProgress) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::INPROGRESS);

  // A start hook registered after the work already transitioned to INPROGRESS
  // must fire immediately (mirrors the end-hook fallback). This is the MCCL
  // case: TorchWorkMCCL's ctor sets INPROGRESS before the clog post-hook
  // registers the start hook, which otherwise drops the "S" clog event.
  int start_count = 0;
  work->registerWorkStartHook([&start_count]() { start_count++; });
  EXPECT_EQ(start_count, 1);
}

TEST(TorchWorkTest, EndHookFiredOnCompleted) {
  auto work = c10::make_intrusive<TestWork>();
  int end_count = 0;
  work->registerWorkEndHook([&end_count]() { end_count++; });

  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  EXPECT_EQ(end_count, 0);

  work->setStatus(TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(end_count, 1);
}

TEST(TorchWorkTest, EndHookFiredOnError) {
  auto work = c10::make_intrusive<TestWork>();
  int end_count = 0;
  work->registerWorkEndHook([&end_count]() { end_count++; });

  work->setStatus(TorchWork::WorkStatus::ERROR);
  EXPECT_EQ(end_count, 1);
}

TEST(TorchWorkTest, EndHookFiredOnTimedOut) {
  auto work = c10::make_intrusive<TestWork>();
  int end_count = 0;
  work->registerWorkEndHook([&end_count]() { end_count++; });

  work->setStatus(TorchWork::WorkStatus::TIMEDOUT);
  EXPECT_EQ(end_count, 1);
}

TEST(TorchWorkTest, WaitPreHookFiredOnWait) {
  auto work = c10::make_intrusive<TestWork>();
  int wait_count = 0;
  work->registerWorkWaitPreHook([&wait_count]() { wait_count++; });

  EXPECT_EQ(wait_count, 0);
  work->wait();
  EXPECT_EQ(wait_count, 1);

  // wait hooks fire every time wait() is called
  work->wait();
  EXPECT_EQ(wait_count, 2);
}

TEST(TorchWorkTest, WaitPostHookFiredOnWait) {
  auto work = c10::make_intrusive<TestWork>();
  int wait_count = 0;
  work->registerWorkWaitPostHook([&wait_count]() { wait_count++; });

  EXPECT_EQ(wait_count, 0);
  work->wait();
  EXPECT_EQ(wait_count, 1);

  work->wait();
  EXPECT_EQ(wait_count, 2);
}

TEST(TorchWorkTest, MultipleHooksFireInOrder) {
  auto work = c10::make_intrusive<TestWork>();
  std::vector<int> order;

  work->registerWorkStartHook([&order]() { order.push_back(1); });
  work->registerWorkStartHook([&order]() { order.push_back(2); });
  work->registerWorkStartHook([&order]() { order.push_back(3); });

  work->setStatus(TorchWork::WorkStatus::INPROGRESS);

  std::vector<int> expected{1, 2, 3};
  EXPECT_EQ(order, expected);
}

TEST(TorchWorkTest, MultipleHookFailuresRunAllHooksAndRethrowFirst) {
  auto work = c10::make_intrusive<TestWork>();
  std::vector<int> order;

  work->registerWorkStartHook([&order]() {
    order.push_back(1);
    throw std::logic_error("first failure");
  });
  work->registerWorkStartHook([&order]() {
    order.push_back(2);
    throw std::runtime_error("second failure");
  });
  work->registerWorkStartHook([&order]() { order.push_back(3); });

  EXPECT_THROW(
      work->setStatus(TorchWork::WorkStatus::INPROGRESS), std::logic_error);

  const std::vector<int> expected{1, 2, 3};
  EXPECT_EQ(order, expected);
}

TEST(TorchWorkTest, StartHookNotFiredOnTerminalStatus) {
  auto work = c10::make_intrusive<TestWork>();
  int start_count = 0;
  work->registerWorkStartHook([&start_count]() { start_count++; });

  work->setStatus(TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(start_count, 0);
}

TEST(TorchWorkTest, EndHookNotFiredOnInProgress) {
  auto work = c10::make_intrusive<TestWork>();
  int end_count = 0;
  work->registerWorkEndHook([&end_count]() { end_count++; });

  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  EXPECT_EQ(end_count, 0);
}

TEST(TorchWorkTest, AllHooksFiredInLifecycle) {
  auto work = c10::make_intrusive<TestWork>();
  std::vector<std::string> events;

  work->registerWorkStartHook([&events]() { events.emplace_back("start"); });
  work->registerWorkEndHook([&events]() { events.emplace_back("end"); });
  work->registerWorkWaitPreHook(
      [&events]() { events.emplace_back("wait_pre"); });
  work->registerWorkWaitPostHook(
      [&events]() { events.emplace_back("wait_post"); });

  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  work->wait();
  work->setStatus(TorchWork::WorkStatus::COMPLETED);

  std::vector<std::string> expected{"start", "wait_pre", "wait_post", "end"};
  EXPECT_EQ(events, expected);
}

TEST(TorchWorkTest, EndHookFiredImmediatelyIfAlreadyTerminal) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::COMPLETED);

  int end_count = 0;
  work->registerWorkEndHook([&end_count]() { end_count++; });
  EXPECT_EQ(end_count, 1);
}

TEST(TorchWorkTest, EndHooksFiredAtMostOnce) {
  auto work = c10::make_intrusive<TestWork>();
  int end_count = 0;
  work->registerWorkEndHook([&end_count]() { end_count++; });

  work->setStatus(TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(end_count, 1);

  // Second terminal status should not fire end hooks again
  work->setStatus(TorchWork::WorkStatus::ERROR);
  EXPECT_EQ(end_count, 1);
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST(TorchWorkTest, ConcurrentTerminalTransitionsLatchFirstResult) {
  constexpr int kIterations = 1000;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    auto work = c10::make_intrusive<TestWork>();
    work->setStatus(TorchWork::WorkStatus::INPROGRESS);
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<int> endCount{0};
    work->registerWorkEndHook([&]() { endCount.fetch_add(1); });
    auto terminalize = [&](TorchWork::WorkStatus status) {
      ready.fetch_add(1);
      while (!go.load()) {
        std::this_thread::yield();
      }
      work->setStatus(status);
    };
    std::thread completeThread(terminalize, TorchWork::WorkStatus::COMPLETED);
    std::thread timeoutThread(terminalize, TorchWork::WorkStatus::TIMEDOUT);
    while (ready.load() != 2) {
      std::this_thread::yield();
    }
    go.store(true);
    completeThread.join();
    timeoutThread.join();

    const auto latched = work->status();
    EXPECT_TRUE(
        latched == TorchWork::WorkStatus::COMPLETED ||
        latched == TorchWork::WorkStatus::TIMEDOUT);
    EXPECT_EQ(endCount.load(), 1);
    work->setStatus(TorchWork::WorkStatus::ERROR);
    EXPECT_EQ(work->status(), latched);
  }
}

TEST(TorchWorkTest, EndHooksContinueAfterFailureAndRethrowFirstError) {
  auto work = c10::make_intrusive<TestWork>();
  std::vector<int> order;
  work->registerWorkEndHook([&order]() {
    order.push_back(1);
    throw std::runtime_error("first hook failed");
  });
  work->registerWorkEndHook([&order]() {
    order.push_back(2);
    throw std::logic_error("second hook failed");
  });
  work->registerWorkEndHook([&order]() { order.push_back(3); });

  try {
    work->setStatus(TorchWork::WorkStatus::COMPLETED);
    FAIL() << "Expected the first hook error";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "first hook failed");
  }

  const std::vector<int> expected{1, 2, 3};
  EXPECT_EQ(order, expected);
}

TEST(TorchWorkTest, StartHooksContinueAfterFailureAndRethrowFirstErrorOnce) {
  auto work = c10::make_intrusive<TestWork>();
  std::vector<int> order;
  work->registerWorkStartHook([&order]() {
    order.push_back(1);
    throw std::runtime_error("first hook failed");
  });
  work->registerWorkStartHook([&order]() {
    order.push_back(2);
    throw std::logic_error("second hook failed");
  });
  work->registerWorkStartHook([&order]() { order.push_back(3); });

  try {
    work->setStatus(TorchWork::WorkStatus::INPROGRESS);
    FAIL() << "Expected the first hook error";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "first hook failed");
  }

  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  const std::vector<int> expected{1, 2, 3};
  EXPECT_EQ(order, expected);
}

TEST(TorchWorkTest, ConcurrentEndHookRegistrationDoesNotLoseCompletion) {
  constexpr int kIterations = 1000;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    auto work = c10::make_intrusive<TestWork>();
    std::atomic<int> endCount{0};
    std::thread registerThread(
        [&]() { work->registerWorkEndHook([&]() { endCount.fetch_add(1); }); });
    std::thread completeThread(
        [&]() { work->setStatus(TorchWork::WorkStatus::COMPLETED); });

    registerThread.join();
    completeThread.join();
    EXPECT_EQ(endCount.load(), 1);
  }
}

// -- Release resources / weak-ref cycle tests --

TEST(TorchWorkTest, WorkDestroyedAfterEndHookWithWeakRef) {
  bool destroyed = false;
  {
    auto work = c10::make_intrusive<TestWork>(&destroyed);
    c10::weak_intrusive_ptr<TestWork> weak_work(work);
    work->registerWorkEndHook(
        [weak_work = std::move(weak_work)]() { (void)weak_work; });
    EXPECT_FALSE(destroyed);
  }
  EXPECT_TRUE(destroyed);
}

TEST(TorchWorkTest, WorkDestroyedWithoutHooks) {
  bool destroyed = false;
  {
    auto work = c10::make_intrusive<TestWork>(&destroyed);
    EXPECT_FALSE(destroyed);
  }
  EXPECT_TRUE(destroyed);
}

} // namespace torch::comms::test
