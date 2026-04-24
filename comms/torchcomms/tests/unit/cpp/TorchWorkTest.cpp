// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

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
    runWaitHooks();
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

TEST(TorchWorkTest, WaitHookFiredOnWait) {
  auto work = c10::make_intrusive<TestWork>();
  int wait_count = 0;
  work->registerWorkWaitHook([&wait_count]() { wait_count++; });

  EXPECT_EQ(wait_count, 0);
  work->wait();
  EXPECT_EQ(wait_count, 1);

  // wait hooks fire every time wait() is called
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

TEST(TorchWorkTest, AllThreeHooksFiredInLifecycle) {
  auto work = c10::make_intrusive<TestWork>();
  std::vector<std::string> events;

  work->registerWorkStartHook([&events]() { events.push_back("start"); });
  work->registerWorkEndHook([&events]() { events.push_back("end"); });
  work->registerWorkWaitHook([&events]() { events.push_back("wait"); });

  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  work->wait();
  work->setStatus(TorchWork::WorkStatus::COMPLETED);

  std::vector<std::string> expected{"start", "wait", "end"};
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
