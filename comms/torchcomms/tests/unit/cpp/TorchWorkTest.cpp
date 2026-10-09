// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <atomic>
#include <functional>
#include <future>
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

void expectWaitHookRegistrationUsesNextSnapshot(
    const c10::intrusive_ptr<TestWork>& work,
    const std::function<void(TorchWork::WorkHook)>& registerHook) {
  std::promise<void> dispatchStartedPromise;
  auto dispatchStarted = dispatchStartedPromise.get_future();
  std::promise<void> releaseDispatchPromise;
  auto releaseDispatch = releaseDispatchPromise.get_future().share();
  std::atomic<int> existingCount{0};
  std::atomic<int> concurrentCount{0};
  std::atomic<int> reentrantCount{0};

  registerHook([&] {
    if (existingCount.fetch_add(1, std::memory_order_relaxed) != 0) {
      return;
    }
    // Entering the callback proves the reusable-hook snapshot is fixed.
    dispatchStartedPromise.set_value();
    releaseDispatch.wait();
    registerHook(
        [&] { reentrantCount.fetch_add(1, std::memory_order_relaxed); });
  });

  std::thread waiter([&] { work->wait(); });
  dispatchStarted.wait();
  registerHook(
      [&] { concurrentCount.fetch_add(1, std::memory_order_relaxed); });
  releaseDispatchPromise.set_value();
  waiter.join();

  EXPECT_EQ(existingCount.load(), 1);
  EXPECT_EQ(concurrentCount.load(), 0);
  EXPECT_EQ(reentrantCount.load(), 0);

  work->wait();

  EXPECT_EQ(existingCount.load(), 2);
  EXPECT_EQ(concurrentCount.load(), 1);
  EXPECT_EQ(reentrantCount.load(), 1);
}

// -- Lifecycle hook tests --

TEST(TorchWorkTest, TerminalStatusProducerCapabilityIsOptInAndSticky) {
  auto work = c10::make_intrusive<TestWork>();

  EXPECT_FALSE(work->hasTerminalStatusProducer());
  work->enableTerminalStatusProducer();
  EXPECT_TRUE(work->hasTerminalStatusProducer());
  work->enableTerminalStatusProducer();
  EXPECT_TRUE(work->hasTerminalStatusProducer());
}

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

TEST(TorchWorkTest, WaitPreHookRegistrationDuringDispatchUsesNextSnapshot) {
  auto work = c10::make_intrusive<TestWork>();
  expectWaitHookRegistrationUsesNextSnapshot(
      work, [&work](TorchWork::WorkHook hook) {
        work->registerWorkWaitPreHook(std::move(hook));
      });
}

TEST(TorchWorkTest, WaitPostHookRegistrationDuringDispatchUsesNextSnapshot) {
  auto work = c10::make_intrusive<TestWork>();
  expectWaitHookRegistrationUsesNextSnapshot(
      work, [&work](TorchWork::WorkHook hook) {
        work->registerWorkWaitPostHook(std::move(hook));
      });
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
}

// -- Thread-safety / state-machine tests --

TEST(TorchWorkTest, StartHookFiredImmediatelyAfterDirectTerminalTransition) {
  // Direct-to-terminal work must latch the start phase so a later start hook
  // fires immediately.
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::COMPLETED);

  int start_count = 0;
  work->registerWorkStartHook([&start_count]() { start_count++; });
  EXPECT_EQ(start_count, 1);
}

TEST(TorchWorkTest, TerminalStatusIsStickyOnSecondTransition) {
  // The first terminal transition wins, status included. End hooks fire on that
  // first transition, so letting a later one overwrite the status would leave
  // status() disagreeing with what the hook observed.
  auto work = c10::make_intrusive<TestWork>();
  TorchWork::WorkStatus seen_by_hook = TorchWork::WorkStatus::NOT_STARTED;
  work->registerWorkEndHook(
      [&seen_by_hook, &work]() { seen_by_hook = work->status(); });

  work->setStatus(TorchWork::WorkStatus::COMPLETED);
  work->setStatus(TorchWork::WorkStatus::TIMEDOUT);

  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(seen_by_hook, TorchWork::WorkStatus::COMPLETED);
}

TEST(TorchWorkTest, ConcurrentTerminalTransitionsAgreeWithTheFiredHook) {
  // Concurrent terminal transitions must publish the same winner to status()
  // and the end hook exactly once.
  for (int i = 0; i < 200; ++i) {
    auto work = c10::make_intrusive<TestWork>();
    std::atomic<int> end_count{0};
    TorchWork::WorkStatus seen_by_hook = TorchWork::WorkStatus::NOT_STARTED;
    work->registerWorkEndHook([&]() {
      seen_by_hook = work->status();
      end_count.fetch_add(1, std::memory_order_relaxed);
    });

    std::atomic<bool> go{false};
    std::thread watchdog([&] {
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      work->setStatus(TorchWork::WorkStatus::TIMEDOUT);
    });

    go.store(true, std::memory_order_release);
    work->setStatus(TorchWork::WorkStatus::COMPLETED);
    watchdog.join();

    EXPECT_EQ(end_count.load(), 1);
    EXPECT_EQ(seen_by_hook, work->status());
  }
}

TEST(TorchWorkTest, HookRegisteredDuringTerminalTransitionIsNotLost) {
  // Each concurrent registration must either join the transition batch or
  // fire immediately afterward, exactly once. Multiple registrars widen the
  // narrow interleaving window enough for dependable coverage.
  constexpr int kRegistrars = 8;
  for (int i = 0; i < 2000; ++i) {
    auto work = c10::make_intrusive<TestWork>();
    std::atomic<int> end_count{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> registrars;
    registrars.reserve(kRegistrars);

    for (int r = 0; r < kRegistrars; ++r) {
      registrars.emplace_back([&] {
        while (!go.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        work->registerWorkEndHook([&end_count]() {
          end_count.fetch_add(1, std::memory_order_relaxed);
        });
      });
    }

    go.store(true, std::memory_order_release);
    work->setStatus(TorchWork::WorkStatus::COMPLETED);
    for (auto& registrar : registrars) {
      registrar.join();
    }

    ASSERT_EQ(end_count.load(), kRegistrars) << "iteration " << i;
  }
}

TEST(TorchWorkTest, HookMayQueryTheWorkItIsAttachedTo) {
  // Callbacks must run outside the hook mutex so querying the work or
  // registering another hook cannot deadlock.
  auto work = c10::make_intrusive<TestWork>();
  int inner_count = 0;
  bool observed_terminal = false;

  work->registerWorkEndHook([&]() {
    observed_terminal = work->status() == TorchWork::WorkStatus::COMPLETED;
    work->registerWorkEndHook([&inner_count]() { inner_count++; });
  });

  work->setStatus(TorchWork::WorkStatus::COMPLETED);

  EXPECT_TRUE(observed_terminal);
  EXPECT_EQ(inner_count, 1);
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
