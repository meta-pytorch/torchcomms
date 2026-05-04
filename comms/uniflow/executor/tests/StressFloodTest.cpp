// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <atomic>
#include <future>
#include <thread>

#include <gtest/gtest.h>

#include "comms/uniflow/executor/LockFreeEventBase.h"
#include "comms/uniflow/executor/MutexEventBase.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"

#ifdef ENABLE_FOLLY
#include <folly/io/async/EventBase.h>
#include <folly/io/async/ScopedEventBaseThread.h>
#endif // ENABLE_FOLLY

namespace uniflow {

/// Adapter that maps a uniform dispatch API to a concrete EventBase type.
/// The default works for uniflow::EventBase. Specialize for other event
/// base implementations (e.g. folly::EventBase).
template <typename EventBaseT>
struct EventBaseAdapter {
  template <typename F>
  static void dispatch(EventBaseT& evb, F&& func) {
    evb.dispatch(std::forward<F>(func));
  }

  template <typename F>
  static void dispatchAndWait(EventBaseT& evb, F&& func) {
    evb.dispatchAndWait(std::forward<F>(func));
  }
};

// --- EventBase flood test (manual loop thread) ---

template <typename T>
class EventBaseFloodTest : public testing::Test {
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
TYPED_TEST_SUITE(EventBaseFloodTest, EventBaseTypes);

// --- ScopedEventBaseThread flood test ---

template <typename T>
class ScopedEventBaseThreadFloodTest : public ::testing::Test {};

TYPED_TEST_SUITE(ScopedEventBaseThreadFloodTest, EventBaseTypes);

// --- Deadline-based progress test ---
//
// Verifies that dispatchAndWait makes forward progress under producer flood
// within a bounded time. So that starvation is detected as a clean test failure
// instead of a hang.
template <typename EventBaseT>
void testDispatchAndWaitProgress(
    EventBaseT& evb,
    int numProducers = 4,
    int numIterations = 100,
    std::chrono::seconds deadline = std::chrono::seconds(30)) {
  using Adapter = EventBaseAdapter<EventBaseT>;

  std::atomic<bool> go{false};
  std::atomic<bool> done{false};

  std::vector<std::thread> producers;
  producers.reserve(numProducers);
  for (int i = 0; i < numProducers; ++i) {
    producers.emplace_back([&evb, &go, &done] {
      while (!go.load(std::memory_order_acquire)) {
      }
      while (!done.load(std::memory_order_acquire)) {
        Adapter::dispatch(evb, []() noexcept {});
      }
    });
  }

  go.store(true, std::memory_order_release);

  // Run dispatchAndWait in a separate thread so we can enforce a deadline.
  std::promise<void> promise;
  auto future = promise.get_future();
  std::thread worker([&evb, &promise, numIterations] {
    for (int i = 0; i < numIterations; ++i) {
      Adapter::dispatchAndWait(evb, []() noexcept {});
    }
    promise.set_value();
  });

  bool completed = (future.wait_for(deadline) == std::future_status::ready);

  // Stop producers. Once the flood stops, any in-progress dispatchAndWait
  // will complete (the queue drains), so worker.join() won't hang.
  done.store(true, std::memory_order_release);
  for (auto& p : producers) {
    p.join();
  }
  worker.join();

  EXPECT_TRUE(completed) << "dispatchAndWait starved: could not complete "
                         << numIterations << " iterations within "
                         << deadline.count() << "s under " << numProducers
                         << " producer flood";
}

// Uniflow typed tests — should pass even under --stress-runs.

TYPED_TEST(EventBaseFloodTest, DispatchAndWaitProgress) {
  testDispatchAndWaitProgress(this->evb_);
}

TYPED_TEST(ScopedEventBaseThreadFloodTest, DispatchAndWaitProgress) {
  TScopedEventBaseThread<TypeParam> evbThread("progress_test");
  testDispatchAndWaitProgress(*evbThread.getEventBase());
}

#ifdef ENABLE_FOLLY

template <>
struct EventBaseAdapter<folly::EventBase> {
  template <typename F>
  static void dispatch(folly::EventBase& evb, F&& func) {
    evb.runInEventBaseThread(std::forward<F>(func));
  }

  template <typename F>
  static void dispatchAndWait(folly::EventBase& evb, F&& func) {
    evb.runInEventBaseThreadAndWait(std::forward<F>(func));
  }
};

// Demonstrates that folly's runInEventBaseThreadAndWait gets starved under
// producer flood. Folly typically cannot complete even a single iteration.
// Run manually with --gtest_also_run_disabled_tests to observe.
TEST(ScopedEventBaseThreadTest, DISABLED_FollyDispatchAndWaitProgress) {
  folly::ScopedEventBaseThread evbThread("progress_test");
  testDispatchAndWaitProgress(*evbThread.getEventBase());
}

#endif // ENABLE_FOLLY

} // namespace uniflow
