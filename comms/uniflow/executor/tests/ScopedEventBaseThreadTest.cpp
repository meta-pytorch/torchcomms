// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <atomic>
#include <future>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "comms/uniflow/executor/EventBase.h"
#include "comms/uniflow/executor/LockFreeEventBase.h"
#include "comms/uniflow/executor/MutexEventBase.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"

namespace uniflow {

template <typename T>
class ScopedEventBaseThreadTypedTest : public ::testing::Test {};

using EventBaseTypes = ::testing::Types<MutexEventBase, LockFreeEventBase>;
TYPED_TEST_SUITE(ScopedEventBaseThreadTypedTest, EventBaseTypes);

TYPED_TEST(ScopedEventBaseThreadTypedTest, StartsAndStops) {
  TScopedEventBaseThread<TypeParam> evbThread("test_thread");
  EXPECT_NE(evbThread.getEventBase(), nullptr);
}

TYPED_TEST(ScopedEventBaseThreadTypedTest, GetEventBase) {
  TScopedEventBaseThread<TypeParam> evbThread;
  EventBase* evb = evbThread.getEventBase();
  EXPECT_NE(evb, nullptr);
}

TYPED_TEST(ScopedEventBaseThreadTypedTest, RunWork) {
  TScopedEventBaseThread<TypeParam> evbThread("worker");
  EventBase* evb = evbThread.getEventBase();

  {
    std::promise<int> p;
    auto f = p.get_future();
    evb->dispatch([&p]() noexcept { p.set_value(77); });
    EXPECT_EQ(f.get(), 77);
  }

  {
    std::promise<std::thread::id> p;
    auto f = p.get_future();
    evb->dispatch([&p]() noexcept { p.set_value(std::this_thread::get_id()); });

    auto executedOn = f.get();
    // Should have run on a different thread
    EXPECT_NE(executedOn, std::this_thread::get_id());
    EXPECT_EQ(executedOn, evbThread.getThreadId());
  }
}

TYPED_TEST(ScopedEventBaseThreadTypedTest, SelfDispatch) {
  TScopedEventBaseThread<TypeParam> evbThread("progress");
  EventBase* evb = evbThread.getEventBase();

  std::promise<int> p;
  auto f = p.get_future();
  std::atomic<int> iterations{0};
  constexpr int kTarget = 20;

  evb->dispatch([evb, &p, &iterations]() noexcept {
    auto progressFn = std::make_shared<std::function<void()>>();
    *progressFn = [evb, &p, &iterations, progressFn] {
      int n = iterations.fetch_add(1) + 1;
      if (n >= kTarget) {
        p.set_value(n);
        return;
      }
      evb->dispatch([progressFn]() noexcept { (*progressFn)(); });
    };
    (*progressFn)();
  });

  EXPECT_EQ(f.get(), kTarget);
}

TYPED_TEST(ScopedEventBaseThreadTypedTest, SelfDispatchInline) {
  TScopedEventBaseThread<TypeParam> evbThread("progress");
  EventBase* evb = evbThread.getEventBase();

  std::promise<int> p;
  auto f = p.get_future();
  std::atomic<int> iterations{0};
  constexpr int kTarget = 20;

  evb->dispatch([evb, &p, &iterations]() noexcept {
    auto progressFn = std::make_shared<std::function<void()>>();
    *progressFn = [evb, &p, &iterations, progressFn] {
      int n = iterations.fetch_add(1) + 1;
      if (n >= kTarget) {
        p.set_value(n);
        return;
      }
      evb->dispatchInline([progressFn]() noexcept { (*progressFn)(); });
    };
    (*progressFn)();
  });

  EXPECT_EQ(f.get(), kTarget);
}

} // namespace uniflow
