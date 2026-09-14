// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/BoundedCleanupExecutor.h"

#include <atomic>
#include <functional>
#include <future>

#include <gtest/gtest.h>

namespace comms::prims::detail {
namespace {

struct TestCleanup {
  std::function<void()> fn;

  void operator()() noexcept {
    fn();
  }
};

TEST(BoundedCleanupExecutorTest, RunsTaskAndDrains) {
  // drain() must not return until the accepted cleanup has completed.
  std::atomic<int> calls{0};
  BoundedCleanupExecutor<TestCleanup> executor;
  TestCleanup task{[&calls] { ++calls; }};

  ASSERT_TRUE(executor.tryEnqueue(task));
  executor.drain();

  EXPECT_EQ(calls.load(), 1);
}

TEST(BoundedCleanupExecutorTest, RejectsWhileWorkerIsOccupied) {
  // A blocked cleanup consumes the sole asynchronous slot; its successor stays
  // caller-owned so the caller can reclaim it synchronously.
  std::promise<void> started;
  auto startedFuture = started.get_future();
  std::promise<void> release;
  auto releaseFuture = release.get_future().share();
  std::atomic<int> calls{0};
  BoundedCleanupExecutor<TestCleanup> executor;
  TestCleanup blockingTask{[&] {
    started.set_value();
    releaseFuture.wait();
    ++calls;
  }};

  ASSERT_TRUE(executor.tryEnqueue(blockingTask));
  startedFuture.wait();

  TestCleanup fallbackTask{[&calls] { ++calls; }};
  ASSERT_FALSE(executor.tryEnqueue(fallbackTask));
  fallbackTask();

  release.set_value();
  executor.drain();
  EXPECT_EQ(calls.load(), 2);
}

TEST(BoundedCleanupExecutorTest, ShutdownDrainsAndRejectsNewTasks) {
  // Shutdown drains accepted cleanup, is idempotent, and permanently closes
  // admission so late owners can fall back synchronously.
  std::atomic<int> calls{0};
  BoundedCleanupExecutor<TestCleanup> executor;
  TestCleanup queuedTask{[&calls] { ++calls; }};

  ASSERT_TRUE(executor.tryEnqueue(queuedTask));
  executor.shutdown();
  EXPECT_EQ(calls.load(), 1);

  TestCleanup rejectedTask{[&calls] { ++calls; }};
  EXPECT_FALSE(executor.tryEnqueue(rejectedTask));
  rejectedTask();
  EXPECT_EQ(calls.load(), 2);

  executor.shutdown();
}

} // namespace
} // namespace comms::prims::detail
