// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/common/AsyncErrorState.h"

#include <atomic>
#include <thread>

#include <gtest/gtest.h>

namespace comms {
namespace {

TEST(AsyncErrorStateTest, DefaultsToSuccess) {
  AsyncErrorState state;

  const auto snapshot = state.get();
  EXPECT_EQ(snapshot.code, commSuccess);
  EXPECT_TRUE(snapshot.message.empty());
  EXPECT_EQ(state.result(), commSuccess);
}

TEST(AsyncErrorStateTest, LaterErrorReplacesCompleteSnapshot) {
  AsyncErrorState state;
  state.set({commRemoteError, "remote"});
  state.set({commInternalError, "internal"});

  const auto snapshot = state.get();
  EXPECT_EQ(snapshot.code, commInternalError);
  EXPECT_EQ(snapshot.message, "internal");
  EXPECT_EQ(state.result(), commInternalError);
}

TEST(AsyncErrorStateTest, InstancesAreIndependent) {
  AsyncErrorState first;
  AsyncErrorState second;
  first.set({commRemoteError, "first"});

  const auto firstSnapshot = first.get();
  const auto secondSnapshot = second.get();
  EXPECT_EQ(firstSnapshot.code, commRemoteError);
  EXPECT_EQ(firstSnapshot.message, "first");
  EXPECT_EQ(secondSnapshot.code, commSuccess);
  EXPECT_TRUE(secondSnapshot.message.empty());
}

TEST(AsyncErrorStateTest, ConcurrentSnapshotsRemainCoherent) {
  AsyncErrorState state;
  std::atomic<bool> start{false};
  std::atomic<bool> wroteOnce{false};
  std::atomic<bool> stop{false};
  std::atomic<bool> mismatch{false};

  std::thread writer([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    state.set({commRemoteError, "remote"});
    wroteOnce.store(true, std::memory_order_release);
    for (int i = 1; !stop.load(std::memory_order_acquire); ++i) {
      state.set(
          i % 2 == 0 ? AsyncErrorSnapshot{commRemoteError, "remote"}
                     : AsyncErrorSnapshot{commInternalError, "internal"});
    }
  });

  start.store(true, std::memory_order_release);
  while (!wroteOnce.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  for (int i = 0; i < 1000; ++i) {
    const auto snapshot = state.get();
    const bool valid =
        (snapshot.code == commRemoteError && snapshot.message == "remote") ||
        (snapshot.code == commInternalError && snapshot.message == "internal");
    if (!valid) {
      mismatch.store(true, std::memory_order_relaxed);
      break;
    }
  }

  stop.store(true, std::memory_order_release);
  writer.join();
  EXPECT_FALSE(mismatch.load(std::memory_order_relaxed));
}

} // namespace
} // namespace comms
