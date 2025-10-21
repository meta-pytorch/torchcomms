// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "comms/utils/colltrace/CudaEventPool.h"

using namespace meta::comms;
using namespace meta::comms::colltrace;

TEST(CudaEventPool, GetEvent) {
  // Get an event from the pool (should create a new one since pool is empty)
  auto event = CudaEventPool::getEvent();

  // Verify that the event is valid
  ASSERT_NE(event.get(), nullptr);
}

TEST(CudaEventPool, ReturnEventViaMove) {
  // Get two events
  auto event1 = CudaEventPool::getEvent();
  auto event1Ptr = event1.get();

  auto event2 = CudaEventPool::getEvent();
  auto event2Ptr = event2.get();

  // Move event2 to event1, this should return event1 to the pool
  event1 = std::move(event2);

  // The moved-from event2 should now be null, while event1 holding the value
  // of event 2. The use after move is intentional to verify the event is moved
  // correctly
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(event2.get(), nullptr);
  EXPECT_EQ(event1.get(), event2Ptr);

  // Get another event, this should hold the same event as the original event1
  auto event3 = CudaEventPool::getEvent();
  EXPECT_EQ(event3.get(), event1Ptr);
}

TEST(CudaEventPool, ReturnEventViaDestory) {
  cudaEvent_t event1Ptr;
  // Get an event
  {
    auto event1 = CudaEventPool::getEvent();
    event1Ptr = event1.get();
  }

  auto event2 = CudaEventPool::getEvent();

  // Value for both event should be the same
  EXPECT_EQ(event2.get(), event1Ptr);
}

TEST(CudaEventPool, MultipleEvents) {
  // Get multiple events
  auto event1 = CudaEventPool::getEvent();
  auto event2 = CudaEventPool::getEvent();
  auto event3 = CudaEventPool::getEvent();

  // Verify all events are valid
  ASSERT_NE(event1.get(), nullptr);
  ASSERT_NE(event2.get(), nullptr);
  ASSERT_NE(event3.get(), nullptr);

  // Verify all events are different
  EXPECT_NE(event1.get(), event2.get());
  EXPECT_NE(event1.get(), event3.get());
  EXPECT_NE(event2.get(), event3.get());
}

TEST(CudaEventPool, ThreadSafety) {
  constexpr int numThreads = 10;
  constexpr int eventsPerThread = 100;

  std::vector<std::thread> threads;

  // Launch multiple threads that get and return events
  threads.reserve(numThreads);
  for (int i = 0; i < numThreads; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < eventsPerThread; ++j) {
        auto event = CudaEventPool::getEvent();
        ASSERT_NE(event.get(), nullptr);
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // After all threads have completed, get an event and verify it's valid
  auto event = CudaEventPool::getEvent();
  ASSERT_NE(event.get(), nullptr);
}
