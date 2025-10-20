// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "comms/utils/colltrace/CPUWaitEvent.h"

using namespace meta::comms;
using namespace meta::comms::colltrace;

TEST(CPUWaitEvent, Constructor) {
  auto event = std::make_unique<CPUWaitEvent>();
  // Verify that the enqueue time is set during construction
  auto enqueueTimeResult = event->getCollEnqueueTime();
  ASSERT_TRUE(enqueueTimeResult.hasValue());

  // Enqueue time should be close to now
  auto now = std::chrono::system_clock::now();
  auto enqueueTime = enqueueTimeResult.value();
  auto diff =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - enqueueTime)
          .count();

  // The difference should be small (less than 1 second)
  EXPECT_LT(diff, 1000);
}

TEST(CPUWaitEvent, SignalAndWaitCollStart) {
  auto event = std::make_unique<CPUWaitEvent>();
  // Start a thread that will wait for the collective to start
  std::thread waiterThread([&]() {
    auto waitResult = event->waitCollStart(std::chrono::milliseconds(10000));
    ASSERT_TRUE(waitResult.hasValue());
    EXPECT_TRUE(waitResult.value());
  });

  // Sleep briefly to ensure the waiter thread is waiting
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Signal that the collective has started
  auto signalResult = event->signalCollStart();
  ASSERT_TRUE(signalResult.hasValue());

  // Wait for the waiter thread to complete
  waiterThread.join();

  // Verify that the start time was set
  auto startTimeResult = event->getCollStartTime();
  ASSERT_TRUE(startTimeResult.hasValue());

  // Start time should be after enqueue time
  auto enqueueTime = event->getCollEnqueueTime().value();
  auto startTime = startTimeResult.value();
  EXPECT_GE(startTime, enqueueTime);
}

TEST(CPUWaitEvent, SignalAndWaitCollStartTwice) {
  auto event = std::make_unique<CPUWaitEvent>();
  // Start a thread that will wait for the collective to start
  std::thread waiterThread([&]() {
    // First wait should fail
    auto waitResult = event->waitCollStart(std::chrono::milliseconds(1));
    ASSERT_TRUE(waitResult.hasValue());
    EXPECT_FALSE(waitResult.value());
    // Second wait should succeed
    waitResult = event->waitCollStart(std::chrono::milliseconds(1000));
    ASSERT_TRUE(waitResult.hasValue());
    EXPECT_TRUE(waitResult.value());
  });

  // Sleep briefly to ensure the waiter thread is waiting
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Signal that the collective has started
  auto signalResult = event->signalCollStart();
  ASSERT_TRUE(signalResult.hasValue());

  // Wait for the waiter thread to complete
  waiterThread.join();

  // Verify that the start time was set
  auto startTimeResult = event->getCollStartTime();
  ASSERT_TRUE(startTimeResult.hasValue());

  // Start time should be after enqueue time
  auto enqueueTime = event->getCollEnqueueTime().value();
  auto startTime = startTimeResult.value();
  EXPECT_GE(startTime, enqueueTime);
}

TEST(CPUWaitEvent, SignalAndWaitCollEndTwice) {
  auto event = std::make_unique<CPUWaitEvent>();
  // Start a thread that will wait for the collective to end
  std::thread waiterThread([&]() {
    // First wait should fail (timeout)
    auto waitResult = event->waitCollEnd(std::chrono::milliseconds(1));
    ASSERT_TRUE(waitResult.hasValue());
    EXPECT_FALSE(waitResult.value());
    // Second wait should succeed after signal
    waitResult = event->waitCollEnd(std::chrono::milliseconds(1000));
    ASSERT_TRUE(waitResult.hasValue());
    EXPECT_TRUE(waitResult.value());
  });

  // Sleep briefly to ensure the waiter thread is waiting
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Signal that the collective has ended
  auto signalResult = event->signalCollEnd();
  ASSERT_TRUE(signalResult.hasValue());

  // Wait for the waiter thread to complete
  waiterThread.join();

  // Verify that the end time was set
  auto endTimeResult = event->getCollEndTime();
  ASSERT_TRUE(endTimeResult.hasValue());

  // End time should be after enqueue time
  auto enqueueTime = event->getCollEnqueueTime().value();
  auto endTime = endTimeResult.value();
  EXPECT_GE(endTime, enqueueTime);
}

TEST(CPUWaitEvent, SignalAndWaitCollEnd) {
  auto event = std::make_unique<CPUWaitEvent>();
  // Start a thread that will wait for the collective to end
  std::thread waiterThread([&]() {
    auto waitResult = event->waitCollEnd(std::chrono::milliseconds(1000));
    ASSERT_TRUE(waitResult.hasValue());
    EXPECT_TRUE(waitResult.value());
  });

  // Sleep briefly to ensure the waiter thread is waiting
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Signal that the collective has ended
  auto signalResult = event->signalCollEnd();
  ASSERT_TRUE(signalResult.hasValue());

  // Wait for the waiter thread to complete
  waiterThread.join();

  // Verify that the end time was set
  auto endTimeResult = event->getCollEndTime();
  ASSERT_TRUE(endTimeResult.hasValue());

  // End time should be after enqueue time
  auto enqueueTime = event->getCollEnqueueTime().value();
  auto endTime = endTimeResult.value();
  EXPECT_GE(endTime, enqueueTime);
}

TEST(CPUWaitEvent, WaitCollStartTimeout) {
  auto event = std::make_unique<CPUWaitEvent>();
  // Wait for a very short time, should time out
  auto waitResult = event->waitCollStart(std::chrono::milliseconds(1));
  ASSERT_TRUE(waitResult.hasValue());
  EXPECT_FALSE(waitResult.value());
}

TEST(CPUWaitEvent, WaitCollEndTimeout) {
  auto event = std::make_unique<CPUWaitEvent>();
  // Wait for a very short time, should time out
  auto waitResult = event->waitCollEnd(std::chrono::milliseconds(1));
  ASSERT_TRUE(waitResult.hasValue());
  EXPECT_FALSE(waitResult.value());
}

TEST(CPUWaitEvent, WaitMultipleCollStartTimeout) {
  auto event = std::make_unique<CPUWaitEvent>();
  // Wait for a very short time, should time out
  for (int i = 0; i < 10; i++) {
    auto waitResult = event->waitCollStart(std::chrono::milliseconds(1));
    ASSERT_TRUE(waitResult.hasValue());
    EXPECT_FALSE(waitResult.value());
  }
}

TEST(CPUWaitEvent, WaitMultipleCollEndTimeout) {
  auto event = std::make_unique<CPUWaitEvent>();
  // Wait for a very short time, should time out
  for (int i = 0; i < 10; i++) {
    auto waitResult = event->waitCollEnd(std::chrono::milliseconds(1));
    ASSERT_TRUE(waitResult.hasValue());
    EXPECT_FALSE(waitResult.value());
  }
}

TEST(CPUWaitEvent, GetCollStartTimeBeforeSignal) {
  auto event = std::make_unique<CPUWaitEvent>();
  // Trying to get start time before signaling should return an error
  auto startTimeResult = event->getCollStartTime();
  ASSERT_FALSE(startTimeResult.hasValue());
  EXPECT_EQ(startTimeResult.error().errorCode, commInternalError);
  EXPECT_EQ(
      startTimeResult.error().message,
      "CPUWaitEvent: getCollStartTime called before start time ready");
}

TEST(CPUWaitEvent, GetCollEndTimeBeforeSignal) {
  auto event = std::make_unique<CPUWaitEvent>();
  // Trying to get end time before signaling should return an error
  auto endTimeResult = event->getCollEndTime();
  ASSERT_FALSE(endTimeResult.hasValue());
  EXPECT_EQ(endTimeResult.error().errorCode, commInternalError);
  EXPECT_EQ(
      endTimeResult.error().message,
      "CPUWaitEvent: getCollEndTime called before end time ready");
}

TEST(CPUWaitEvent, CompleteSequence) {
  auto event = std::make_unique<CPUWaitEvent>();
  // Test the complete sequence of operations

  // 1. Get enqueue time
  auto enqueueTimeResult = event->getCollEnqueueTime();
  ASSERT_TRUE(enqueueTimeResult.hasValue());
  auto enqueueTime = enqueueTimeResult.value();

  // 2. Signal collective start
  auto signalStartResult = event->signalCollStart();
  ASSERT_TRUE(signalStartResult.hasValue());

  // 3. Get start time
  auto startTimeResult = event->getCollStartTime();
  ASSERT_TRUE(startTimeResult.hasValue());
  auto startTime = startTimeResult.value();

  // 4. Signal collective end
  auto signalEndResult = event->signalCollEnd();
  ASSERT_TRUE(signalEndResult.hasValue());

  // 5. Get end time
  auto endTimeResult = event->getCollEndTime();
  ASSERT_TRUE(endTimeResult.hasValue());
  auto endTime = endTimeResult.value();

  // 6. Verify the sequence of timestamps
  EXPECT_LE(enqueueTime, startTime);
  EXPECT_LE(startTime, endTime);
}
