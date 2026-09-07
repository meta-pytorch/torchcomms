// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/backends/ib/FlushCompletionTracker.h"

using ctran::ib::FlushCompletionTracker;

TEST(FlushCompletionTrackerDeathTest, ZeroDevicesAborts) {
  EXPECT_DEATH(
      FlushCompletionTracker(/*numDevices=*/0),
      "Flush completion tracking requires at least one device");
}

TEST(FlushCompletionTrackerDeathTest, CompleteOnEmptyDeviceAborts) {
  FlushCompletionTracker tracker(/*numDevices=*/2);

  EXPECT_DEATH(
      tracker.complete(0), "No outstanding flush tracked for device 0");
}

TEST(FlushCompletionTrackerTest, SingleDeviceIsFifo) {
  FlushCompletionTracker tracker(/*numDevices=*/1);
  CtranIbRequest firstFlush;
  CtranIbRequest secondFlush;

  tracker.track(&firstFlush);
  tracker.track(&secondFlush);
  EXPECT_EQ(tracker.outstanding(0), 2);

  EXPECT_EQ(tracker.complete(0), commSuccess);
  EXPECT_TRUE(firstFlush.isComplete());
  EXPECT_FALSE(secondFlush.isComplete());

  EXPECT_EQ(tracker.complete(0), commSuccess);
  EXPECT_TRUE(secondFlush.isComplete());
  EXPECT_EQ(tracker.outstanding(0), 0);
}

// Encodes the defect this class exists to prevent: with a single shared FIFO,
// draining two completions from device 0 would positionally retire the first
// flush even though its read on device 1 is still in flight.
TEST(FlushCompletionTrackerTest, SkewedCrossDeviceCompletion) {
  FlushCompletionTracker tracker(/*numDevices=*/2);
  CtranIbRequest firstFlush;
  CtranIbRequest secondFlush;

  tracker.track(&firstFlush);
  tracker.track(&secondFlush);

  EXPECT_EQ(tracker.complete(0), commSuccess);
  // Retiring a device-0 slot must leave device 1 untouched, otherwise the
  // deques are not independent and cross-device skew is unrepresentable.
  EXPECT_EQ(tracker.outstanding(0), 1);
  EXPECT_EQ(tracker.outstanding(1), 2);

  EXPECT_EQ(tracker.complete(0), commSuccess);
  EXPECT_FALSE(firstFlush.isComplete());
  EXPECT_FALSE(secondFlush.isComplete());

  EXPECT_EQ(tracker.complete(1), commSuccess);
  EXPECT_TRUE(firstFlush.isComplete());
  EXPECT_FALSE(secondFlush.isComplete());

  EXPECT_EQ(tracker.complete(1), commSuccess);
  EXPECT_TRUE(secondFlush.isComplete());
}

TEST(FlushCompletionTrackerTest, AllDevicesRequiredForSingleFlush) {
  constexpr int kNumDevices = 4;
  FlushCompletionTracker tracker(kNumDevices);
  CtranIbRequest flush;

  tracker.track(&flush);
  for (int device = 0; device < kNumDevices - 1; device++) {
    EXPECT_EQ(tracker.complete(device), commSuccess);
    EXPECT_FALSE(flush.isComplete());
  }

  EXPECT_EQ(tracker.complete(kNumDevices - 1), commSuccess);
  EXPECT_TRUE(flush.isComplete());
}

TEST(FlushCompletionTrackerTest, NullSlotsDrainWithoutCompleting) {
  FlushCompletionTracker tracker(/*numDevices=*/2);
  CtranIbRequest flush;

  tracker.track(nullptr);
  tracker.track(&flush);

  EXPECT_EQ(tracker.complete(0), commSuccess);
  EXPECT_EQ(tracker.complete(1), commSuccess);
  EXPECT_FALSE(flush.isComplete());

  EXPECT_EQ(tracker.complete(0), commSuccess);
  EXPECT_FALSE(flush.isComplete());

  EXPECT_EQ(tracker.complete(1), commSuccess);
  EXPECT_TRUE(flush.isComplete());
}
