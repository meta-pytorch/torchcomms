// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <thread>

#include "comms/utils/trainer/TrainerContext.h"

TEST(TrainerContextTest, SetIterationToNegativeOne) {
  ncclxSetIteration(-1);
  EXPECT_EQ(ncclxGetIteration(), -1);
  // Timestamp is still set (setIteration always records wall-clock time)
  EXPECT_GT(ncclxGetIterationTimestampUs(), 0);
}

TEST(TrainerContextTest, SetIterationUpdatesValue) {
  ncclxSetIteration(42);
  EXPECT_EQ(ncclxGetIteration(), 42);
  EXPECT_GT(ncclxGetIterationTimestampUs(), 0);
  ncclxSetIteration(-1);
}

TEST(TrainerContextTest, SnapshotIsConsistent) {
  ncclxSetIteration(10);
  auto snap = ncclxGetIterationSnapshot();
  EXPECT_EQ(snap.iteration, 10);
  EXPECT_GT(snap.timestampUs, 0);
  ncclxSetIteration(-1);
}

TEST(TrainerContextTest, TimestampAdvancesWithIteration) {
  ncclxSetIteration(1);
  auto snap1 = ncclxGetIterationSnapshot();

  std::this_thread::sleep_for(std::chrono::milliseconds(2));

  ncclxSetIteration(2);
  auto snap2 = ncclxGetIterationSnapshot();

  EXPECT_GT(snap2.iteration, snap1.iteration);
  EXPECT_GT(snap2.timestampUs, snap1.timestampUs);
  ncclxSetIteration(-1);
}

TEST(TrainerContextTest, PackUnpackRoundtrip) {
  ncclxSetIteration(999);
  auto snap = ncclxGetIterationSnapshot();
  EXPECT_EQ(snap.iteration, 999);

  ncclxSetIteration(0);
  snap = ncclxGetIterationSnapshot();
  EXPECT_EQ(snap.iteration, 0);
  EXPECT_GT(snap.timestampUs, 0);

  ncclxSetIteration(-1);
  snap = ncclxGetIterationSnapshot();
  EXPECT_EQ(snap.iteration, -1);
}

TEST(TrainerContextTest, SnapshotNotTorn) {
  ncclxSetIteration(100);
  auto snap1 = ncclxGetIterationSnapshot();

  std::this_thread::sleep_for(std::chrono::milliseconds(2));

  ncclxSetIteration(200);
  auto snap2 = ncclxGetIterationSnapshot();

  // Each snapshot must be self-consistent: snap1's timestamp must be <=
  // snap2's timestamp, and neither should have a mismatched pair.
  EXPECT_EQ(snap1.iteration, 100);
  EXPECT_EQ(snap2.iteration, 200);
  EXPECT_LE(snap1.timestampUs, snap2.timestampUs);
  ncclxSetIteration(-1);
}
