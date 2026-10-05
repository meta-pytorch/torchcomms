// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "comms/utils/collstats/switchstats/SwitchSampleClock.h"

namespace meta::comms::switchstats {

// A stamp left at its default, or a pair in the wrong order, must not become a
// number that looks like a measurement.
TEST(SwitchSampleClockTest, AnUnsetStampSaturatesRatherThanWrapping) {
  EXPECT_EQ(switchElapsedNs(1000, 0), 0u);
  EXPECT_EQ(switchElapsedNs(1000, 1000), 0u);
  EXPECT_EQ(switchElapsedNs(0, 1000), 1000u);
  EXPECT_EQ(switchElapsedNs(std::numeric_limits<uint64_t>::max(), 1), 0u);
}

TEST(SwitchSampleClockTest, TheSteadyClockDoesNotCountFromTheUnixEpoch) {
  // Around fifty years apart, so this cannot pass by coincidence: it is what
  // makes a steady stamp unusable as a timestamp and a wall stamp unusable as
  // a duration.
  EXPECT_LT(nowSteadyNs(), nowUnixNs());
}

TEST(SwitchSampleClockTest, TheSteadyClockOnlyMovesForward) {
  const uint64_t first = nowSteadyNs();
  const uint64_t second = nowSteadyNs();
  EXPECT_GE(second, first);
}

} // namespace meta::comms::switchstats
