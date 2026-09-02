// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/CollStatsReadout.h"

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include "comms/utils/collstats/CollStatsFinalize.h"
#include "comms/utils/collstats/CollStatsHistogram.h"
#include "comms/utils/collstats/CollStatsTypes.h"

namespace meta::comms::collstats {

/* The tests assert against the cvar defaults explicitly, rather than against
 * whatever the compiled-in constants happen to be. */
const CollStatHistGeometry kTestGeom = collStatDefaultHistGeometry();
namespace {

constexpr uint64_t kSec = 1'000'000'000ull;

TEST(CollStatsReadoutTest, BucketLowerIsInverseOfLogBucket) {
  // A duration at an exact octave start must invert back to itself.
  for (uint32_t k = 0; k < kDefaultHistOctaves; ++k) {
    const uint64_t durNs = kDefaultHistTMinNs << k;
    const uint32_t bucket = logBucketNs(kTestGeom, durNs);
    EXPECT_NEAR(
        collStatBucketLowerNs(kTestGeom, bucket),
        static_cast<double>(durNs),
        1.0)
        << "octave k=" << k;
  }
}

TEST(CollStatsReadoutTest, AverageIsSumOverCount) {
  CollStatValue v{};
  collStatAccumulate(&v, 2 * kSec, 0);
  collStatAccumulate(&v, 4 * kSec, 0);
  EXPECT_DOUBLE_EQ(collStatAvgDurationNs(v), 3.0 * kSec);
}

TEST(CollStatsReadoutTest, PercentileOfSingleBucketIsThatBucket) {
  CollStatValue v{};
  const uint64_t dur = 10 * kSec;
  for (int i = 0; i < 100; ++i) {
    collStatAccumulate(&v, dur, 0);
  }
  const double edge =
      collStatBucketLowerNs(kTestGeom, logBucketNs(kTestGeom, dur));
  EXPECT_DOUBLE_EQ(collStatPercentileNs(kTestGeom, v, 0.5), edge);
  EXPECT_DOUBLE_EQ(collStatPercentileNs(kTestGeom, v, 0.99), edge);
}

TEST(CollStatsReadoutTest, PercentileSeparatesBulkFromTail) {
  CollStatValue v{};
  for (int i = 0; i < 99; ++i) {
    collStatAccumulate(&v, 1 * kSec, 0);
  }
  collStatAccumulate(&v, 500 * kSec, 0); // one tail sample

  // P50 sits in the bulk; P99.9 must reach the tail bucket.
  EXPECT_LT(
      collStatPercentileNs(kTestGeom, v, 0.5), static_cast<double>(100 * kSec));
  EXPECT_GE(
      collStatPercentileNs(kTestGeom, v, 0.999),
      static_cast<double>(400 * kSec));
}

// The lower edge of the last interior bucket is the value a consumer needs to
// place the tail correctly, and it is the one a naive reconstruction from the
// octave count gets wrong: at the defaults it opens at 1us * 2^(239/8) =
// 984.6s, not at a whole 2^30 octaves above tMinNs.
TEST(CollStatsReadoutTest, LastInteriorBucketOpensBelowTMax) {
  const uint32_t lastInterior = kTestGeom.numBuckets - 2;
  const double edge = collStatBucketLowerNs(kTestGeom, lastInterior);
  EXPECT_NEAR(edge, 984.6e9, 0.1e9);
  EXPECT_LT(edge, static_cast<double>(kTestGeom.tMaxNs));
}

TEST(CollStatsReadoutTest, UnderflowAndOverflowEdgesAreExact) {
  EXPECT_DOUBLE_EQ(collStatBucketLowerNs(kTestGeom, kHistUnderflowBucket), 0.0);
  EXPECT_DOUBLE_EQ(
      collStatBucketLowerNs(kTestGeom, kTestGeom.numBuckets - 1),
      static_cast<double>(kTestGeom.tMaxNs));
}

TEST(CollStatsReadoutTest, NoObservationsReadsAsZero) {
  const CollStatValue empty{};
  EXPECT_DOUBLE_EQ(collStatAvgDurationNs(empty), 0.0);
  EXPECT_DOUBLE_EQ(collStatPercentileNs(kTestGeom, empty, 0.5), 0.0);
}

} // namespace
} // namespace meta::comms::collstats
