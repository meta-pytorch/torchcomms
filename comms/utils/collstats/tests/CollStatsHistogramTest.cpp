// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/CollStatsHistogram.h"

#include <cstdint>

#include <gtest/gtest.h>

#include "comms/utils/collstats/CollStatsTypes.h"

namespace meta::comms::collstats {

/* The tests assert against the cvar defaults explicitly, rather than against
 * whatever the compiled-in constants happen to be. */
const CollStatHistGeometry kTestGeom = collStatDefaultHistGeometry();
namespace {

TEST(CollStatsHistogramTest, DurationsBelowTMinAreUnderflow) {
  EXPECT_EQ(logBucketNs(kTestGeom, 0), kHistUnderflowBucket);
  EXPECT_EQ(logBucketNs(kTestGeom, 1), kHistUnderflowBucket);
  EXPECT_EQ(
      logBucketNs(kTestGeom, kDefaultHistTMinNs - 1), kHistUnderflowBucket);
}

TEST(CollStatsHistogramTest, DurationsAtOrAboveTMaxAreOverflow) {
  EXPECT_EQ(logBucketNs(kTestGeom, kDefaultHistTMaxNs), (kHistMaxBuckets - 1));
  EXPECT_EQ(
      logBucketNs(kTestGeom, kDefaultHistTMaxNs + 1), (kHistMaxBuckets - 1));
  EXPECT_EQ(logBucketNs(kTestGeom, ~0ull), (kHistMaxBuckets - 1));
}

TEST(CollStatsHistogramTest, TMinLandsInFirstInteriorBucket) {
  // The first in-range bucket sits immediately after the underflow bucket.
  EXPECT_EQ(
      logBucketNs(kTestGeom, kDefaultHistTMinNs), kHistUnderflowBucket + 1);
}

TEST(CollStatsHistogramTest, OctaveStartsAreEvenlySpaced) {
  // Independently derived expectation: a duration exactly k octaves above t_min
  // must land at the start of octave k, i.e. bucket 1 + k*S.
  for (uint32_t k = 0; k < kDefaultHistOctaves; ++k) {
    const uint64_t durNs = kDefaultHistTMinNs << k;
    const uint32_t expected = 1u + k * kDefaultHistSubBucketsPerOctave;
    EXPECT_EQ(logBucketNs(kTestGeom, durNs), expected) << "octave k=" << k;
  }
}

TEST(CollStatsHistogramTest, MidOctaveResolvesToSubBucket) {
  // A point 0.3 octaves in (2^0.3 above the octave start) is sub-bucket
  // floor(8 * 0.3) = 2. The fraction is kept well off a sub-bucket edge so the
  // double->uint64 truncation of the test input can't tip it across a boundary.
  const uint32_t k = 5;
  const double frac = 0.3;
  const double point =
      static_cast<double>(kDefaultHistTMinNs << k) * 1.2311444133449163;
  const uint32_t expectedSub =
      static_cast<uint32_t>(frac * kDefaultHistSubBucketsPerOctave); // 2
  const uint32_t expected =
      1u + k * kDefaultHistSubBucketsPerOctave + expectedSub;
  EXPECT_EQ(logBucketNs(kTestGeom, static_cast<uint64_t>(point)), expected);
}

TEST(CollStatsHistogramTest, BucketsAreMonotonicAndInRange) {
  uint32_t prev = 0;
  for (uint64_t durNs = 1; durNs < kDefaultHistTMaxNs * 2;
       durNs += durNs / 16 + 1) {
    const uint32_t b = logBucketNs(kTestGeom, durNs);
    EXPECT_LT(b, kHistMaxBuckets);
    EXPECT_GE(b, prev) << "non-monotonic at durNs=" << durNs;
    prev = b;
  }
}

// The one function here with a validation contract: a rejected geometry must
// report numBuckets 0 rather than something a caller would bucket with.
TEST(CollStatsHistogramTest, RejectedGeometriesReportZeroBuckets) {
  EXPECT_EQ(collStatMakeHistGeometry(0, 1'000'000, 8).numBuckets, 0u)
      << "tMin of zero has no log";
  EXPECT_EQ(collStatMakeHistGeometry(1'000, 1'000, 8).numBuckets, 0u)
      << "degenerate span";
  EXPECT_EQ(collStatMakeHistGeometry(1'000'000, 1'000, 8).numBuckets, 0u)
      << "inverted bounds";
  EXPECT_EQ(collStatMakeHistGeometry(1'000, 1'000'000, 0).numBuckets, 0u)
      << "no sub-buckets";
  // 30 octaves at 16 sub-buckets is 482 buckets, well past the capacity.
  EXPECT_EQ(
      collStatMakeHistGeometry(kDefaultHistTMinNs, kDefaultHistTMaxNs, 16)
          .numBuckets,
      0u)
      << "over kHistMaxBuckets";
}

// The defaults are derived, not asserted, so they must land exactly on the
// capacity they are sized against.
TEST(CollStatsHistogramTest, DefaultGeometryFillsTheCapacity) {
  const CollStatHistGeometry g = collStatDefaultHistGeometry();
  EXPECT_EQ(g.numBuckets, kHistMaxBuckets);
  EXPECT_EQ(g.tMinNs, kDefaultHistTMinNs);
  EXPECT_EQ(g.subBucketsPerOctave, kDefaultHistSubBucketsPerOctave);
}

// Bucketing must follow a configured geometry, not the compiled-in one --
// that configurability is the whole point of carrying the geometry around.
TEST(CollStatsHistogramTest, BucketingFollowsAConfiguredGeometry) {
  // 1ms..1s at 4 sub-buckets per octave: ~9.97 octaves -> 40 interior + 2.
  const CollStatHistGeometry g =
      collStatMakeHistGeometry(1'000'000, 1'000'000'000, 4);
  ASSERT_EQ(g.numBuckets, 42u);

  EXPECT_EQ(logBucketNs(g, 999'999), kHistUnderflowBucket)
      << "below tMin under this geometry, though interior under the default";
  EXPECT_EQ(logBucketNs(g, 1'000'000), 1u);
  EXPECT_EQ(logBucketNs(g, 2'000'000), 1u + 4u) << "one octave in";
  EXPECT_EQ(logBucketNs(g, 1'000'000'000), g.numBuckets - 1) << "overflow";
}

// The default edges must keep reproducing floor(log2(bytes)) exactly, so rows
// recorded before the edges became configurable stay comparable.
TEST(CollStatsHistogramTest, DefaultSizeClassesAreFloorLog2) {
  const CollStatSizeClasses sc = collStatDefaultSizeClasses();
  EXPECT_EQ(sizeClassOf(sc, 0), 0u);
  EXPECT_EQ(sizeClassOf(sc, 1), 0u);
  EXPECT_EQ(sizeClassOf(sc, 2), 1u);
  EXPECT_EQ(sizeClassOf(sc, 3), 1u);
  EXPECT_EQ(sizeClassOf(sc, 1024), 10u);
  EXPECT_EQ(sizeClassOf(sc, (1ull << 30) + 7), 30u);
  EXPECT_EQ(sizeClassOf(sc, 8ull << 30), 33u);
}

// A configured edge set replaces the defaults wholesale: class 0 is everything
// below the first edge, and the last class is open-ended.
TEST(CollStatsHistogramTest, ConfiguredEdgesBucketByLastEdgeAtOrBelow) {
  CollStatSizeClasses sc{};
  sc.n = 3;
  sc.edges[0] = 1024;
  sc.edges[1] = 1ull << 20;
  sc.edges[2] = 1ull << 30;

  EXPECT_EQ(sizeClassOf(sc, 0), 0u);
  EXPECT_EQ(sizeClassOf(sc, 1023), 0u);
  EXPECT_EQ(sizeClassOf(sc, 1024), 1u);
  EXPECT_EQ(sizeClassOf(sc, (1ull << 20) - 1), 1u);
  EXPECT_EQ(sizeClassOf(sc, 1ull << 20), 2u);
  EXPECT_EQ(sizeClassOf(sc, 8ull << 30), 3u);
}

} // namespace
} // namespace meta::comms::collstats
