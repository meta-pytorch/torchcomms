// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/CollStatsFinalize.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "comms/utils/collstats/CollStatsBank.h"
#include "comms/utils/collstats/CollStatsHistogram.h"
#include "comms/utils/collstats/CollStatsTypes.h"

namespace meta::comms::collstats {

/* The tests assert against the cvar defaults explicitly, rather than against
 * whatever the compiled-in constants happen to be. */
const CollStatHistGeometry kTestGeom = collStatDefaultHistGeometry();
namespace {

constexpr uint64_t kSec = 1'000'000'000ull;

uint64_t histogramTotal(const CollStatValue& v) {
  uint64_t total = 0;
  for (uint32_t b = 0; b < kHistMaxBuckets; ++b) {
    total += v.histogram[b];
  }
  return total;
}

/* Host models of the two readout steps. Production runs both against device
 * memory from CollStatsReader.cu -- the flip is a one-thread kernel on the
 * reader stream, the reset a cudaMemsetAsync -- so neither has a callable host
 * form, and deliberately so: a host helper taking a bare CollStatValue* would
 * invite a host memset of a device pointer. These stand in for them so the
 * bank-independence contract can be exercised without a GPU. Single-threaded,
 * so plain arithmetic is enough. */
uint32_t flipEpochLikeReader(CollStatDoubleBank& bank) {
  const uint32_t retired = static_cast<uint32_t>(bank.epoch & 1u);
  ++bank.epoch;
  return retired;
}

void zeroBankLikeReader(CollStatValue* values, uint32_t slots) {
  std::memset(
      values, 0, static_cast<std::size_t>(slots) * sizeof(CollStatValue));
}

// The minimum rides an atomicMax on the complement so a zeroed bank reads as
// unset; check both the unset case and that it tracks the smallest duration.
TEST(CollStatsFinalizeTest, MinimumIsExactAndUnsetOnAZeroedValue) {
  CollStatValue v{};
  EXPECT_EQ(collStatDurMinNs(v), 0ull) << "zeroed bank must read as no data";

  collStatAccumulate(&v, 5'000ull, 128);
  collStatAccumulate(&v, 900ull, 128);
  collStatAccumulate(&v, 70'000ull, 128);

  EXPECT_EQ(collStatDurMinNs(v), 900ull);
  EXPECT_EQ(v.durMaxNs, 70'000ull);
}

TEST(CollStatsFinalizeTest, AccumulateAggregatesScalarsAndThresholds) {
  CollStatValue v{};
  // Durations chosen so the exceed-counts are hand-derivable against the
  // default thresholds {1s, 10s, 60s, 600s}.
  const std::vector<uint64_t> durs = {
      kSec / 2, 2 * kSec, 15 * kSec, 70 * kSec, 700 * kSec};
  const uint64_t bytesEach = 1000;
  uint64_t expectedSum = 0;
  for (uint64_t d : durs) {
    collStatAccumulate(&v, d, bytesEach);
    expectedSum += d;
  }

  EXPECT_EQ(v.count, durs.size());
  EXPECT_EQ(v.logicalBytes, bytesEach * durs.size());
  EXPECT_EQ(v.durationSumNs, expectedSum);
  EXPECT_EQ(v.durMaxNs, 700 * kSec);
  EXPECT_EQ(histogramTotal(v), durs.size());

  // Exceed-counts: >=1s -> 4, >=10s -> 3, >=60s -> 2, >=600s -> 1.
  const std::vector<uint64_t> expectedThresholds = {4, 3, 2, 1};
  std::vector<uint64_t> actual(
      v.thresholdCounts, v.thresholdCounts + kMaxThresholds);
  EXPECT_EQ(actual, expectedThresholds);
}

TEST(CollStatsFinalizeTest, HistogramBucketMatchesLogBucket) {
  CollStatValue v{};
  const uint64_t dur = 42 * kSec;
  collStatAccumulate(&v, dur, 0);
  EXPECT_EQ(v.histogram[logBucketNs(kTestGeom, dur)], 1u);
}

// Each window gets its own totals: a flip must leave the retired bank intact
// for the reader and start the new one clean. The flip and the reset here are
// host models of the reader's device steps (see above), so what this pins is
// the bank/epoch contract in CollStatsBank.h, not the reader's CUDA sequence --
// that is covered by CollStatsReaderGpuTest.
TEST(CollStatsFinalizeTest, BanksStayIndependentAcrossAnEpochFlip) {
  const uint32_t numKeys = 4;
  std::vector<CollStatValue> bankA(numKeys + 1);
  std::vector<CollStatValue> bankB(numKeys + 1);
  CollStatDoubleBank bank{};
  bank.numKeys = numKeys;
  bank.epoch = 0;
  bank.values[0] = bankA.data();
  bank.values[1] = bankB.data();

  // Window 0 writes bank A.
  collStatAccumulate(&collStatCurrentValues(&bank)[0], 5 * kSec, 100);
  EXPECT_EQ(bankA[0].count, 1u);
  EXPECT_EQ(bankB[0].count, 0u);

  // Flip: A retires, B becomes current.
  const uint32_t retired = flipEpochLikeReader(bank);
  EXPECT_EQ(retired, 0u);
  collStatAccumulate(&collStatCurrentValues(&bank)[0], 9 * kSec, 200);
  EXPECT_EQ(bankB[0].count, 1u);
  EXPECT_EQ(bankA[0].count, 1u); // untouched by the new window

  // Reader zeroes the retired bank; the live bank is unaffected.
  zeroBankLikeReader(bank.values[retired], numKeys + 1);
  EXPECT_EQ(bankA[0].count, 0u);
  EXPECT_EQ(bankB[0].count, 1u);
}

} // namespace
} // namespace meta::comms::collstats
