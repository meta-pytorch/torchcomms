// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Exercises the PrecisionClock fallback path (NCCL_USE_PTP=0). The
// fbclock daemon is not assumed to be present in this test environment;
// PTP-mode behavior is left to integration tests on PTP-enabled hosts.
//
// PrecisionClock initializes its singleton on first use and caches the
// kill-switch decision for the process lifetime, so tests in this file
// must run in a binary that never calls precision* APIs before SetUp.

#include <chrono>
#include <cstdlib>

#include <gtest/gtest.h>

#include "comms/utils/colltrace/PrecisionClock.h"

using meta::comms::colltrace::precisionErrorNs;
using meta::comms::colltrace::precisionNow;
using meta::comms::colltrace::precisionNowNs;
using meta::comms::colltrace::precisionNowRangeNs;
using meta::comms::colltrace::precisionUsingPtp;

class PrecisionClockFallbackTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Force fallback before any precision* call lazily initializes the
    // singleton. setenv with overwrite=1 so a stale env from the runner
    // can't accidentally enable PTP.
    setenv("NCCL_USE_PTP", "0", 1);
  }
};

TEST_F(PrecisionClockFallbackTest, ReportsFallbackMode) {
  EXPECT_FALSE(precisionUsingPtp());
}

TEST_F(PrecisionClockFallbackTest, RangeCollapsesInFallback) {
  auto [earliest, latest] = precisionNowRangeNs();
  EXPECT_EQ(earliest, latest);
}

TEST_F(PrecisionClockFallbackTest, ErrorIsZeroInFallback) {
  EXPECT_EQ(precisionErrorNs(), 0u);
}

TEST_F(PrecisionClockFallbackTest, NowTracksSystemClock) {
  auto before = std::chrono::system_clock::now();
  auto p = precisionNow();
  auto after = std::chrono::system_clock::now();
  EXPECT_GE(p, before - std::chrono::seconds(1));
  EXPECT_LE(p, after + std::chrono::seconds(1));
}

TEST_F(PrecisionClockFallbackTest, NsIsMonotonicEnough) {
  // system_clock is not strictly monotonic but consecutive calls on the
  // same thread without an NTP step should be non-decreasing in practice.
  uint64_t prev = precisionNowNs();
  for (int i = 0; i < 100; ++i) {
    uint64_t cur = precisionNowNs();
    EXPECT_GE(cur, prev);
    prev = cur;
  }
}
