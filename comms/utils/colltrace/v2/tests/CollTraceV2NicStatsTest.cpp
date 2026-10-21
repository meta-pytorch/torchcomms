// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "comms/utils/colltrace/nicstats/NicStatsLatest.h"
#include "comms/utils/colltrace/v2/CollTraceV2.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace meta::comms::colltracev2 {
namespace {

using colltrace::nicstats::NicStatsLatest;
using colltrace::nicstats::NicStatsReading;

NicStatsReading readingOf(const std::string& device) {
  NicStatsReading reading;
  device.copy(reading.rail.device.data(), device.size());
  reading.rail.port = 1;
  return reading;
}

// Construction starts NIC stats, and nicStats() copies out what they put.
TEST(CollTraceV2NicStatsTest, ConstructionStartsNicStats) {
  int starts = 0;
  CollTraceV2 ct{
      [&starts](
          std::shared_ptr<NicStatsLatest> latest) -> std::shared_ptr<void> {
        ++starts;
        latest->put(readingOf("mlx5_0"));
        return std::make_shared<int>(0);
      }};

  const auto readings = ct.nicStats();

  EXPECT_EQ(starts, 1);
  ASSERT_EQ(readings.size(), 1u);
  EXPECT_STREQ(readings[0].rail.device.data(), "mlx5_0");
}

// A start that reads nothing leaves nicStats() empty.
TEST(CollTraceV2NicStatsTest, AStartThatReadsNothingLeavesNicStatsEmpty) {
  CollTraceV2 ct{[](std::shared_ptr<NicStatsLatest>) -> std::shared_ptr<void> {
    return nullptr;
  }};

  EXPECT_TRUE(ct.nicStats().empty());
}

// With NICSTATS_ENABLE off, the real start reads nothing.
TEST(CollTraceV2NicStatsTest, WithNicStatsOffNothingIsRead) {
  ncclCvarInit();
  CollTraceV2 ct;

  EXPECT_TRUE(ct.nicStats().empty());
}

} // namespace
} // namespace meta::comms::colltracev2
