// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/backends/ib/CtranIbVc.h"
#include "comms/testinfra/TestXPlatUtils.h"

class CtranIbQpConfigTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {
 protected:
  void SetUp() override {
    ncclCvarInit();
  }
};

TEST_P(CtranIbQpConfigTest, MaxNumQpsIsValidForAllConfigs) {
  const auto [maxQps, devicesPerRank] = GetParam();

  EnvRAII envMaxQps(NCCL_CTRAN_IB_MAX_QPS, maxQps);
  EnvRAII envDevPerRank(NCCL_CTRAN_IB_DEVICES_PER_RANK, devicesPerRank);

  std::vector<CtranIbDevice> dummyDevices(devicesPerRank, CtranIbDevice{});
  std::vector<int> activeDevices(devicesPerRank);
  std::iota(activeDevices.begin(), activeDevices.end(), 0);
  CtranIbVirtualConn vc(
      dummyDevices,
      /*peerRank=*/0,
      /*comm=*/nullptr,
      /*pgTrafficClass=*/0,
      /*cudaDev=*/0,
      /*activeDevices=*/std::move(activeDevices),
      /*vcsPerPeer=*/1);

  const int maxNumQps = vc.getMaxNumQp();
  EXPECT_GE(maxNumQps, 1) << "maxNumQps must never be zero";
  EXPECT_GE(maxNumQps, devicesPerRank)
      << "maxNumQps must be at least devicesPerRank";
  EXPECT_EQ(maxNumQps % devicesPerRank, 0)
      << "maxNumQps must be an exact multiple of devicesPerRank";
}

INSTANTIATE_TEST_SUITE_P(
    QpRounding,
    CtranIbQpConfigTest,
    ::testing::Values(
        std::make_tuple(1, 1),
        std::make_tuple(16, 1),
        std::make_tuple(1, 2),
        std::make_tuple(2, 2),
        std::make_tuple(3, 2),
        std::make_tuple(5, 2),
        std::make_tuple(1, 3),
        std::make_tuple(4, 3),
        std::make_tuple(1, 4),
        std::make_tuple(3, 4),
        std::make_tuple(5, 4),
        std::make_tuple(128, 1),
        std::make_tuple(128, 2),
        std::make_tuple(129, 3)));

// Exact regression test for the bug case: maxQps=1, devicesPerRank=2
// produced maxNumQps_=0 with the old round-down logic.
TEST(CtranIbQpConfigRegressionTest, MaxQps1DevPerRank2) {
  ncclCvarInit();
  EnvRAII envMaxQps(NCCL_CTRAN_IB_MAX_QPS, 1);
  EnvRAII envDevPerRank(NCCL_CTRAN_IB_DEVICES_PER_RANK, 2);

  std::vector<CtranIbDevice> dummyDevices(2, CtranIbDevice{});
  CtranIbVirtualConn vc(
      dummyDevices,
      /*peerRank=*/0,
      /*comm=*/nullptr,
      /*pgTrafficClass=*/0,
      /*cudaDev=*/0,
      /*activeDevices=*/std::vector<int>{0, 1},
      /*vcsPerPeer=*/1);

  EXPECT_EQ(vc.getMaxNumQp(), 2);
}

TEST(CtranIbDefaultFlushTest, EnablesFlushForOldNvidiaAndGb300) {
  EXPECT_TRUE(CtranIb::shouldEnableLocalFlushByDefault(800));
  EXPECT_FALSE(CtranIb::shouldEnableLocalFlushByDefault(900));
  EXPECT_FALSE(CtranIb::shouldEnableLocalFlushByDefault(1000));
  EXPECT_TRUE(CtranIb::shouldEnableLocalFlushByDefault(1030));
}
