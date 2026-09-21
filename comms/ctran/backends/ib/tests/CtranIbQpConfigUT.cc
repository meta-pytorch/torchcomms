// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/backends/ib/CtranIbVc.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/testinfra/TestXPlatUtils.h"

class CtranIbQpConfigTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {
 protected:
  void SetUp() override {
    ncclCvarInit();
  }
};

TEST_P(CtranIbQpConfigTest, MaxNumQpsIsValidForAllConfigs) {
  constexpr int kBusCardQpCapacity = 128;
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
      /*trafficClass=*/0,
      /*cudaDev=*/0,
      /*activeDevices=*/std::move(activeDevices),
      /*numVcs=*/1);

  const int maxNumQps = vc.getMaxNumQp();
  EXPECT_GE(maxNumQps, 1) << "maxNumQps must never be zero";
  EXPECT_GE(maxNumQps, devicesPerRank)
      << "maxNumQps must be at least devicesPerRank";
  EXPECT_LE(maxNumQps, kBusCardQpCapacity)
      << "maxNumQps must not exceed the business-card capacity";
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
      /*trafficClass=*/0,
      /*cudaDev=*/0,
      /*activeDevices=*/std::vector<int>{0, 1},
      /*numVcs=*/1);

  EXPECT_EQ(vc.getMaxNumQp(), 2);
}

TEST(CtranIbConfigTest, DefaultVcModeFollowsCvar) {
  ncclCvarInit();
  EnvRAII envVcMode(NCCL_CTRAN_IB_VC_MODE, NCCL_CTRAN_IB_VC_MODE::dqplb);

  std::vector<CtranIbDevice> dummyDevices(1, CtranIbDevice{});
  CtranIbVirtualConn vc(
      dummyDevices,
      /*peerRank=*/0,
      /*comm=*/nullptr,
      /*trafficClass=*/0,
      /*cudaDev=*/0,
      /*activeDevices=*/std::vector<int>{0},
      /*numVcs=*/1);

  EXPECT_EQ(vc.getVcMode(), NCCL_CTRAN_IB_VC_MODE::dqplb);
}

TEST(CtranIbQpConfigTest, DividesPerPeerQpsAcrossVcs) {
  ncclCvarInit();
  EnvRAII envMaxQps(NCCL_CTRAN_IB_MAX_QPS, 16);

  std::vector<CtranIbDevice> dummyDevices(1, CtranIbDevice{});
  CtranIbVirtualConn vc(
      dummyDevices,
      /*peerRank=*/0,
      /*comm=*/nullptr,
      /*trafficClass=*/0,
      /*cudaDev=*/0,
      /*activeDevices=*/std::vector<int>{0},
      /*numVcs=*/4);

  EXPECT_EQ(vc.getMaxNumQp(), 4);
}

TEST(CtranIbQpConfigDeathTest, RejectsFewerQpsThanVcs) {
  ncclCvarInit();
  EnvRAII envMaxQps(NCCL_CTRAN_IB_MAX_QPS, 3);
  std::vector<CtranIbDevice> dummyDevices(1, CtranIbDevice{});

  EXPECT_DEATH(
      CtranIbVirtualConn(
          dummyDevices,
          /*peerRank=*/0,
          /*comm=*/nullptr,
          /*trafficClass=*/0,
          /*cudaDev=*/0,
          /*activeDevices=*/std::vector<int>{0},
          /*numVcs=*/4),
      "per-peer MAX_QPS.*must be at least numVcs");
}

TEST(CtranIbQpConfigDeathTest, RejectsUnevenQpDivisionAcrossVcs) {
  ncclCvarInit();
  EnvRAII envMaxQps(NCCL_CTRAN_IB_MAX_QPS, 5);
  std::vector<CtranIbDevice> dummyDevices(1, CtranIbDevice{});

  EXPECT_DEATH(
      CtranIbVirtualConn(
          dummyDevices,
          /*peerRank=*/0,
          /*comm=*/nullptr,
          /*trafficClass=*/0,
          /*cudaDev=*/0,
          /*activeDevices=*/std::vector<int>{0},
          /*numVcs=*/2),
      "per-peer MAX_QPS.*must be a multiple of numVcs");
}

TEST(CtranIbQpConfigTest, AlignmentDoesNotExceedBusCardCapacity) {
  ncclCvarInit();
  EnvRAII envMaxQps(NCCL_CTRAN_IB_MAX_QPS, 129);
  EnvRAII envDevPerRank(NCCL_CTRAN_IB_DEVICES_PER_RANK, 3);

  std::vector<CtranIbDevice> dummyDevices(3, CtranIbDevice{});
  CtranIbVirtualConn vc(
      dummyDevices,
      /*peerRank=*/0,
      /*comm=*/nullptr,
      /*trafficClass=*/0,
      /*cudaDev=*/0,
      /*activeDevices=*/std::vector<int>{0, 1, 2},
      /*numVcs=*/1);

  EXPECT_EQ(vc.getMaxNumQp(), 126);
}

// Without caller configuration, the comm-less path uses the base cvars.
TEST(CtranIbQpConfigTest, BaseCvarsApplyWithoutTopologyOverride) {
  ncclCvarInit();
  EnvRAII envMaxQps(NCCL_CTRAN_IB_MAX_QPS, 8);
  EnvRAII envDevPerRank(NCCL_CTRAN_IB_DEVICES_PER_RANK, 1);
  EnvRAII envScalingTh(NCCL_CTRAN_IB_QP_SCALING_THRESHOLD, uint64_t{1048576});
  EnvRAII envQpMaxMsgs(NCCL_CTRAN_IB_QP_MAX_MSGS, uint64_t{128});
  EnvRAII envVcMode(NCCL_CTRAN_IB_VC_MODE, NCCL_CTRAN_IB_VC_MODE::dqplb);

  std::vector<CtranIbDevice> dummyDevices(1, CtranIbDevice{});
  CtranIbVirtualConn vc(
      dummyDevices,
      /*peerRank=*/0,
      /*comm=*/nullptr,
      /*trafficClass=*/0,
      /*cudaDev=*/0,
      /*activeDevices=*/std::vector<int>{0},
      /*numVcs=*/1);

  EXPECT_EQ(vc.getMaxNumQp(), 8);
  EXPECT_EQ(vc.getQpScalingTh(), 1048576);
  EXPECT_EQ(vc.getMaxQpMsgs(), 128);
  EXPECT_EQ(vc.getVcMode(), NCCL_CTRAN_IB_VC_MODE::dqplb);
}

TEST(CtranIbQpConfigTest, NonVcOverridePreservesConnectionConfig) {
  ncclCvarInit();
  EnvRAII envExConfig(
      NCCL_CTRAN_EX_IB_QP_CONFIG,
      std::vector<std::string>{"262144", "8", "dqplb", "64"});
  const CtranIbConfig ibConfig{.maxNumCqe = 4096};

  std::vector<CtranIbDevice> dummyDevices(1, CtranIbDevice{});
  CtranIbVirtualConn vc(
      dummyDevices,
      /*peerRank=*/0,
      /*comm=*/nullptr,
      /*trafficClass=*/0,
      /*cudaDev=*/0,
      /*activeDevices=*/std::vector<int>{0},
      /*numVcs=*/1,
      ibConfig);

  EXPECT_EQ(vc.getMaxNumQp(), 8);
  EXPECT_EQ(vc.getQpScalingTh(), 262144);
  EXPECT_EQ(vc.getMaxQpMsgs(), 64);
  EXPECT_EQ(vc.getVcMode(), NCCL_CTRAN_IB_VC_MODE::dqplb);
}

TEST(CtranIbQpConfigTest, OverridesOnlySpecifiedVcField) {
  ncclCvarInit();
  EnvRAII envExConfig(
      NCCL_CTRAN_EX_IB_QP_CONFIG,
      std::vector<std::string>{"262144", "8", "dqplb", "64"});
  const CtranIbConfig ibConfig{.numQps = 4};

  std::vector<CtranIbDevice> dummyDevices(1, CtranIbDevice{});
  CtranIbVirtualConn vc(
      dummyDevices,
      /*peerRank=*/0,
      /*comm=*/nullptr,
      /*trafficClass=*/0,
      /*cudaDev=*/0,
      /*activeDevices=*/std::vector<int>{0},
      /*numVcs=*/1,
      ibConfig);

  EXPECT_EQ(vc.getMaxNumQp(), 4);
  EXPECT_EQ(vc.getQpScalingTh(), 262144);
  EXPECT_EQ(vc.getMaxQpMsgs(), 64);
  EXPECT_EQ(vc.getVcMode(), NCCL_CTRAN_IB_VC_MODE::dqplb);
}

TEST(CtranIbQpConfigTest, RejectsNonPositiveQpDepth) {
  ncclCvarInit();
  EnvRAII envMaxQps(NCCL_CTRAN_IB_MAX_QPS, 1);
  EnvRAII envQpMaxMsgs(NCCL_CTRAN_IB_QP_MAX_MSGS, uint64_t{0});

  std::vector<CtranIbDevice> dummyDevices(1, CtranIbDevice{});
  EXPECT_THROW(
      CtranIbVirtualConn(
          dummyDevices,
          /*peerRank=*/0,
          /*comm=*/nullptr,
          /*trafficClass=*/0,
          /*cudaDev=*/0,
          /*activeDevices=*/std::vector<int>{0},
          /*numVcs=*/1),
      ctran::utils::Exception);
}

// Build the physical QP visit order produced by the NIC-interleaved
// round-robin for the first `steps` logical positions.
static std::vector<int>
interleavedOrder(int numActive, int numQpsPerDevice, int steps) {
  std::vector<int> order;
  order.reserve(steps);
  for (int logical = 0; logical < steps; ++logical) {
    order.push_back(
        CtranIbVirtualConn::interleaveQpIdx(
            logical, numActive, numQpsPerDevice));
  }
  return order;
}

// D=2, K=8, full cycle (qps=16): consecutive sub-chunks alternate NIC0/NIC1
// (physical 0,8,1,9,...) instead of filling NIC0's qp0..7 first.
TEST(CtranIbQpInterleaveTest, TwoNicsFullCycle) {
  const std::vector<int> expected = {
      0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
  EXPECT_EQ(
      interleavedOrder(/*numActive=*/2, /*numQpsPerDevice=*/8, /*steps=*/16),
      expected);
}

// D=2, K=8, partial walk (per-op qps=8): the chunk lands 4 QPs on each NIC
// instead of all 8 on NIC0 — this is the qps < maxNumQps_ skew the change
// fixes.
TEST(CtranIbQpInterleaveTest, TwoNicsHalfCycleSpreadsBothNics) {
  const std::vector<int> expected = {0, 8, 1, 9, 2, 10, 3, 11};
  EXPECT_EQ(
      interleavedOrder(/*numActive=*/2, /*numQpsPerDevice=*/8, /*steps=*/8),
      expected);
}

// D=1 (single-NIC VC, e.g. H100 default): interleaving is the identity.
TEST(CtranIbQpInterleaveTest, SingleNicIsIdentity) {
  std::vector<int> expected(16);
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_EQ(
      interleavedOrder(/*numActive=*/1, /*numQpsPerDevice=*/16, /*steps=*/16),
      expected);
}

// Over a full cycle the interleaved order is a permutation of [0, maxQps), so
// every physical QP is visited exactly once and the result stays in-bounds.
TEST(CtranIbQpInterleaveTest, FullCycleIsPermutation) {
  const std::vector<std::pair<int, int>> configs = {
      {2, 8}, {4, 4}, {2, 1}, {3, 5}};
  for (const auto& [numActive, numQpsPerDevice] : configs) {
    const int maxQps = numActive * numQpsPerDevice;
    auto order = interleavedOrder(numActive, numQpsPerDevice, maxQps);
    std::sort(order.begin(), order.end());
    std::vector<int> identity(maxQps);
    std::iota(identity.begin(), identity.end(), 0);
    EXPECT_EQ(order, identity)
        << "numActive=" << numActive << " numQpsPerDevice=" << numQpsPerDevice;
  }
}

// shouldInterleaveQp gates interleaving on enable + >1 NIC + per-WQE size
// (interleave only when wqeSize > minWqeSize).
TEST(CtranIbQpInterleaveTest, ShouldInterleaveQpGate) {
  constexpr uint64_t k64K = 65536;
  constexpr uint64_t k128K = 131072;
  constexpr uint64_t k512K = 524288;

  // Disabled -> never interleave, regardless of size.
  EXPECT_FALSE(
      CtranIbVirtualConn::shouldInterleaveQp(
          /*interleaveDevices=*/false, /*numActive=*/2, k512K, k64K));

  // Single NIC -> nothing to spread across.
  EXPECT_FALSE(
      CtranIbVirtualConn::shouldInterleaveQp(
          /*interleaveDevices=*/true, /*numActive=*/1, k512K, k64K));

  // Enabled, multi-NIC: gate purely on wqeSize > minWqeSize.
  // 64K piece (== threshold) is latency-bound -> skip (AllGather regression).
  EXPECT_FALSE(
      CtranIbVirtualConn::shouldInterleaveQp(
          /*interleaveDevices=*/true, /*numActive=*/2, k64K, k64K));
  // Below threshold -> skip.
  EXPECT_FALSE(
      CtranIbVirtualConn::shouldInterleaveQp(
          /*interleaveDevices=*/true, /*numActive=*/2, k64K / 2, k64K));
  // Above threshold -> interleave (preserves the mid/large-message wins).
  EXPECT_TRUE(
      CtranIbVirtualConn::shouldInterleaveQp(
          /*interleaveDevices=*/true, /*numActive=*/2, k128K, k64K));
  EXPECT_TRUE(
      CtranIbVirtualConn::shouldInterleaveQp(
          /*interleaveDevices=*/true, /*numActive=*/2, k512K, k64K));

  // More than 2 NICs is fine too.
  EXPECT_TRUE(
      CtranIbVirtualConn::shouldInterleaveQp(
          /*interleaveDevices=*/true, /*numActive=*/4, k128K, k64K));
}
