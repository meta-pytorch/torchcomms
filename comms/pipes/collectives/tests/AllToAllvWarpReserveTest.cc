// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include "comms/pipes/collectives/AllToAllv.h"

namespace comms::pipes {

class WarpReserveTest : public ::testing::Test {};

// --- All-zero config returns unconfigured (backward compatible default) ---

TEST_F(WarpReserveTest, AllZeroConfig_NoPeers) {
  WarpReserveConfig config{};
  auto dc = resolveWarpReserve(config, 0, 0, nullptr, nullptr);
  EXPECT_FALSE(dc.isConfigured());
}

TEST_F(WarpReserveTest, AllZeroConfig_NvlOnly_ReturnsUnconfigured) {
  WarpReserveConfig config{};
  int nvlRanks[] = {1, 2, 3};
  auto dc = resolveWarpReserve(config, 3, 0, nvlRanks, nullptr);
  // All-zero config = use uniform partition_interleaved (unconfigured)
  EXPECT_FALSE(dc.isConfigured());
}

TEST_F(WarpReserveTest, AllZeroConfig_IbgdaOnly_ReturnsUnconfigured) {
  WarpReserveConfig config{};
  int ibgdaRanks[] = {4, 5};
  auto dc = resolveWarpReserve(config, 0, 2, nullptr, ibgdaRanks);
  EXPECT_FALSE(dc.isConfigured());
}

TEST_F(WarpReserveTest, AllZeroConfig_HybridNvlIbgda_ReturnsUnconfigured) {
  WarpReserveConfig config{};
  int nvlRanks[] = {1, 2};
  int ibgdaRanks[] = {3, 4, 5};
  auto dc = resolveWarpReserve(config, 2, 3, nvlRanks, ibgdaRanks);
  EXPECT_FALSE(dc.isConfigured());
}

TEST_F(WarpReserveTest, SingleRank_NoPeers) {
  WarpReserveConfig config{};
  auto dc = resolveWarpReserve(config, 0, 0, nullptr, nullptr);
  EXPECT_FALSE(dc.isConfigured());
}

// --- Explicit values produce correct boundaries ---

TEST_F(WarpReserveTest, ExplicitValues) {
  WarpReserveConfig config{
      .nvlSendWarps = 8,
      .nvlRecvWarps = 4,
      .ibgdaSendWarps = 2,
      .ibgdaRecvWarps = 2,
      .selfWarps = 2,
  };
  int nvlRanks[] = {1, 2, 3};
  int ibgdaRanks[] = {4};
  auto dc = resolveWarpReserve(config, 3, 1, nvlRanks, ibgdaRanks);
  EXPECT_TRUE(dc.isConfigured());

  // nvlSend: (8/3)*3=6, nvlRecv: (4/3)*3=3 (rounded to multiple of peers)
  // Block-aligned: nvlRecvEnd=11 → ibgdaSendBase=16 (next 8-warp boundary)
  // with default numThreadsPerBlock=256 → warpsPerBlock=8
  EXPECT_EQ(dc.selfEnd, 2u);
  EXPECT_EQ(dc.nvlSendEnd, 8u); // 2 + 6
  EXPECT_EQ(dc.nvlRecvEnd, 11u); // 8 + 3
  EXPECT_EQ(dc.ibgdaSendBase, 16u); // 11 → round up to 16 (next 8-boundary)
  EXPECT_EQ(dc.ibgdaSendEnd, 18u); // 16 + 2
  EXPECT_EQ(dc.ibgdaRecvBase, 24u); // 18 → round up to 24 (next 8-boundary)
  EXPECT_EQ(dc.ibgdaRecvEnd, 26u); // 24 + 2
}

// --- Mixed auto and explicit: explicit triggers activation, 0 = auto ---

TEST_F(WarpReserveTest, MixedAutoAndExplicit) {
  WarpReserveConfig config{
      .nvlSendWarps = 6, // explicit
      .nvlRecvWarps = 0, // auto = 2*2 = 4
      .ibgdaSendWarps = 0, // auto = 1*1 = 1
      .ibgdaRecvWarps = 3, // explicit
      .selfWarps = 0, // auto = 1
  };
  int nvlRanks[] = {1, 2};
  int ibgdaRanks[] = {3};
  auto dc = resolveWarpReserve(config, 2, 1, nvlRanks, ibgdaRanks);
  EXPECT_TRUE(dc.isConfigured());

  EXPECT_EQ(dc.selfEnd, 1u);
  EXPECT_EQ(dc.nvlSendEnd, 7u); // 1 + 6
  EXPECT_EQ(dc.nvlRecvEnd, 11u); // 7 + 4
  // Block-aligned: nvlRecvEnd=11 → ibgdaSendBase=16 (next 8-boundary)
  EXPECT_EQ(dc.ibgdaSendBase, 16u);
  EXPECT_EQ(dc.ibgdaSendEnd, 17u); // 16 + 1
  // ibgdaRecvEnd = round_up(17,8)=24 + 3 = 27
}

TEST_F(WarpReserveTest, OnlySelfExplicit_AutoComputesPeerWarps) {
  // Setting only selfWarps triggers activation; peer warps auto-compute
  WarpReserveConfig config{
      .selfWarps = 2,
  };
  int nvlRanks[] = {1, 2, 3};
  int ibgdaRanks[] = {4, 5};
  auto dc = resolveWarpReserve(config, 3, 2, nvlRanks, ibgdaRanks);
  EXPECT_TRUE(dc.isConfigured());

  // Auto: nvlSend=2*3=6, nvlRecv=2*3=6, ibgdaSend=1*2=2
  EXPECT_EQ(dc.selfEnd, 2u);
  EXPECT_EQ(dc.nvlSendEnd, 8u); // 2 + 6
  EXPECT_EQ(dc.nvlRecvEnd, 14u); // 8 + 6
  // Block-aligned: nvlRecvEnd=14 → ibgdaSendBase=16 (next 8-boundary)
  EXPECT_EQ(dc.ibgdaSendBase, 16u);
  EXPECT_EQ(dc.ibgdaSendEnd, 18u); // 16 + 2
  EXPECT_EQ(dc.numNvlPeers, 3u);
  EXPECT_EQ(dc.numIbgdaPeers, 2u);
  EXPECT_EQ(dc.nvlPeerRanks, nvlRanks);
  EXPECT_EQ(dc.ibgdaPeerRanks, ibgdaRanks);
}

TEST_F(WarpReserveTest, OnlyIbgdaSendExplicit_NvlOnlySetup) {
  // Setting ibgdaSendWarps on an NVL-only setup: numIbgdaPeers is 0,
  // so ibgda warp counts are clamped to 0 (prevents partition_interleaved(0)
  // division by zero)
  WarpReserveConfig config{
      .ibgdaSendWarps = 4,
  };
  int nvlRanks[] = {1};
  auto dc = resolveWarpReserve(config, 1, 0, nvlRanks, nullptr);
  EXPECT_TRUE(dc.isConfigured());

  // Auto: self=1, nvlSend=2*1=2, nvlRecv=2*1=2
  // ibgdaSend=0 (clamped: numIbgdaPeers=0), ibgdaRecv=0 (clamped)
  EXPECT_EQ(dc.selfEnd, 1u);
  EXPECT_EQ(dc.nvlSendEnd, 3u); // 1 + 2
  EXPECT_EQ(dc.nvlRecvEnd, 5u); // 3 + 2
  EXPECT_EQ(dc.ibgdaSendEnd, 5u); // 5 + 0 (clamped, no block alignment needed)
  EXPECT_EQ(dc.ibgdaSendBase, 5u); // no IBGDA warps → base == nvlRecvEnd
  EXPECT_EQ(dc.numNvlPeers, 1u);
  EXPECT_EQ(dc.numIbgdaPeers, 0u);
}

// --- Default-constructed device config is unconfigured ---

TEST_F(WarpReserveTest, IsConfiguredFalseWhenDefault) {
  WarpReserveDeviceConfig dc{};
  EXPECT_FALSE(dc.isConfigured());
  EXPECT_EQ(dc.maxChannelsPerPeer, 1u);
}

// --- Validation: ibgdaSendWarps == ibgdaRecvWarps when both explicit ---

TEST_F(WarpReserveTest, Validation_RejectsAsymmetricIbgdaWarps) {
  WarpReserveConfig config{
      .ibgdaSendWarps = 4,
      .ibgdaRecvWarps = 2,
  };
  int ibgdaRanks[] = {1, 2};
  EXPECT_THROW(
      resolveWarpReserve(config, 0, 2, nullptr, ibgdaRanks),
      std::runtime_error);
}

TEST_F(WarpReserveTest, Validation_RoundsUnevenIbgdaWarps) {
  // ibgdaSendWarps=3 with 2 peers → rounded to (3/2)*2=2
  // No longer throws — round-down ensures even distribution
  WarpReserveConfig config{
      .ibgdaSendWarps = 3,
      .ibgdaRecvWarps = 3,
  };
  int ibgdaRanks[] = {1, 2};
  auto dc = resolveWarpReserve(config, 0, 2, nullptr, ibgdaRanks);
  EXPECT_TRUE(dc.isConfigured());
  EXPECT_EQ(dc.ibgdaSendEnd, 8u + 2u); // ibgdaSendBase=8, ibgdaSend=(3/2)*2=2
  // With default 256 threads → 8 warps/block:
  // ibgdaSendBase = round_up(1, 8) = 8, ibgdaSendEnd = 8 + 2 = 10
  EXPECT_EQ(dc.ibgdaSendBase, 8u);
}

TEST_F(WarpReserveTest, Validation_AcceptsSymmetricIbgdaWarps) {
  WarpReserveConfig config{
      .ibgdaSendWarps = 4,
      .ibgdaRecvWarps = 4,
      .selfWarps = 1,
  };
  int ibgdaRanks[] = {1, 2};
  auto dc = resolveWarpReserve(config, 0, 2, nullptr, ibgdaRanks);
  EXPECT_TRUE(dc.isConfigured());
  EXPECT_EQ(dc.ibgdaSendEnd, 8u + 4u); // ibgdaSendBase=8 + 4 warps = 12
  // ibgdaSendBase = round_up(1, 8) = 8, ibgdaSendEnd = 8 + 4 = 12
  EXPECT_EQ(dc.ibgdaSendBase, 8u);
}

TEST_F(WarpReserveTest, Validation_SkipsCheckWhenOneSideAuto) {
  // ibgdaSendWarps=0 (auto) → validation only fires when BOTH are explicit
  WarpReserveConfig config{
      .ibgdaSendWarps = 0, // auto
      .ibgdaRecvWarps = 3, // explicit (asymmetric to auto)
      .selfWarps = 1,
  };
  int ibgdaRanks[] = {1};
  // Should NOT throw — only one side is explicit
  auto dc = resolveWarpReserve(config, 0, 1, nullptr, ibgdaRanks);
  EXPECT_TRUE(dc.isConfigured());
}

// --- Minimum config: 1 warp everywhere ---

TEST_F(WarpReserveTest, MinimumConfig_OneWarpEverywhere) {
  WarpReserveConfig config{
      .nvlSendWarps = 1,
      .nvlRecvWarps = 1,
      .ibgdaSendWarps = 1,
      .ibgdaRecvWarps = 1,
      .selfWarps = 1,
  };
  int nvlRanks[] = {1};
  int ibgdaRanks[] = {2};
  auto dc = resolveWarpReserve(config, 1, 1, nvlRanks, ibgdaRanks);
  EXPECT_TRUE(dc.isConfigured());

  EXPECT_EQ(dc.selfEnd, 1u);
  EXPECT_EQ(dc.nvlSendEnd, 2u);
  EXPECT_EQ(dc.nvlRecvEnd, 3u);
  // Block-aligned: nvlRecvEnd=3 → ibgdaSendBase=8 (next 8-boundary)
  EXPECT_EQ(dc.ibgdaSendBase, 8u);
  EXPECT_EQ(dc.ibgdaSendEnd, 9u); // 8 + 1
  EXPECT_EQ(dc.ibgdaRecvBase, 16u); // round_up(9, 8) = 16
  EXPECT_EQ(dc.ibgdaRecvEnd, 17u); // 16 + 1
}

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
