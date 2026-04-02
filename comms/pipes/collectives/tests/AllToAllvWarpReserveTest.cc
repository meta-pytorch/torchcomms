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
      .ibgdaRecvWarps = 1,
      .selfWarps = 2,
  };
  int nvlRanks[] = {1, 2, 3};
  int ibgdaRanks[] = {4};
  auto dc = resolveWarpReserve(config, 3, 1, nvlRanks, ibgdaRanks);
  EXPECT_TRUE(dc.isConfigured());

  EXPECT_EQ(dc.selfEnd, 2u);
  EXPECT_EQ(dc.nvlSendEnd, 10u); // 2 + 8
  EXPECT_EQ(dc.nvlRecvEnd, 14u); // 10 + 4
  EXPECT_EQ(dc.ibgdaSendEnd, 16u); // 14 + 2
  // ibgdaRecvEnd = 16 + 1 = 17 (implicit)
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
  EXPECT_EQ(dc.ibgdaSendEnd, 12u); // 11 + 1
  // ibgdaRecvEnd = 12 + 3 = 15 (implicit)
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
  EXPECT_EQ(dc.ibgdaSendEnd, 16u); // 14 + 2
  EXPECT_EQ(dc.numNvlPeers, 3u);
  EXPECT_EQ(dc.numIbgdaPeers, 2u);
  EXPECT_EQ(dc.nvlPeerRanks, nvlRanks);
  EXPECT_EQ(dc.ibgdaPeerRanks, ibgdaRanks);
}

TEST_F(WarpReserveTest, OnlyIbgdaSendExplicit_NvlOnlySetup) {
  // Setting ibgdaSendWarps on an NVL-only setup: ibgda count is 0,
  // so ibgda category is empty
  WarpReserveConfig config{
      .ibgdaSendWarps = 4,
  };
  int nvlRanks[] = {1};
  auto dc = resolveWarpReserve(config, 1, 0, nvlRanks, nullptr);
  EXPECT_TRUE(dc.isConfigured());

  // Auto: self=1, nvlSend=2*1=2, nvlRecv=2*1=2, ibgdaSend=4 (explicit but
  // numIbgdaPeers=0, so no effect on thread count)
  EXPECT_EQ(dc.selfEnd, 1u);
  EXPECT_EQ(dc.nvlSendEnd, 3u); // 1 + 2
  EXPECT_EQ(dc.nvlRecvEnd, 5u); // 3 + 2
  EXPECT_EQ(dc.ibgdaSendEnd, 9u); // 5 + 4
  EXPECT_EQ(dc.numNvlPeers, 1u);
  EXPECT_EQ(dc.numIbgdaPeers, 0u);
}

// --- Default-constructed device config is unconfigured ---

TEST_F(WarpReserveTest, IsConfiguredFalseWhenDefault) {
  WarpReserveDeviceConfig dc{};
  EXPECT_FALSE(dc.isConfigured());
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
  EXPECT_EQ(dc.ibgdaSendEnd, 4u);
  // ibgdaRecvEnd = 4 + 1 = 5 (implicit, total 5 warps)
}

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
