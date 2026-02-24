// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Unit tests for TopologyDiscovery::discover(). Fully mocked — no GPU, CUDA,
// NVML, or specific hardware required. Uses a mock LocalInfoFn to inject
// synthetic RankTopologyInfo and a mock bootstrap for allGather.

#include <cstring>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "comms/pipes/NvmlFabricInfo.h"
#include "comms/pipes/TopologyDiscovery.h"
#include "comms/pipes/tests/MockBootstrap.h"
#include "comms/pipes/tests/TopologyTestUtils.h"

namespace comms::pipes::tests {

using ::testing::_;

namespace {

/// Create a simple mock LocalInfoFn that always returns the given info.
LocalInfoFn make_simple_local_info_fn(const RankTopologyInfo& info) {
  return [info](int /*deviceId*/) -> RankTopologyInfo { return info; };
}

/// Configure mock bootstrap so allGather fills in pre-built data for all
/// ranks except the caller's own slot (which discover() fills in itself).
void expect_prefilled_all_gather(
    testing::MockBootstrap& mock,
    const std::vector<RankTopologyInfo>& allInfo) {
  EXPECT_CALL(mock, allGather(_, _, _, _))
      .WillRepeatedly(
          [allInfo](void* buf, int len, int rank, int nRanks)
              -> folly::SemiFuture<int> {
            auto* charBuf = static_cast<char*>(buf);
            for (int r = 0; r < nRanks; ++r) {
              if (r != rank) {
                std::memcpy(
                    charBuf + r * len,
                    reinterpret_cast<const char*>(&allInfo[r]),
                    len);
              }
            }
            return folly::makeSemiFuture(0);
          });
}

} // namespace

// =============================================================================
// Basic discover() with mocked local info
// =============================================================================

// Verify discover() gathers local info via LocalInfoFn and classifies
// fake same-host peers.
TEST(TopologyDiscoveryTest, DiscoverWithFakeSameHostPeers) {
  constexpr const char* kHostname = "test-host-001";

  // 3 ranks: all on the same host.
  constexpr int nRanks = 3;
  std::vector<RankTopologyInfo> allInfo(nRanks);
  allInfo[0] = make_rank_info(kHostname, 0);
  allInfo[1] = make_rank_info(kHostname, 1);
  allInfo[2] = make_rank_info(kHostname, 2);

  testing::MockBootstrap bootstrap;
  expect_prefilled_all_gather(bootstrap, allInfo);

  PeerAccessFn alwaysAccess = [](int, int) { return true; };
  TopologyDiscovery topo(alwaysAccess, make_simple_local_info_fn(allInfo[0]));
  auto result = topo.discover(/*myRank=*/0, nRanks, /*deviceId=*/0, bootstrap);

  EXPECT_EQ(static_cast<int>(result.nvlPeerRanks.size()), 2);
  EXPECT_EQ(result.globalToNvlLocal.size(), 3u);
  EXPECT_NE(result.globalToNvlLocal.find(0), result.globalToNvlLocal.end());

  // Self should not appear in nvlPeerRanks.
  for (int peer : result.nvlPeerRanks) {
    EXPECT_NE(peer, 0);
  }
}

// Verify discover() classifies a remote peer (different hostname) as non-NVL.
TEST(TopologyDiscoveryTest, DiscoverWithRemotePeer) {
  constexpr const char* kHostname = "test-host-001";

  constexpr int nRanks = 2;
  std::vector<RankTopologyInfo> allInfo(nRanks);
  allInfo[0] = make_rank_info(kHostname, 0);
  allInfo[1] = make_rank_info("remote-host-xyz", 0);

  testing::MockBootstrap bootstrap;
  expect_prefilled_all_gather(bootstrap, allInfo);

  PeerAccessFn alwaysAccess = [](int, int) { return true; };
  TopologyDiscovery topo(alwaysAccess, make_simple_local_info_fn(allInfo[0]));
  auto result = topo.discover(/*myRank=*/0, nRanks, /*deviceId=*/0, bootstrap);

  // Different host, no fabric → remote peer is not NVL.
  EXPECT_TRUE(result.nvlPeerRanks.empty());
  EXPECT_EQ(result.globalToNvlLocal.size(), 1u);
}

// Verify NVL local indices are dense and consistent.
TEST(TopologyDiscoveryTest, NvlLocalIndicesDense) {
  constexpr const char* kHostname = "test-host-001";

  constexpr int nRanks = 4;
  std::vector<RankTopologyInfo> allInfo(nRanks);
  for (int r = 0; r < nRanks; ++r) {
    allInfo[r] = make_rank_info(kHostname, r);
  }

  testing::MockBootstrap bootstrap;
  expect_prefilled_all_gather(bootstrap, allInfo);

  PeerAccessFn alwaysAccess = [](int, int) { return true; };
  TopologyDiscovery topo(alwaysAccess, make_simple_local_info_fn(allInfo[0]));
  auto result = topo.discover(/*myRank=*/0, nRanks, /*deviceId=*/0, bootstrap);

  int nvlNRanks = static_cast<int>(result.globalToNvlLocal.size());
  EXPECT_EQ(nvlNRanks, nRanks);

  std::vector<bool> seen(nvlNRanks, false);
  for (const auto& [gRank, nvlLocal] : result.globalToNvlLocal) {
    ASSERT_GE(nvlLocal, 0);
    ASSERT_LT(nvlLocal, nvlNRanks);
    EXPECT_FALSE(seen[nvlLocal]) << "Duplicate NVL local index " << nvlLocal;
    seen[nvlLocal] = true;
  }

  for (int i = 0; i < nvlNRanks; ++i) {
    EXPECT_TRUE(seen[i]) << "Missing NVL local index " << i;
  }
}

// Single rank: no peers, but self should be in the NVL local mapping.
TEST(TopologyDiscoveryTest, DiscoverSingleRank) {
  constexpr const char* kHostname = "test-host-001";

  constexpr int nRanks = 1;
  auto localInfo = make_rank_info(kHostname, 0);

  std::vector<RankTopologyInfo> allInfo(nRanks);
  allInfo[0] = localInfo;

  testing::MockBootstrap bootstrap;
  expect_prefilled_all_gather(bootstrap, allInfo);

  TopologyDiscovery topo(PeerAccessFn{}, make_simple_local_info_fn(localInfo));
  auto result = topo.discover(/*myRank=*/0, nRanks, /*deviceId=*/0, bootstrap);

  EXPECT_TRUE(result.nvlPeerRanks.empty());
  EXPECT_EQ(result.globalToNvlLocal.size(), 1u);
  EXPECT_EQ(result.globalToNvlLocal.at(0), 0);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
