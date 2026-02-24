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

// =============================================================================
// MnnvlMode tests
// =============================================================================

// MnnvlMode::kDisabled suppresses fabric info, peers found via Tier 2.
TEST(TopologyDiscoveryTest, MnnvlModeDisabled) {
  constexpr const char* kHostname = "test-host-001";

  constexpr int nRanks = 2;
  std::vector<RankTopologyInfo> allInfo(nRanks);
  allInfo[0] = make_rank_info(kHostname, 0);
  allInfo[1] = make_rank_info(kHostname, 1);

  testing::MockBootstrap bootstrap;
  expect_prefilled_all_gather(bootstrap, allInfo);

  PeerAccessFn alwaysAccess = [](int, int) { return true; };
  TopologyDiscovery topo(alwaysAccess, make_simple_local_info_fn(allInfo[0]));
  TopologyConfig config{.mnnvlMode = MnnvlMode::kDisabled};
  auto result = topo.discover(
      /*myRank=*/0, nRanks, /*deviceId=*/0, bootstrap, config);

  EXPECT_FALSE(result.fabricAvailable);
  // Same-host peers still detected via Tier 2.
  EXPECT_EQ(static_cast<int>(result.nvlPeerRanks.size()), 1);
}

// MnnvlMode::kEnabled throws when fabric info is not available.
TEST(TopologyDiscoveryTest, MnnvlModeEnabledThrowsOnNonMnnvl) {
  constexpr const char* kHostname = "test-host-001";

  constexpr int nRanks = 1;
  // Fabric info NOT available.
  auto localInfo = make_rank_info(kHostname, 0);

  std::vector<RankTopologyInfo> allInfo(nRanks);
  allInfo[0] = localInfo;

  testing::MockBootstrap bootstrap;
  expect_prefilled_all_gather(bootstrap, allInfo);

  TopologyDiscovery topo(PeerAccessFn{}, make_simple_local_info_fn(localInfo));
  TopologyConfig config{.mnnvlMode = MnnvlMode::kEnabled};
  EXPECT_THROW(
      topo.discover(/*myRank=*/0, nRanks, /*deviceId=*/0, bootstrap, config),
      std::runtime_error);
}

// MnnvlMode::kEnabled succeeds when fabric info is available (mocked).
TEST(TopologyDiscoveryTest, MnnvlModeEnabledSucceedsOnMnnvl) {
  constexpr const char* kHostname = "test-host-001";
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  constexpr int nRanks = 1;
  auto localInfo = make_mnnvl_rank_info(kHostname, 0, kUuid, 1);

  std::vector<RankTopologyInfo> allInfo(nRanks);
  allInfo[0] = localInfo;

  testing::MockBootstrap bootstrap;
  expect_prefilled_all_gather(bootstrap, allInfo);

  TopologyDiscovery topo(PeerAccessFn{}, make_simple_local_info_fn(localInfo));
  TopologyConfig config{.mnnvlMode = MnnvlMode::kEnabled};
  auto result =
      topo.discover(/*myRank=*/0, nRanks, /*deviceId=*/0, bootstrap, config);

  EXPECT_TRUE(result.fabricAvailable);
}

// MnnvlMode::kAuto produces the same result as default config.
TEST(TopologyDiscoveryTest, MnnvlModeAutoMatchesDefault) {
  constexpr const char* kHostname = "test-host-001";

  constexpr int nRanks = 2;
  std::vector<RankTopologyInfo> allInfo(nRanks);
  allInfo[0] = make_rank_info(kHostname, 0);
  allInfo[1] = make_rank_info(kHostname, 1);

  PeerAccessFn alwaysAccess = [](int, int) { return true; };
  auto localInfoFn = make_simple_local_info_fn(allInfo[0]);

  testing::MockBootstrap bootstrap1;
  expect_prefilled_all_gather(bootstrap1, allInfo);
  TopologyDiscovery topo1(alwaysAccess, localInfoFn);
  auto defaultResult =
      topo1.discover(/*myRank=*/0, nRanks, /*deviceId=*/0, bootstrap1);

  testing::MockBootstrap bootstrap2;
  expect_prefilled_all_gather(bootstrap2, allInfo);
  TopologyDiscovery topo2(alwaysAccess, localInfoFn);
  TopologyConfig config{.mnnvlMode = MnnvlMode::kAuto};
  auto autoResult =
      topo2.discover(/*myRank=*/0, nRanks, /*deviceId=*/0, bootstrap2, config);

  EXPECT_EQ(defaultResult.nvlPeerRanks, autoResult.nvlPeerRanks);
  EXPECT_EQ(defaultResult.fabricAvailable, autoResult.fabricAvailable);
}

// =============================================================================
// MNNVL override tests
// =============================================================================

// Verify clique ID override is reflected with mocked MNNVL fabric info.
TEST(TopologyDiscoveryTest, CliqueIdOverride) {
  constexpr const char* kHostname = "test-host-001";
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  constexpr int nRanks = 1;
  constexpr int kOverrideCliqueId = 42;
  auto localInfo = make_mnnvl_rank_info(kHostname, 0, kUuid, 1);

  std::vector<RankTopologyInfo> allInfo(nRanks);
  allInfo[0] = localInfo;

  testing::MockBootstrap bootstrap;
  expect_prefilled_all_gather(bootstrap, allInfo);

  TopologyDiscovery topo(PeerAccessFn{}, make_simple_local_info_fn(localInfo));
  TopologyConfig config{.mnnvlCliqueId = kOverrideCliqueId};
  auto result =
      topo.discover(/*myRank=*/0, nRanks, /*deviceId=*/0, bootstrap, config);

  EXPECT_TRUE(result.fabricAvailable);
  EXPECT_EQ(result.cliqueId, static_cast<unsigned int>(kOverrideCliqueId));
}

// Verify UUID override is reflected with mocked MNNVL fabric info.
TEST(TopologyDiscoveryTest, UuidOverride) {
  constexpr const char* kHostname = "test-host-001";
  constexpr int64_t kOrigUuid = 0xAAAABBBBCCCCDDDD;

  constexpr int nRanks = 1;
  constexpr int64_t kOverrideUuid = 0xDEADBEEFCAFEBABE;
  auto localInfo = make_mnnvl_rank_info(kHostname, 0, kOrigUuid, 1);

  std::vector<RankTopologyInfo> allInfo(nRanks);
  allInfo[0] = localInfo;

  testing::MockBootstrap bootstrap;
  expect_prefilled_all_gather(bootstrap, allInfo);

  TopologyDiscovery topo(PeerAccessFn{}, make_simple_local_info_fn(localInfo));
  TopologyConfig config{.mnnvlUuid = kOverrideUuid};
  auto result =
      topo.discover(/*myRank=*/0, nRanks, /*deviceId=*/0, bootstrap, config);

  EXPECT_TRUE(result.fabricAvailable);
  int64_t lo = 0;
  int64_t hi = 0;
  std::memcpy(&lo, result.clusterUuid, sizeof(lo));
  std::memcpy(&hi, result.clusterUuid + sizeof(lo), sizeof(hi));
  EXPECT_EQ(lo, kOverrideUuid);
  EXPECT_EQ(hi, kOverrideUuid);
}

// Verify kDisabled prevents overrides from taking effect.
TEST(TopologyDiscoveryTest, DisabledModePreventsOverrides) {
  constexpr const char* kHostname = "test-host-001";

  constexpr int nRanks = 1;
  auto localInfo = make_rank_info(kHostname, 0);

  std::vector<RankTopologyInfo> allInfo(nRanks);
  allInfo[0] = localInfo;

  testing::MockBootstrap bootstrap;
  expect_prefilled_all_gather(bootstrap, allInfo);

  TopologyDiscovery topo(PeerAccessFn{}, make_simple_local_info_fn(localInfo));
  TopologyConfig config{
      .mnnvlMode = MnnvlMode::kDisabled,
      .mnnvlUuid = 0xFFFF,
      .mnnvlCliqueId = 999,
  };
  auto result =
      topo.discover(/*myRank=*/0, nRanks, /*deviceId=*/0, bootstrap, config);

  EXPECT_FALSE(result.fabricAvailable);
  EXPECT_EQ(result.cliqueId, 0u);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
