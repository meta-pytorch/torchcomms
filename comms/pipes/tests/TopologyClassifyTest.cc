// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Unit tests for TopologyDiscovery::classify() — the pure-logic core of
// topology discovery. These tests use synthetic RankTopologyInfo data and
// PeerAccessFn lambdas, requiring no CUDA, NVML, MPI, or specific hardware.

#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "comms/pipes/NvmlFabricInfo.h"
#include "comms/pipes/TopologyDiscovery.h"
#include "comms/pipes/tests/TopologyTestUtils.h"

namespace comms::pipes::tests {

namespace {

/// PeerAccessFn that always returns true (all same-host GPUs can peer).
bool always_can_access(int /*deviceA*/, int /*deviceB*/) {
  return true;
}

/// PeerAccessFn that always returns false (no peer access).
bool never_can_access(int /*deviceA*/, int /*deviceB*/) {
  return false;
}

} // namespace

// =============================================================================
// Basic classify() behavior (no MNNVL, same host)
// =============================================================================

// Two ranks on the same host with peer access → both are NVL peers.
TEST(TopologyClassifyTest, SameHostPeerAccess) {
  std::vector<RankTopologyInfo> allInfo = {
      make_rank_info("host0", 0),
      make_rank_info("host0", 1),
  };

  TopologyDiscovery topo(always_can_access);
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo);

  EXPECT_EQ(result.nvlPeerRanks.size(), 1u);
  EXPECT_EQ(result.nvlPeerRanks[0], 1);
  EXPECT_EQ(result.globalToNvlLocal.size(), 2u);
  EXPECT_EQ(result.globalToNvlLocal.at(0), 0);
  EXPECT_EQ(result.globalToNvlLocal.at(1), 1);
  EXPECT_FALSE(result.fabricAvailable);
}

// Two ranks on the same host without peer access → no NVL peers.
TEST(TopologyClassifyTest, SameHostNoPeerAccess) {
  std::vector<RankTopologyInfo> allInfo = {
      make_rank_info("host0", 0),
      make_rank_info("host0", 1),
  };

  TopologyDiscovery topo(never_can_access);
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo);

  EXPECT_TRUE(result.nvlPeerRanks.empty());
  EXPECT_EQ(result.globalToNvlLocal.size(), 1u); // Only self
}

// Two ranks on different hosts without MNNVL → no NVL peers.
TEST(TopologyClassifyTest, DifferentHostsNoMnnvl) {
  std::vector<RankTopologyInfo> allInfo = {
      make_rank_info("host0", 0),
      make_rank_info("host1", 0),
  };

  TopologyDiscovery topo(always_can_access);
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo);

  EXPECT_TRUE(result.nvlPeerRanks.empty());
}

// No peerAccessFn provided → Tier 2 is skipped entirely.
TEST(TopologyClassifyTest, NoPeerAccessFnSkipsTier2) {
  std::vector<RankTopologyInfo> allInfo = {
      make_rank_info("host0", 0),
      make_rank_info("host0", 1),
  };

  TopologyDiscovery topo(PeerAccessFn{});
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo);

  EXPECT_TRUE(result.nvlPeerRanks.empty());
}

// =============================================================================
// Multi-host / large-scale scenarios
// =============================================================================

// 8 ranks on 2 hosts (4 GPUs per host), no MNNVL. Ranks on the same host
// should be NVL peers via Tier 2.
TEST(TopologyClassifyTest, TwoHostsFourGpusEachNoMnnvl) {
  std::vector<RankTopologyInfo> allInfo;
  for (int r = 0; r < 8; ++r) {
    std::string host = (r < 4) ? "host0" : "host1";
    allInfo.push_back(make_rank_info(host.c_str(), r % 4));
  }

  TopologyDiscovery topo(always_can_access);

  // From rank 0's perspective: ranks 1,2,3 are NVL peers (same host).
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/8, allInfo);

  ASSERT_EQ(result.nvlPeerRanks.size(), 3u);
  EXPECT_EQ(result.nvlPeerRanks[0], 1);
  EXPECT_EQ(result.nvlPeerRanks[1], 2);
  EXPECT_EQ(result.nvlPeerRanks[2], 3);

  // From rank 5's perspective: ranks 4,6,7 are NVL peers.
  TopologyDiscovery topo5(always_can_access);
  auto result5 = topo5.classify(/*myRank=*/5, /*nRanks=*/8, allInfo);

  ASSERT_EQ(result5.nvlPeerRanks.size(), 3u);
  EXPECT_EQ(result5.nvlPeerRanks[0], 4);
  EXPECT_EQ(result5.nvlPeerRanks[1], 6);
  EXPECT_EQ(result5.nvlPeerRanks[2], 7);
}

// =============================================================================
// NVL local rank consistency
// =============================================================================

// Verify NVL local indices are dense [0, N) and consistent regardless of
// which rank calls classify().
TEST(TopologyClassifyTest, NvlLocalRanksConsistentAcrossRanks) {
  std::vector<RankTopologyInfo> baseInfo = {
      make_rank_info("host0", 0),
      make_rank_info("host0", 1),
      make_rank_info("host0", 2),
      make_rank_info("host0", 3),
  };

  // All 4 ranks should agree on the global→NVL-local mapping.
  std::unordered_map<int, int> referenceMapping;

  for (int myRank = 0; myRank < 4; ++myRank) {
    // classify modifies allInfo[myRank], so make a fresh copy each time.
    auto allInfo = baseInfo;
    TopologyDiscovery topo(always_can_access);
    auto result = topo.classify(myRank, /*nRanks=*/4, allInfo);

    EXPECT_EQ(result.globalToNvlLocal.size(), 4u);

    if (myRank == 0) {
      referenceMapping = result.globalToNvlLocal;
    } else {
      EXPECT_EQ(result.globalToNvlLocal, referenceMapping)
          << "Rank " << myRank
          << " has different NVL local mapping than rank 0";
    }
  }
}

// Verify NVL local indices form a dense [0, N) range.
TEST(TopologyClassifyTest, NvlLocalIndicesDense) {
  std::vector<RankTopologyInfo> allInfo = {
      make_rank_info("host0", 0),
      make_rank_info("host0", 1),
      make_rank_info("host0", 2),
  };

  TopologyDiscovery topo(always_can_access);
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/3, allInfo);

  int nvlNRanks = static_cast<int>(result.globalToNvlLocal.size());
  ASSERT_EQ(nvlNRanks, 3);

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

// =============================================================================
// Edge cases
// =============================================================================

// Single rank → no peers, self is in NVL local mapping.
TEST(TopologyClassifyTest, SingleRank) {
  std::vector<RankTopologyInfo> allInfo = {
      make_rank_info("host0", 0),
  };

  TopologyDiscovery topo;
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/1, allInfo);

  EXPECT_TRUE(result.nvlPeerRanks.empty());
  EXPECT_EQ(result.globalToNvlLocal.size(), 1u);
  EXPECT_EQ(result.globalToNvlLocal.at(0), 0);
}

} // namespace comms::pipes::tests
