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
// MNNVL Tier 1 classification
// =============================================================================

// Two ranks on different hosts with matching MNNVL fabric → NVL peers via
// Tier 1.
TEST(TopologyClassifyTest, MnnvlTier1CrossHost) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;
  constexpr unsigned int kCliqueId = 1;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuid, kCliqueId),
      make_mnnvl_rank_info("host1", 0, kUuid, kCliqueId),
  };

  TopologyDiscovery topo;
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo);

  EXPECT_EQ(result.nvlPeerRanks.size(), 1u);
  EXPECT_EQ(result.nvlPeerRanks[0], 1);
  EXPECT_TRUE(result.fabricAvailable);
  EXPECT_EQ(result.cliqueId, kCliqueId);
}

// Two ranks with different MNNVL cluster UUIDs → not NVL peers.
TEST(TopologyClassifyTest, MnnvlDifferentUuid) {
  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, 0x1111, 1),
      make_mnnvl_rank_info("host1", 0, 0x2222, 1),
  };

  TopologyDiscovery topo;
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo);

  EXPECT_TRUE(result.nvlPeerRanks.empty());
}

// Two ranks with same UUID but different clique IDs → not NVL peers.
TEST(TopologyClassifyTest, MnnvlDifferentCliqueId) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuid, 1),
      make_mnnvl_rank_info("host1", 0, kUuid, 2),
  };

  TopologyDiscovery topo;
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo);

  EXPECT_TRUE(result.nvlPeerRanks.empty());
}

// One rank has MNNVL, the other does not → Tier 1 does not match. Falls
// through to Tier 2 (same host check).
TEST(TopologyClassifyTest, MnnvlMixedAvailability) {
  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, 0xAAAA, 1),
      make_rank_info("host0", 1), // No MNNVL
  };

  // Same host + peer access → Tier 2 catches it.
  TopologyDiscovery topo(always_can_access);
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo);

  EXPECT_EQ(result.nvlPeerRanks.size(), 1u);
  EXPECT_EQ(result.nvlPeerRanks[0], 1);
}

// =============================================================================
// MnnvlMode tests
// =============================================================================

// MnnvlMode::kDisabled suppresses fabric info even when available. Ranks on
// different hosts lose Tier 1 connectivity.
TEST(TopologyClassifyTest, MnnvlModeDisabledSuppressesFabric) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;
  constexpr unsigned int kCliqueId = 1;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuid, kCliqueId),
      make_mnnvl_rank_info("host1", 0, kUuid, kCliqueId),
  };

  TopologyDiscovery topo;
  TopologyConfig config{.mnnvlMode = MnnvlMode::kDisabled};
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo, config);

  // Tier 1 suppressed, different hosts → no NVL peers.
  EXPECT_TRUE(result.nvlPeerRanks.empty());
  EXPECT_FALSE(result.fabricAvailable);
}

// MnnvlMode::kDisabled still allows same-host Tier 2 peers.
TEST(TopologyClassifyTest, MnnvlModeDisabledAllowsTier2) {
  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, 0xAAAA, 1),
      make_mnnvl_rank_info("host0", 1, 0xAAAA, 1),
  };

  TopologyDiscovery topo(always_can_access);
  TopologyConfig config{.mnnvlMode = MnnvlMode::kDisabled};
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo, config);

  // Same host + peer access → Tier 2 still works.
  EXPECT_EQ(result.nvlPeerRanks.size(), 1u);
  EXPECT_FALSE(result.fabricAvailable);
}

// MnnvlMode::kEnabled throws when fabric info is not available.
TEST(TopologyClassifyTest, MnnvlModeEnabledThrowsWithoutFabric) {
  std::vector<RankTopologyInfo> allInfo = {
      make_rank_info("host0", 0),
  };

  TopologyDiscovery topo;
  TopologyConfig config{.mnnvlMode = MnnvlMode::kEnabled};
  EXPECT_THROW(
      topo.classify(/*myRank=*/0, /*nRanks=*/1, allInfo, config),
      std::runtime_error);
}

// MnnvlMode::kEnabled succeeds when fabric info is available.
TEST(TopologyClassifyTest, MnnvlModeEnabledSucceedsWithFabric) {
  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, 0xAAAA, 1),
  };

  TopologyDiscovery topo;
  TopologyConfig config{.mnnvlMode = MnnvlMode::kEnabled};
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/1, allInfo, config);

  EXPECT_TRUE(result.fabricAvailable);
}

// MnnvlMode::kAuto uses fabric info when available.
TEST(TopologyClassifyTest, MnnvlModeAutoWithFabric) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuid, 1),
      make_mnnvl_rank_info("host1", 0, kUuid, 1),
  };

  TopologyDiscovery topo;
  TopologyConfig config{.mnnvlMode = MnnvlMode::kAuto};
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo, config);

  EXPECT_EQ(result.nvlPeerRanks.size(), 1u);
  EXPECT_TRUE(result.fabricAvailable);
}

// MnnvlMode::kAuto silently falls back when fabric info is unavailable.
TEST(TopologyClassifyTest, MnnvlModeAutoWithoutFabric) {
  std::vector<RankTopologyInfo> allInfo = {
      make_rank_info("host0", 0),
      make_rank_info("host0", 1),
  };

  TopologyDiscovery topo(always_can_access);
  TopologyConfig config{.mnnvlMode = MnnvlMode::kAuto};
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo, config);

  EXPECT_EQ(result.nvlPeerRanks.size(), 1u);
  EXPECT_FALSE(result.fabricAvailable);
}

// MnnvlMode::kDisabled breaks a multi-rank MNNVL domain — cross-host ranks
// lose NVL connectivity, same-host ranks fall back to Tier 2.
TEST(TopologyClassifyTest, MnnvlModeDisabledBreaksMultiRankDomain) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  // 4 ranks across 2 hosts, all same UUID/clique → normally all NVL peers.
  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuid, 1),
      make_mnnvl_rank_info("host0", 1, kUuid, 1),
      make_mnnvl_rank_info("host1", 0, kUuid, 1),
      make_mnnvl_rank_info("host1", 1, kUuid, 1),
  };

  TopologyDiscovery topo(always_can_access);
  TopologyConfig config{.mnnvlMode = MnnvlMode::kDisabled};
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/4, allInfo, config);

  // Cross-host ranks 2,3 lose NVL; same-host rank 1 remains via Tier 2.
  ASSERT_EQ(result.nvlPeerRanks.size(), 1u);
  EXPECT_EQ(result.nvlPeerRanks[0], 1);
  EXPECT_FALSE(result.fabricAvailable);
}

// MnnvlMode::kEnabled with a multi-rank MNNVL domain — all ranks connected.
TEST(TopologyClassifyTest, MnnvlModeEnabledMultiRankDomain) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuid, 1),
      make_mnnvl_rank_info("host0", 1, kUuid, 1),
      make_mnnvl_rank_info("host1", 0, kUuid, 1),
      make_mnnvl_rank_info("host1", 1, kUuid, 1),
  };

  TopologyDiscovery topo;
  TopologyConfig config{.mnnvlMode = MnnvlMode::kEnabled};
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/4, allInfo, config);

  EXPECT_EQ(result.nvlPeerRanks.size(), 3u);
  EXPECT_TRUE(result.fabricAvailable);
}

// MnnvlMode::kEnabled combined with UUID override — fabric validation passes
// first, then the override changes domain membership.
TEST(TopologyClassifyTest, MnnvlModeEnabledWithUuidOverride) {
  constexpr int64_t kUuidA = 0xAAAABBBBCCCCDDDD;
  constexpr int64_t kUuidB = 0x1111222233334444;

  // Rank 0 has UUID_A, ranks 1-3 have UUID_B. Without override, rank 0 is
  // isolated.
  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuidA, 1),
      make_mnnvl_rank_info("host1", 0, kUuidB, 1),
      make_mnnvl_rank_info("host2", 0, kUuidB, 1),
      make_mnnvl_rank_info("host3", 0, kUuidB, 1),
  };

  // kEnabled validates fabric is available (it is), then UUID override
  // changes rank 0's UUID to match ranks 1-3.
  TopologyDiscovery topo(PeerAccessFn{});
  TopologyConfig config{
      .mnnvlMode = MnnvlMode::kEnabled,
      .mnnvlUuid = kUuidB,
  };
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/4, allInfo, config);

  EXPECT_EQ(result.nvlPeerRanks.size(), 3u);
  EXPECT_TRUE(result.fabricAvailable);
}

// MnnvlMode::kEnabled combined with clique ID override.
TEST(TopologyClassifyTest, MnnvlModeEnabledWithCliqueIdOverride) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  // Rank 0 has clique 2, ranks 1-3 have clique 1. Without override, rank 0
  // is isolated.
  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuid, 2),
      make_mnnvl_rank_info("host1", 0, kUuid, 1),
      make_mnnvl_rank_info("host2", 0, kUuid, 1),
      make_mnnvl_rank_info("host3", 0, kUuid, 1),
  };

  TopologyDiscovery topo(PeerAccessFn{});
  TopologyConfig config{
      .mnnvlMode = MnnvlMode::kEnabled,
      .mnnvlCliqueId = 1,
  };
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/4, allInfo, config);

  EXPECT_EQ(result.nvlPeerRanks.size(), 3u);
  EXPECT_TRUE(result.fabricAvailable);
  EXPECT_EQ(result.cliqueId, 1u);
}

// =============================================================================

// UUID override writes the 64-bit value into both halves of the 128-bit UUID.
TEST(TopologyClassifyTest, UuidOverrideBothHalves) {
  constexpr int64_t kOrigUuid = 0x1111222233334444;
  constexpr int64_t kOverrideUuid = 0xDEADBEEFCAFEBABE;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kOrigUuid, 1),
  };

  TopologyDiscovery topo;
  TopologyConfig config{.mnnvlUuid = kOverrideUuid};
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/1, allInfo, config);

  EXPECT_TRUE(result.fabricAvailable);

  int64_t lo = 0;
  int64_t hi = 0;
  std::memcpy(&lo, result.clusterUuid, sizeof(lo));
  std::memcpy(&hi, result.clusterUuid + sizeof(lo), sizeof(hi));
  EXPECT_EQ(lo, kOverrideUuid) << "Lower 64 bits should match override";
  EXPECT_EQ(hi, kOverrideUuid) << "Upper 64 bits should match override";
}

// UUID override unifies ranks that originally had different UUIDs.
TEST(TopologyClassifyTest, UuidOverrideUnifiesRanks) {
  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, 0x1111, 1),
      make_mnnvl_rank_info("host1", 0, 0x2222, 1),
  };

  // Without override: different UUIDs → no NVL peers.
  TopologyDiscovery topo1;
  auto resultBefore = topo1.classify(/*myRank=*/0, /*nRanks=*/2, allInfo);
  EXPECT_TRUE(resultBefore.nvlPeerRanks.empty());

  // Reset allInfo (classify modifies myInfo in place).
  allInfo = {
      make_mnnvl_rank_info("host0", 0, 0x1111, 1),
      make_mnnvl_rank_info("host1", 0, 0x2222, 1),
  };

  // With UUID override on rank 0 to match rank 1's UUID: should become peers.
  TopologyDiscovery topo2;
  TopologyConfig config{.mnnvlUuid = 0x2222};
  auto resultAfter = topo2.classify(
      /*myRank=*/0, /*nRanks=*/2, allInfo, config);

  EXPECT_EQ(resultAfter.nvlPeerRanks.size(), 1u);
}

// UUID override is a no-op when fabric info is unavailable.
TEST(TopologyClassifyTest, UuidOverrideNoopWithoutFabric) {
  std::vector<RankTopologyInfo> allInfo = {
      make_rank_info("host0", 0),
  };

  TopologyDiscovery topo;
  TopologyConfig config{.mnnvlUuid = 0xAAAA};
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/1, allInfo, config);

  EXPECT_FALSE(result.fabricAvailable);
}

// =============================================================================
// Clique ID override tests
// =============================================================================

// Clique ID override is reflected in the result.
TEST(TopologyClassifyTest, CliqueIdOverrideReflected) {
  constexpr int kOverrideCliqueId = 42;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, 0xAAAA, 1),
  };

  TopologyDiscovery topo;
  TopologyConfig config{.mnnvlCliqueId = kOverrideCliqueId};
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/1, allInfo, config);

  EXPECT_TRUE(result.fabricAvailable);
  EXPECT_EQ(result.cliqueId, static_cast<unsigned int>(kOverrideCliqueId));
}

// Different clique ID overrides split MNNVL ranks into separate NVL groups.
TEST(TopologyClassifyTest, CliqueIdOverrideSplitsGroups) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  // 4 ranks, all same UUID, but rank 0 and 2 get clique 10, rank 1 and 3
  // get clique 20. From rank 0's perspective, only rank 2 should be an NVL
  // peer.
  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuid, 10),
      make_mnnvl_rank_info("host1", 0, kUuid, 20),
      make_mnnvl_rank_info("host2", 0, kUuid, 10),
      make_mnnvl_rank_info("host3", 0, kUuid, 20),
  };

  TopologyDiscovery topo;
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/4, allInfo);

  ASSERT_EQ(result.nvlPeerRanks.size(), 1u);
  EXPECT_EQ(result.nvlPeerRanks[0], 2);
}

// Clique ID override is a no-op when fabric info is unavailable.
TEST(TopologyClassifyTest, CliqueIdOverrideNoopWithoutFabric) {
  std::vector<RankTopologyInfo> allInfo = {
      make_rank_info("host0", 0),
  };

  TopologyDiscovery topo;
  TopologyConfig config{.mnnvlCliqueId = 42};
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/1, allInfo, config);

  EXPECT_FALSE(result.fabricAvailable);
  EXPECT_EQ(result.cliqueId, 0u); // Default, not overridden
}

// =============================================================================
// Combined overrides
// =============================================================================

// UUID + clique ID overrides applied together.
TEST(TopologyClassifyTest, UuidAndCliqueIdOverrideCombined) {
  constexpr int64_t kOverrideUuid = 0x1234567890ABCDEF;
  constexpr int kOverrideCliqueId = 7;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, 0x1111, 1),
      make_mnnvl_rank_info("host1", 0, kOverrideUuid, kOverrideCliqueId),
  };

  // Override rank 0's UUID and clique ID to match rank 1.
  TopologyDiscovery topo;
  TopologyConfig config{
      .mnnvlUuid = kOverrideUuid,
      .mnnvlCliqueId = kOverrideCliqueId,
  };
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo, config);

  EXPECT_EQ(result.nvlPeerRanks.size(), 1u);
  EXPECT_TRUE(result.fabricAvailable);
  EXPECT_EQ(result.cliqueId, static_cast<unsigned int>(kOverrideCliqueId));

  int64_t lo = 0;
  std::memcpy(&lo, result.clusterUuid, sizeof(lo));
  EXPECT_EQ(lo, kOverrideUuid);
}

// MnnvlMode::kDisabled prevents UUID/clique overrides from taking effect.
TEST(TopologyClassifyTest, DisabledModePreventsOverrides) {
  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, 0xAAAA, 1),
  };

  TopologyDiscovery topo;
  TopologyConfig config{
      .mnnvlMode = MnnvlMode::kDisabled,
      .mnnvlUuid = 0xFFFF,
      .mnnvlCliqueId = 999,
  };
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/1, allInfo, config);

  EXPECT_FALSE(result.fabricAvailable);
  EXPECT_EQ(result.cliqueId, 0u); // Default, not 999
}

// =============================================================================
// Override changes NVL domain membership
// =============================================================================

// UUID override causes a rank to join an existing multi-rank NVL domain.
// 4 ranks: ranks 1-3 share UUID_A, rank 0 has UUID_B. Override rank 0's UUID
// to UUID_A → rank 0 joins the domain.
TEST(TopologyClassifyTest, UuidOverrideJoinsDomain) {
  constexpr int64_t kUuidA = 0xAAAABBBBCCCCDDDD;
  constexpr int64_t kUuidB = 0x1111222233334444;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuidB, 1), // rank 0: different UUID
      make_mnnvl_rank_info("host1", 0, kUuidA, 1),
      make_mnnvl_rank_info("host2", 0, kUuidA, 1),
      make_mnnvl_rank_info("host3", 0, kUuidA, 1),
  };

  // Without override: rank 0 is isolated (different UUID).
  auto allInfoCopy = allInfo;
  TopologyDiscovery topo1(PeerAccessFn{});
  auto before = topo1.classify(/*myRank=*/0, /*nRanks=*/4, allInfoCopy);
  EXPECT_TRUE(before.nvlPeerRanks.empty());

  // With UUID override to match ranks 1-3: rank 0 joins the domain.
  TopologyDiscovery topo2(PeerAccessFn{});
  TopologyConfig config{.mnnvlUuid = kUuidA};
  auto after = topo2.classify(/*myRank=*/0, /*nRanks=*/4, allInfo, config);

  ASSERT_EQ(after.nvlPeerRanks.size(), 3u);
  EXPECT_EQ(after.nvlPeerRanks[0], 1);
  EXPECT_EQ(after.nvlPeerRanks[1], 2);
  EXPECT_EQ(after.nvlPeerRanks[2], 3);
  EXPECT_TRUE(after.fabricAvailable);
}

// UUID override causes a rank to leave an existing multi-rank NVL domain.
// 4 ranks all share UUID_A. Override rank 0's UUID to UUID_B → rank 0 is
// isolated.
TEST(TopologyClassifyTest, UuidOverrideLeavesDomain) {
  constexpr int64_t kUuidA = 0xAAAABBBBCCCCDDDD;
  constexpr int64_t kUuidB = 0x9999888877776666;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuidA, 1),
      make_mnnvl_rank_info("host1", 0, kUuidA, 1),
      make_mnnvl_rank_info("host2", 0, kUuidA, 1),
      make_mnnvl_rank_info("host3", 0, kUuidA, 1),
  };

  // Without override: all 4 ranks in same domain.
  auto allInfoCopy = allInfo;
  TopologyDiscovery topo1(PeerAccessFn{});
  auto before = topo1.classify(/*myRank=*/0, /*nRanks=*/4, allInfoCopy);
  EXPECT_EQ(before.nvlPeerRanks.size(), 3u);

  // With UUID override: rank 0 leaves the domain.
  TopologyDiscovery topo2(PeerAccessFn{});
  TopologyConfig config{.mnnvlUuid = kUuidB};
  auto after = topo2.classify(/*myRank=*/0, /*nRanks=*/4, allInfo, config);

  EXPECT_TRUE(after.nvlPeerRanks.empty());
  EXPECT_TRUE(after.fabricAvailable);
}

// Clique ID override causes a rank to join an existing multi-rank NVL domain.
// 4 ranks with same UUID: ranks 1-3 have clique 1, rank 0 has clique 2.
// Override rank 0's clique to 1 → rank 0 joins the domain.
TEST(TopologyClassifyTest, CliqueIdOverrideJoinsDomain) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuid, 2), // rank 0: different clique
      make_mnnvl_rank_info("host1", 0, kUuid, 1),
      make_mnnvl_rank_info("host2", 0, kUuid, 1),
      make_mnnvl_rank_info("host3", 0, kUuid, 1),
  };

  // Without override: rank 0 is isolated (different clique).
  auto allInfoCopy = allInfo;
  TopologyDiscovery topo1(PeerAccessFn{});
  auto before = topo1.classify(/*myRank=*/0, /*nRanks=*/4, allInfoCopy);
  EXPECT_TRUE(before.nvlPeerRanks.empty());

  // With clique ID override to match ranks 1-3: rank 0 joins the domain.
  TopologyDiscovery topo2(PeerAccessFn{});
  TopologyConfig config{.mnnvlCliqueId = 1};
  auto after = topo2.classify(/*myRank=*/0, /*nRanks=*/4, allInfo, config);

  ASSERT_EQ(after.nvlPeerRanks.size(), 3u);
  EXPECT_EQ(after.nvlPeerRanks[0], 1);
  EXPECT_EQ(after.nvlPeerRanks[1], 2);
  EXPECT_EQ(after.nvlPeerRanks[2], 3);
  EXPECT_TRUE(after.fabricAvailable);
}

// Clique ID override causes a rank to leave an existing multi-rank NVL domain.
// 4 ranks all same UUID and clique 1. Override rank 0's clique to 99 → rank 0
// is isolated.
TEST(TopologyClassifyTest, CliqueIdOverrideLeavesDomain) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuid, 1),
      make_mnnvl_rank_info("host1", 0, kUuid, 1),
      make_mnnvl_rank_info("host2", 0, kUuid, 1),
      make_mnnvl_rank_info("host3", 0, kUuid, 1),
  };

  // Without override: all 4 ranks in same domain.
  auto allInfoCopy = allInfo;
  TopologyDiscovery topo1(PeerAccessFn{});
  auto before = topo1.classify(/*myRank=*/0, /*nRanks=*/4, allInfoCopy);
  EXPECT_EQ(before.nvlPeerRanks.size(), 3u);

  // With clique ID override: rank 0 leaves the domain.
  TopologyDiscovery topo2(PeerAccessFn{});
  TopologyConfig config{.mnnvlCliqueId = 99};
  auto after = topo2.classify(/*myRank=*/0, /*nRanks=*/4, allInfo, config);

  EXPECT_TRUE(after.nvlPeerRanks.empty());
  EXPECT_TRUE(after.fabricAvailable);
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

// 8 ranks on 2 hosts with MNNVL, all in the same clique → all 8 are NVL
// peers.
TEST(TopologyClassifyTest, TwoHostsMnnvlAllSameClique) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  std::vector<RankTopologyInfo> allInfo;
  for (int r = 0; r < 8; ++r) {
    std::string host = (r < 4) ? "host0" : "host1";
    allInfo.push_back(make_mnnvl_rank_info(host.c_str(), r % 4, kUuid, 1));
  }

  TopologyDiscovery topo;
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/8, allInfo);

  EXPECT_EQ(result.nvlPeerRanks.size(), 7u);
  EXPECT_TRUE(result.fabricAvailable);
}

// 16 ranks on 4 hosts, MNNVL with 2 cliques: hosts 0,1 in clique 1,
// hosts 2,3 in clique 2. Each clique forms a separate NVL domain.
TEST(TopologyClassifyTest, FourHostsTwoCliques) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  std::vector<RankTopologyInfo> allInfo;
  for (int r = 0; r < 16; ++r) {
    int hostIdx = r / 4;
    std::string host = "host" + std::to_string(hostIdx);
    unsigned int cliqueId = (hostIdx < 2) ? 1 : 2;
    allInfo.push_back(
        make_mnnvl_rank_info(host.c_str(), r % 4, kUuid, cliqueId));
  }

  // Rank 0 (host0, clique 1) → NVL peers are ranks 1..7 (hosts 0,1).
  TopologyDiscovery topo0;
  auto result0 = topo0.classify(/*myRank=*/0, /*nRanks=*/16, allInfo);

  EXPECT_EQ(result0.nvlPeerRanks.size(), 7u);
  for (int peer : result0.nvlPeerRanks) {
    EXPECT_LT(peer, 8) << "Clique 1 rank should only see ranks 0-7 as NVL";
  }

  // Rank 8 (host2, clique 2) → NVL peers are ranks 9..15 (hosts 2,3).
  TopologyDiscovery topo8;
  auto result8 = topo8.classify(/*myRank=*/8, /*nRanks=*/16, allInfo);

  EXPECT_EQ(result8.nvlPeerRanks.size(), 7u);
  for (int peer : result8.nvlPeerRanks) {
    EXPECT_GE(peer, 8) << "Clique 2 rank should only see ranks 8-15 as NVL";
    EXPECT_LE(peer, 15);
  }
}

// 64 ranks across 8 hosts (8 GPUs per host), MNNVL with per-host clique IDs.
// Simulates a scenario where each host is its own MNNVL clique.
TEST(TopologyClassifyTest, LargeScale64RanksPerHostCliques) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;
  constexpr int kRanksPerHost = 8;
  constexpr int kNumHosts = 8;
  constexpr int kTotalRanks = kRanksPerHost * kNumHosts;

  std::vector<RankTopologyInfo> allInfo;
  for (int r = 0; r < kTotalRanks; ++r) {
    int hostIdx = r / kRanksPerHost;
    std::string host = "host" + std::to_string(hostIdx);
    // Each host gets its own clique ID.
    allInfo.push_back(
        make_mnnvl_rank_info(host.c_str(), r % kRanksPerHost, kUuid, hostIdx));
  }

  // From rank 0's perspective: only ranks 1..7 (same host/clique) are NVL.
  TopologyDiscovery topo;
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/kTotalRanks, allInfo);

  EXPECT_EQ(result.nvlPeerRanks.size(), static_cast<size_t>(kRanksPerHost - 1));
  for (int peer : result.nvlPeerRanks) {
    EXPECT_LT(peer, kRanksPerHost);
  }

  // Verify from a middle rank (rank 24, host 3).
  TopologyDiscovery topo24;
  auto result24 = topo24.classify(
      /*myRank=*/24, /*nRanks=*/kTotalRanks, allInfo);

  EXPECT_EQ(
      result24.nvlPeerRanks.size(), static_cast<size_t>(kRanksPerHost - 1));
  for (int peer : result24.nvlPeerRanks) {
    EXPECT_GE(peer, 24);
    EXPECT_LT(peer, 32);
  }
}

// 64 ranks across 8 hosts, MNNVL with UUID override isolates rank 0.
TEST(TopologyClassifyTest, LargeScale64RanksUuidOverrideIsolatesRank) {
  constexpr int64_t kOrigUuid = 0xAAAABBBBCCCCDDDD;
  constexpr int64_t kOverrideUuid = 0x9999888877776666;
  constexpr int kRanksPerHost = 8;
  constexpr int kNumHosts = 8;
  constexpr int kTotalRanks = kRanksPerHost * kNumHosts;

  std::vector<RankTopologyInfo> allInfo;
  for (int r = 0; r < kTotalRanks; ++r) {
    int hostIdx = r / kRanksPerHost;
    std::string host = "host" + std::to_string(hostIdx);
    // All ranks same UUID and same clique 0.
    allInfo.push_back(
        make_mnnvl_rank_info(host.c_str(), r % kRanksPerHost, kOrigUuid, 0));
  }

  // Override UUID on rank 0 to a different value — rank 0 should now be
  // isolated since no other rank has this UUID.
  // Use null PeerAccessFn to disable Tier 2 fallback (this test is purely
  // testing MNNVL UUID override behavior, not peer access).
  TopologyDiscovery topo(PeerAccessFn{});
  TopologyConfig config{.mnnvlUuid = kOverrideUuid};
  auto result = topo.classify(
      /*myRank=*/0, /*nRanks=*/kTotalRanks, allInfo, config);

  // Rank 0's UUID is now different from everyone else → no NVL peers.
  EXPECT_TRUE(result.nvlPeerRanks.empty());
  EXPECT_TRUE(result.fabricAvailable);
}

// =============================================================================
// NVL local rank consistency
// =============================================================================

// Verify NVL local indices are dense [0, N) and consistent regardless of
// which rank calls classify().
TEST(TopologyClassifyTest, NvlLocalRanksConsistentAcrossRanks) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  std::vector<RankTopologyInfo> baseInfo = {
      make_mnnvl_rank_info("host0", 0, kUuid, 1),
      make_mnnvl_rank_info("host0", 1, kUuid, 1),
      make_mnnvl_rank_info("host1", 0, kUuid, 1),
      make_mnnvl_rank_info("host1", 1, kUuid, 1),
  };

  // All 4 ranks should agree on the global→NVL-local mapping.
  std::unordered_map<int, int> referenceMapping;

  for (int myRank = 0; myRank < 4; ++myRank) {
    // classify modifies allInfo[myRank], so make a fresh copy each time.
    auto allInfo = baseInfo;
    TopologyDiscovery topo;
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
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuid, 1),
      make_mnnvl_rank_info("host0", 1, kUuid, 1),
      make_mnnvl_rank_info("host1", 0, kUuid, 1),
  };

  TopologyDiscovery topo;
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

// Single MNNVL rank → no peers, fabric info preserved.
TEST(TopologyClassifyTest, SingleMnnvlRank) {
  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, 0xAAAA, 1),
  };

  TopologyDiscovery topo;
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/1, allInfo);

  EXPECT_TRUE(result.nvlPeerRanks.empty());
  EXPECT_TRUE(result.fabricAvailable);
  EXPECT_EQ(result.cliqueId, 1u);
}

// Tier 1 takes priority over Tier 2: if MNNVL matches, we don't need
// peer access check. Verify by providing never_can_access — MNNVL peers
// should still be found.
TEST(TopologyClassifyTest, Tier1TakesPriorityOverTier2) {
  constexpr int64_t kUuid = 0xAAAABBBBCCCCDDDD;

  std::vector<RankTopologyInfo> allInfo = {
      make_mnnvl_rank_info("host0", 0, kUuid, 1),
      make_mnnvl_rank_info("host0", 1, kUuid, 1),
  };

  // Tier 1 matches → peer found even though peerAccessFn returns false.
  TopologyDiscovery topo(never_can_access);
  auto result = topo.classify(/*myRank=*/0, /*nRanks=*/2, allInfo);

  EXPECT_EQ(result.nvlPeerRanks.size(), 1u);
  EXPECT_EQ(result.nvlPeerRanks[0], 1);
}

} // namespace comms::pipes::tests
