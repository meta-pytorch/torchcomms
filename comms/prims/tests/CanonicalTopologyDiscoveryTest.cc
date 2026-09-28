// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/common/bootstrap/tests/MockBootstrap.h"
#include "comms/prims/tests/TopologyTestUtils.h"
#include "comms/prims/topology/TopologyDiscovery.h"

namespace comms::prims::tests {
namespace {

using meta::comms::testing::MockBootstrap;
using ::testing::_;
using ::testing::HasSubstr;
using ::testing::InSequence;

CanonicalTopologyPolicyWire makePolicy(const CanonicalTopologyConfig& config) {
  return {
      .mnnvlUuid = config.mnnvlUuid.value_or(0),
      .mnnvlCliqueId = config.mnnvlCliqueId.value_or(0),
      .virtualDomainSize = config.virtualDomainSize,
      .mnnvlMode = static_cast<std::uint8_t>(config.mnnvlMode),
      .hasMnnvlUuid = config.mnnvlUuid.has_value(),
      .hasMnnvlCliqueId = config.mnnvlCliqueId.has_value(),
      .p2pDisable = config.p2pDisable,
      .enableNvlFabricDomains = config.enableNvlFabricDomains,
      .mnnvlTrunkDisable = config.mnnvlTrunkDisable,
      .domainMode = static_cast<std::uint8_t>(config.domainMode),
  };
}

template <std::size_t N>
void setWireString(std::array<char, N>& destination, std::string_view value) {
  ASSERT_LT(value.size(), destination.size());
  std::copy(value.begin(), value.end(), destination.begin());
  destination[value.size()] = '\0';
}

CanonicalRankTopologyInfo makeCanonicalRank(
    int rank,
    std::string_view hostname,
    const CanonicalTopologyConfig& config = {},
    std::string_view rack = {},
    bool fabricInfoAvailable = false,
    bool fabricHandleAvailable = false,
    std::int64_t uuid = 0,
    std::uint32_t cliqueId = 0) {
  CanonicalRankTopologyInfo info;
  info.rank = rank;
  info.cudaDevice = rank;
  info.pid = 1000 + rank;
  info.policy = makePolicy(config);
  setWireString(info.hostname, hostname);
  setWireString(info.deviceRack, rack);
  info.fabricInfoAvailable = fabricInfoAvailable;
  info.fabricHandleAvailable = fabricHandleAvailable;
  if (fabricInfoAvailable) {
    const auto uuidBytes = std::bit_cast<std::array<char, sizeof(uuid)>>(uuid);
    std::copy(uuidBytes.begin(), uuidBytes.end(), info.clusterUuid.begin());
    std::copy(
        uuidBytes.begin(),
        uuidBytes.end(),
        info.clusterUuid.begin() + sizeof(uuid));
    info.cliqueId = cliqueId;
  }
  return info;
}

std::vector<std::uint8_t> identityReachability(int nRanks) {
  const auto rowBytes =
      (static_cast<std::size_t>(nRanks) + 7) / static_cast<std::size_t>(8);
  std::vector<std::uint8_t> reachability(
      static_cast<std::size_t>(nRanks) * rowBytes, 0);
  for (int rank = 0; rank < nRanks; ++rank) {
    const auto offset = static_cast<std::size_t>(rank) * rowBytes +
        static_cast<std::size_t>(rank) / 8;
    reachability[offset] |=
        static_cast<std::uint8_t>(1U << (static_cast<unsigned int>(rank) % 8));
  }
  return reachability;
}

void connect(
    std::vector<std::uint8_t>& reachability,
    int nRanks,
    int first,
    int second) {
  const auto rowBytes =
      (static_cast<std::size_t>(nRanks) + 7) / static_cast<std::size_t>(8);
  const auto set = [&](int from, int to) {
    const auto offset = static_cast<std::size_t>(from) * rowBytes +
        static_cast<std::size_t>(to) / 8;
    reachability[offset] |=
        static_cast<std::uint8_t>(1U << (static_cast<unsigned int>(to) % 8));
  };
  set(first, second);
  set(second, first);
}

[[noreturn]] RankTopologyInfo throwLocalInfoFailure(int) {
  throw std::runtime_error("local-info-failure");
}

[[noreturn]] RankTopologyInfo throwNonStandardLocalInfoFailure(int) {
  throw 7;
}

[[noreturn]] bool throwPeerAccessFailure(int, int) {
  throw std::runtime_error("peer-access-failure");
}

[[noreturn]] bool throwUnexpectedPeerAccess(int, int) {
  throw std::runtime_error("peer access must not be queried");
}

TEST(CanonicalTopologyDiscoveryTest, SupportsSingleRank) {
  constexpr int kNRanks = 1;
  const CanonicalTopologyConfig config;
  TopologyDiscovery discovery;

  const auto result = discovery.classifyCanonical(
      /*myRank=*/0,
      kNRanks,
      {makeCanonicalRank(0, "host-a", config)},
      identityReachability(kNRanks),
      config);

  EXPECT_EQ(result.ranksByDomain, (std::vector<std::vector<int>>{{0}}));
  EXPECT_TRUE(result.mptTopology.nvlPeerRanks.empty());
  EXPECT_EQ(result.mptTopology.globalToNvlLocal.at(0), 0);
}

TEST(CanonicalTopologyDiscoveryTest, SupportsUnevenInterleavedHosts) {
  constexpr int kNRanks = 5;
  const CanonicalTopologyConfig config;
  std::vector<CanonicalRankTopologyInfo> ranks = {
      makeCanonicalRank(0, "host-a", config),
      makeCanonicalRank(1, "host-b", config),
      makeCanonicalRank(2, "host-a", config),
      makeCanonicalRank(3, "host-b", config),
      makeCanonicalRank(4, "host-a", config),
  };
  auto reachability = identityReachability(kNRanks);
  connect(reachability, kNRanks, 0, 2);
  connect(reachability, kNRanks, 0, 4);
  connect(reachability, kNRanks, 2, 4);
  connect(reachability, kNRanks, 1, 3);
  TopologyDiscovery discovery;

  const auto result = discovery.classifyCanonical(
      /*myRank=*/2, kNRanks, std::move(ranks), reachability, config);

  EXPECT_EQ(
      result.ranksByDomain, (std::vector<std::vector<int>>{{0, 2, 4}, {1, 3}}));
  EXPECT_EQ(result.mptTopology.nvlPeerRanks, (std::vector<int>{0, 4}));
  EXPECT_EQ(result.mptTopology.globalToNvlLocal.at(0), 0);
  EXPECT_EQ(result.mptTopology.globalToNvlLocal.at(2), 1);
  EXPECT_EQ(result.mptTopology.globalToNvlLocal.at(4), 2);
}

TEST(CanonicalTopologyDiscoveryTest, UsesUuidAndCliqueAsFabricIdentity) {
  constexpr int kNRanks = 4;
  constexpr std::int64_t kUuidA = 11;
  constexpr std::int64_t kUuidB = 22;
  constexpr std::uint32_t kClique = 7;
  const CanonicalTopologyConfig config;
  std::vector<CanonicalRankTopologyInfo> ranks = {
      makeCanonicalRank(
          0, "host-a", config, "rack-a", true, true, kUuidA, kClique),
      makeCanonicalRank(
          1, "host-b", config, "rack-a", true, true, kUuidA, kClique),
      makeCanonicalRank(
          2, "host-c", config, "rack-b", true, true, kUuidB, kClique),
      makeCanonicalRank(
          3, "host-d", config, "rack-b", true, true, kUuidA, kClique + 1),
  };
  TopologyDiscovery discovery;

  const auto result = discovery.classifyCanonical(
      /*myRank=*/0,
      kNRanks,
      std::move(ranks),
      identityReachability(kNRanks),
      config);

  EXPECT_EQ(
      result.ranksByDomain, (std::vector<std::vector<int>>{{0, 1}, {2}, {3}}));
  EXPECT_EQ(result.mptTopology.nvlPeerRanks, (std::vector<int>{1}));
  EXPECT_TRUE(result.mptTopology.fabricAvailable);
  EXPECT_TRUE(result.fabricActive);
}

TEST(CanonicalTopologyDiscoveryTest, SameHostP2pBridgesDifferentFabricIds) {
  constexpr int kNRanks = 2;
  const CanonicalTopologyConfig config;
  std::vector<CanonicalRankTopologyInfo> ranks = {
      makeCanonicalRank(0, "host-a", config, "rack-a", true, true, 11, 1),
      makeCanonicalRank(1, "host-a", config, "rack-a", true, true, 22, 2),
  };
  auto reachability = identityReachability(kNRanks);
  connect(reachability, kNRanks, 0, 1);
  TopologyDiscovery discovery;

  const auto result = discovery.classifyCanonical(
      /*myRank=*/0, kNRanks, std::move(ranks), reachability, config);

  EXPECT_EQ(result.ranksByDomain, (std::vector<std::vector<int>>{{0, 1}}));
  EXPECT_EQ(result.mptTopology.nvlPeerRanks, (std::vector<int>{1}));
}

TEST(CanonicalTopologyDiscoveryTest, FabricReadinessIsCommunicatorWide) {
  constexpr int kNRanks = 2;
  constexpr std::int64_t kUuid = 17;
  CanonicalTopologyConfig config;
  std::vector<CanonicalRankTopologyInfo> ranks = {
      makeCanonicalRank(0, "host-a", config, "rack-a", true, true, kUuid, 1),
      makeCanonicalRank(1, "host-b", config, "rack-a", true, false, kUuid, 1),
  };
  TopologyDiscovery discovery;

  const auto autoResult = discovery.classifyCanonical(
      /*myRank=*/0, kNRanks, ranks, identityReachability(kNRanks), config);
  EXPECT_EQ(
      autoResult.ranksByDomain, (std::vector<std::vector<int>>{{0}, {1}}));
  EXPECT_FALSE(autoResult.mptTopology.fabricAvailable);

  config.mnnvlMode = MnnvlMode::kEnabled;
  for (auto& rank : ranks) {
    rank.policy = makePolicy(config);
  }
  EXPECT_THROW(
      discovery.classifyCanonical(
          /*myRank=*/0,
          kNRanks,
          std::move(ranks),
          identityReachability(kNRanks),
          config),
      std::runtime_error);
}

TEST(CanonicalTopologyDiscoveryTest, TrunkDisableFailsClosedOnRack) {
  constexpr int kNRanks = 4;
  constexpr std::int64_t kUuid = 31;
  CanonicalTopologyConfig config{.mnnvlTrunkDisable = true};
  std::vector<CanonicalRankTopologyInfo> ranks = {
      makeCanonicalRank(0, "host-a", config, "rack-a", true, true, kUuid, 1),
      makeCanonicalRank(1, "host-b", config, "rack-a", true, true, kUuid, 1),
      makeCanonicalRank(2, "host-c", config, "rack-b", true, true, kUuid, 1),
      makeCanonicalRank(3, "host-d", config, "", true, true, kUuid, 1),
  };
  TopologyDiscovery discovery;

  const auto result = discovery.classifyCanonical(
      /*myRank=*/0,
      kNRanks,
      std::move(ranks),
      identityReachability(kNRanks),
      config);

  EXPECT_EQ(
      result.ranksByDomain, (std::vector<std::vector<int>>{{0, 1}, {2}, {3}}));
  EXPECT_EQ(result.mptTopology.nvlPeerRanks, (std::vector<int>{1}));
}

TEST(CanonicalTopologyDiscoveryTest, AppliesNoLocalAndVirtualModes) {
  constexpr int kNRanks = 4;
  TopologyDiscovery discovery;

  CanonicalTopologyConfig noLocalConfig{
      .domainMode = TopologyDomainMode::kNoLocal};
  std::vector<CanonicalRankTopologyInfo> noLocalRanks;
  noLocalRanks.reserve(kNRanks);
  for (int rank = 0; rank < kNRanks; ++rank) {
    noLocalRanks.push_back(makeCanonicalRank(
        rank,
        "host-a",
        noLocalConfig,
        {},
        /*fabricInfoAvailable=*/true,
        /*fabricHandleAvailable=*/true,
        /*uuid=*/1,
        /*cliqueId=*/static_cast<std::uint32_t>(rank + 1)));
  }
  auto noLocalReachability = identityReachability(kNRanks);
  connect(noLocalReachability, kNRanks, 0, 1);
  connect(noLocalReachability, kNRanks, 1, 2);
  const auto noLocal = discovery.classifyCanonical(
      /*myRank=*/1,
      kNRanks,
      std::move(noLocalRanks),
      noLocalReachability,
      noLocalConfig);
  EXPECT_EQ(
      noLocal.ranksByDomain,
      (std::vector<std::vector<int>>{{0}, {1}, {2}, {3}}));
  EXPECT_TRUE(noLocal.mptTopology.nvlPeerRanks.empty());
  EXPECT_FALSE(noLocal.mptTopology.fabricAvailable);

  CanonicalTopologyConfig virtualConfig{
      .domainMode = TopologyDomainMode::kVirtual, .virtualDomainSize = 2};
  std::vector<CanonicalRankTopologyInfo> virtualRanks;
  virtualRanks.reserve(kNRanks);
  for (int rank = 0; rank < kNRanks; ++rank) {
    virtualRanks.push_back(makeCanonicalRank(rank, "host-a", virtualConfig));
  }
  auto reachability = identityReachability(kNRanks);
  for (int first = 0; first < kNRanks; ++first) {
    for (int second = first + 1; second < kNRanks; ++second) {
      connect(reachability, kNRanks, first, second);
    }
  }
  const auto virtualResult = discovery.classifyCanonical(
      /*myRank=*/2,
      kNRanks,
      std::move(virtualRanks),
      reachability,
      virtualConfig);
  EXPECT_EQ(
      virtualResult.ranksByDomain,
      (std::vector<std::vector<int>>{{0, 1}, {2, 3}}));

  CanonicalTopologyConfig invalidConfig{
      .domainMode = TopologyDomainMode::kVirtual, .virtualDomainSize = 3};
  EXPECT_THROW(
      discovery.classifyCanonical(
          /*myRank=*/0,
          kNRanks,
          {},
          identityReachability(kNRanks),
          invalidConfig),
      std::invalid_argument);
}

TEST(CanonicalTopologyDiscoveryTest, NormalizesOverridesForEveryRank) {
  constexpr int kNRanks = 3;
  constexpr std::int64_t kOverrideUuid = 41;
  constexpr int kOverrideClique = 9;
  const CanonicalTopologyConfig config{
      .mnnvlUuid = kOverrideUuid, .mnnvlCliqueId = kOverrideClique};
  std::vector<CanonicalRankTopologyInfo> ranks = {
      makeCanonicalRank(0, "host-a", config, "rack-a", true, true, 1, 1),
      makeCanonicalRank(1, "host-b", config, "rack-a", true, true, 2, 2),
      makeCanonicalRank(2, "host-c", config, "rack-a", true, true, 3, 3),
  };
  TopologyDiscovery discovery;

  const auto result = discovery.classifyCanonical(
      /*myRank=*/1,
      kNRanks,
      std::move(ranks),
      identityReachability(kNRanks),
      config);

  EXPECT_EQ(result.ranksByDomain, (std::vector<std::vector<int>>{{0, 1, 2}}));
  EXPECT_EQ(result.mptTopology.nvlPeerRanks, (std::vector<int>{0, 2}));
  for (const auto& rank : result.rankInfo) {
    std::int64_t uuidLow = 0;
    std::int64_t uuidHigh = 0;
    std::memcpy(&uuidLow, rank.clusterUuid.data(), sizeof(uuidLow));
    std::memcpy(
        &uuidHigh, rank.clusterUuid.data() + sizeof(uuidLow), sizeof(uuidHigh));
    EXPECT_EQ(uuidLow, kOverrideUuid);
    EXPECT_EQ(uuidHigh, kOverrideUuid);
    EXPECT_EQ(rank.cliqueId, kOverrideClique);
  }
}

TEST(CanonicalTopologyDiscoveryTest, RejectsPolicyAndWireMismatches) {
  constexpr int kNRanks = 2;
  const CanonicalTopologyConfig config;
  const auto reachability = identityReachability(kNRanks);
  TopologyDiscovery discovery;
  const auto makeRanks = [&] {
    return std::vector<CanonicalRankTopologyInfo>{
        makeCanonicalRank(0, "host-a", config),
        makeCanonicalRank(1, "host-b", config),
    };
  };

  auto ranks = makeRanks();
  ++ranks[1].version;
  EXPECT_THROW(
      discovery.classifyCanonical(
          0, kNRanks, std::move(ranks), reachability, config),
      std::runtime_error);

  ranks = makeRanks();
  ranks[1].rank = 0;
  EXPECT_THROW(
      discovery.classifyCanonical(
          0, kNRanks, std::move(ranks), reachability, config),
      std::runtime_error);

  ranks = makeRanks();
  ranks[1].hostname.fill('x');
  EXPECT_THROW(
      discovery.classifyCanonical(
          0, kNRanks, std::move(ranks), reachability, config),
      std::runtime_error);

  ranks = makeRanks();
  ranks[1].policy.p2pDisable = 1;
  EXPECT_THROW(
      discovery.classifyCanonical(
          0, kNRanks, std::move(ranks), reachability, config),
      std::runtime_error);

  ranks = makeRanks();
  ranks[1].fabricHandleAvailable = 1;
  EXPECT_THROW(
      discovery.classifyCanonical(
          0, kNRanks, std::move(ranks), reachability, config),
      std::runtime_error);
}

TEST(CanonicalTopologyDiscoveryTest, RejectsInvalidPeerRelations) {
  constexpr int kNRanks = 3;
  const CanonicalTopologyConfig config;
  std::vector<CanonicalRankTopologyInfo> ranks = {
      makeCanonicalRank(0, "host-a", config),
      makeCanonicalRank(1, "host-a", config),
      makeCanonicalRank(2, "host-a", config),
  };
  TopologyDiscovery discovery;

  auto asymmetric = identityReachability(kNRanks);
  asymmetric[0] |= static_cast<std::uint8_t>(1U << 1);
  EXPECT_THROW(
      discovery.classifyCanonical(0, kNRanks, ranks, asymmetric, config),
      std::runtime_error);

  auto nonTransitive = identityReachability(kNRanks);
  connect(nonTransitive, kNRanks, 0, 1);
  connect(nonTransitive, kNRanks, 1, 2);
  EXPECT_THROW(
      discovery.classifyCanonical(
          0, kNRanks, std::move(ranks), nonTransitive, config),
      std::runtime_error);
}

TEST(CanonicalTopologyDiscoveryTest, PacksRanksAcrossByteBoundaries) {
  constexpr int kNRanks = 9;
  const CanonicalTopologyConfig config;
  std::vector<CanonicalRankTopologyInfo> ranks;
  ranks.reserve(kNRanks);
  for (int rank = 0; rank < kNRanks; ++rank) {
    ranks.push_back(makeCanonicalRank(rank, "host-a", config));
  }
  auto reachability = identityReachability(kNRanks);
  connect(reachability, kNRanks, 0, 8);
  TopologyDiscovery discovery;

  const auto result = discovery.classifyCanonical(
      /*myRank=*/0, kNRanks, ranks, reachability, config);

  EXPECT_EQ(result.ranksByDomain.front(), (std::vector<int>{0, 8}));
  EXPECT_EQ(result.mptTopology.nvlPeerRanks, (std::vector<int>{8}));

  reachability[1] |= 0x80;
  EXPECT_THROW(
      discovery.classifyCanonical(
          /*myRank=*/0, kNRanks, std::move(ranks), reachability, config),
      std::invalid_argument);
}

TEST(CanonicalTopologyDiscoveryTest, DiscoverUsesThreeValidatedGatherPhases) {
  constexpr int kNRanks = 2;
  const CanonicalTopologyConfig config{
      .localPid = 4321,
      .localHostname = "host-a",
      .localZone = "zone-a",
      .localDc = "dc-a",
      .localDeviceRack = "rack-a",
  };
  const auto remote = makeCanonicalRank(1, "host-a", config);
  MockBootstrap bootstrap;
  InSequence sequence;
  EXPECT_CALL(
      bootstrap,
      allGather(
          _, kCanonicalTopologyPreambleSize, /*rank=*/0, /*nranks=*/kNRanks))
      .WillOnce([](void* buffer, int, int, int) {
        auto* preambles = static_cast<CanonicalTopologyPreamble*>(buffer);
        EXPECT_EQ(preambles[0].rank, 0);
        preambles[1].rank = 1;
        return folly::makeSemiFuture(0);
      });
  EXPECT_CALL(
      bootstrap,
      allGather(_, kCanonicalTopologyWireSize, /*rank=*/0, /*nranks=*/kNRanks))
      .WillOnce([remote](void* buffer, int, int, int) {
        auto* ranks = static_cast<CanonicalRankTopologyInfo*>(buffer);
        EXPECT_EQ(ranks[0].rank, 0);
        EXPECT_EQ(ranks[0].cudaDevice, 0);
        EXPECT_EQ(ranks[0].pid, 4321);
        EXPECT_STREQ(ranks[0].hostname.data(), "host-a");
        EXPECT_STREQ(ranks[0].zone.data(), "zone-a");
        EXPECT_STREQ(ranks[0].dc.data(), "dc-a");
        EXPECT_STREQ(ranks[0].deviceRack.data(), "rack-a");
        ranks[1] = remote;
        return folly::makeSemiFuture(0);
      });
  EXPECT_CALL(
      bootstrap,
      allGather(
          _,
          /*status=*/1 + /*packed row=*/1,
          /*rank=*/0,
          /*nranks=*/kNRanks))
      .WillOnce([](void* buffer, int, int, int) {
        auto* rows = static_cast<std::uint8_t*>(buffer);
        EXPECT_EQ(rows[0], 0);
        EXPECT_EQ(rows[1], 0b11);
        rows[2] = 0;
        rows[3] = 0b11;
        return folly::makeSemiFuture(0);
      });
  const auto localInfo = make_rank_info("host-a", /*cudaDevice=*/0);
  TopologyDiscovery discovery(
      [](int, int) { return true; },
      [localInfo](int) { return localInfo; },
      [](int) { return false; });

  const auto result = discovery.discoverCanonical(
      /*myRank=*/0, kNRanks, /*deviceId=*/0, bootstrap, config);

  EXPECT_EQ(result.ranksByDomain, (std::vector<std::vector<int>>{{0, 1}}));
  EXPECT_EQ(result.mptTopology.nvlPeerRanks, (std::vector<int>{1}));
  EXPECT_EQ(result.mptTopology.globalToNvlLocal.at(0), 0);
  EXPECT_EQ(result.mptTopology.globalToNvlLocal.at(1), 1);
}

TEST(CanonicalTopologyDiscoveryTest, PreambleContainsLocalPreparationFailure) {
  constexpr int kNRanks = 2;
  MockBootstrap bootstrap;
  EXPECT_CALL(
      bootstrap,
      allGather(
          _, kCanonicalTopologyPreambleSize, /*rank=*/0, /*nranks=*/kNRanks))
      .WillOnce([](void* buffer, int, int, int) {
        auto* preambles = static_cast<CanonicalTopologyPreamble*>(buffer);
        EXPECT_EQ(preambles[0].status, 1);
        preambles[1].rank = 1;
        return folly::makeSemiFuture(0);
      });
  TopologyDiscovery discovery(
      [](int, int) { return true; },
      throwLocalInfoFailure,
      [](int) { return false; });

  EXPECT_THROW(
      discovery.discoverCanonical(
          /*myRank=*/0,
          kNRanks,
          /*deviceId=*/0,
          bootstrap,
          CanonicalTopologyConfig{}),
      std::runtime_error);
}

TEST(
    CanonicalTopologyDiscoveryTest,
    NonStandardPreparationFailureStillJoinsPreambleGather) {
  constexpr int kNRanks = 2;
  MockBootstrap bootstrap;
  EXPECT_CALL(
      bootstrap,
      allGather(
          _, kCanonicalTopologyPreambleSize, /*rank=*/0, /*nranks=*/kNRanks))
      .WillOnce([](void* buffer, int, int, int) {
        auto* preambles = static_cast<CanonicalTopologyPreamble*>(buffer);
        EXPECT_EQ(preambles[0].status, 1);
        preambles[1].rank = 1;
        return folly::makeSemiFuture(0);
      });
  TopologyDiscovery discovery(
      [](int, int) { return true; },
      throwNonStandardLocalInfoFailure,
      [](int) { return false; });

  EXPECT_THROW(
      discovery.discoverCanonical(
          /*myRank=*/0,
          kNRanks,
          /*deviceId=*/0,
          bootstrap,
          CanonicalTopologyConfig{}),
      std::runtime_error);
}

TEST(CanonicalTopologyDiscoveryTest, PreambleRejectsIncompatibleRecordSize) {
  constexpr int kNRanks = 2;
  MockBootstrap bootstrap;
  EXPECT_CALL(
      bootstrap,
      allGather(
          _, kCanonicalTopologyPreambleSize, /*rank=*/0, /*nranks=*/kNRanks))
      .WillOnce([](void* buffer, int, int, int) {
        auto* preambles = static_cast<CanonicalTopologyPreamble*>(buffer);
        preambles[1].rank = 1;
        ++preambles[1].recordSize;
        return folly::makeSemiFuture(0);
      });
  const auto localInfo = make_rank_info("host-a", /*cudaDevice=*/0);
  TopologyDiscovery discovery(
      [](int, int) { return true; },
      [localInfo](int) { return localInfo; },
      [](int) { return false; });

  EXPECT_THROW(
      discovery.discoverCanonical(
          /*myRank=*/0,
          kNRanks,
          /*deviceId=*/0,
          bootstrap,
          CanonicalTopologyConfig{}),
      std::runtime_error);
}

TEST(CanonicalTopologyDiscoveryTest, PreambleRejectsVersionOneBeforePayload) {
  constexpr int kNRanks = 2;
  MockBootstrap bootstrap;
  EXPECT_CALL(
      bootstrap,
      allGather(
          _, kCanonicalTopologyPreambleSize, /*rank=*/0, /*nranks=*/kNRanks))
      .WillOnce([](void* buffer, int, int, int) {
        auto* preambles = static_cast<CanonicalTopologyPreamble*>(buffer);
        preambles[1] = preambles[0];
        preambles[1].rank = 1;
        preambles[1].version = 1;
        preambles[1].recordSize = 192;
        return folly::makeSemiFuture(0);
      });
  EXPECT_CALL(bootstrap, allGather(_, kCanonicalTopologyWireSize, 0, kNRanks))
      .Times(0);
  const auto localInfo = make_rank_info("host-a", /*cudaDevice=*/0);
  TopologyDiscovery discovery(
      [](int, int) { return true; },
      [localInfo](int) { return localInfo; },
      [](int) { return false; });

  EXPECT_THROW(
      discovery.discoverCanonical(
          /*myRank=*/0,
          kNRanks,
          /*deviceId=*/0,
          bootstrap,
          CanonicalTopologyConfig{}),
      std::runtime_error);
}

TEST(CanonicalTopologyDiscoveryTest, PeerFailureStillJoinsRowGather) {
  constexpr int kNRanks = 2;
  const CanonicalTopologyConfig config;
  const auto remote = makeCanonicalRank(1, "host-a", config);
  MockBootstrap bootstrap;
  InSequence sequence;
  EXPECT_CALL(
      bootstrap, allGather(_, kCanonicalTopologyPreambleSize, 0, kNRanks))
      .WillOnce([](void* buffer, int, int, int) {
        auto* preambles = static_cast<CanonicalTopologyPreamble*>(buffer);
        preambles[1].rank = 1;
        return folly::makeSemiFuture(0);
      });
  EXPECT_CALL(bootstrap, allGather(_, kCanonicalTopologyWireSize, 0, kNRanks))
      .WillOnce([remote](void* buffer, int, int, int) {
        static_cast<CanonicalRankTopologyInfo*>(buffer)[1] = remote;
        return folly::makeSemiFuture(0);
      });
  EXPECT_CALL(bootstrap, allGather(_, 2, 0, kNRanks))
      .WillOnce([](void* buffer, int, int, int) {
        auto* rows = static_cast<std::uint8_t*>(buffer);
        EXPECT_EQ(rows[0], 1);
        rows[2] = 0;
        rows[3] = 0b11;
        return folly::makeSemiFuture(0);
      });
  const auto localInfo = make_rank_info("host-a", /*cudaDevice=*/0);
  TopologyDiscovery discovery(
      throwPeerAccessFailure,
      [localInfo](int) { return localInfo; },
      [](int) { return false; });

  EXPECT_THROW(
      discovery.discoverCanonical(
          /*myRank=*/0, kNRanks, /*deviceId=*/0, bootstrap, config),
      std::runtime_error);
}

TEST(CanonicalTopologyDiscoveryTest, NoLocalSkipsPeerAccessProbe) {
  constexpr int kNRanks = 2;
  const CanonicalTopologyConfig config{
      .domainMode = TopologyDomainMode::kNoLocal};
  const auto remote = makeCanonicalRank(1, "host-a", config);
  MockBootstrap bootstrap;
  InSequence sequence;
  EXPECT_CALL(
      bootstrap, allGather(_, kCanonicalTopologyPreambleSize, 0, kNRanks))
      .WillOnce([](void* buffer, int, int, int) {
        static_cast<CanonicalTopologyPreamble*>(buffer)[1].rank = 1;
        return folly::makeSemiFuture(0);
      });
  EXPECT_CALL(bootstrap, allGather(_, kCanonicalTopologyWireSize, 0, kNRanks))
      .WillOnce([remote](void* buffer, int, int, int) {
        static_cast<CanonicalRankTopologyInfo*>(buffer)[1] = remote;
        return folly::makeSemiFuture(0);
      });
  EXPECT_CALL(bootstrap, allGather(_, 2, 0, kNRanks))
      .WillOnce([](void* buffer, int, int, int) {
        auto* rows = static_cast<std::uint8_t*>(buffer);
        EXPECT_EQ(rows[0], 0);
        EXPECT_EQ(rows[1], 0b01);
        rows[2] = 0;
        rows[3] = 0b10;
        return folly::makeSemiFuture(0);
      });
  const auto localInfo = make_rank_info("host-a", /*cudaDevice=*/0);
  TopologyDiscovery discovery(
      throwUnexpectedPeerAccess,
      [localInfo](int) { return localInfo; },
      [](int) { return false; });

  const auto result = discovery.discoverCanonical(
      /*myRank=*/0, kNRanks, /*deviceId=*/0, bootstrap, config);

  EXPECT_EQ(result.ranksByDomain, (std::vector<std::vector<int>>{{0}, {1}}));
  EXPECT_TRUE(result.mptTopology.nvlPeerRanks.empty());
}

TEST(CanonicalTopologyDiscoveryTest, RejectsTwoRanksOnOneDevice) {
  constexpr int kNRanks = 2;
  const CanonicalTopologyConfig config;
  auto first = makeCanonicalRank(0, "host-a", config);
  auto second = makeCanonicalRank(1, "host-a", config);
  second.cudaDevice = first.cudaDevice;
  auto reachability = identityReachability(kNRanks);
  connect(reachability, kNRanks, 0, 1);
  TopologyDiscovery discovery;

  try {
    discovery.classifyCanonical(
        /*myRank=*/0, kNRanks, {first, second}, reachability, config);
    FAIL() << "expected two ranks on one device to be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_THAT(error.what(), HasSubstr("one rank per GPU"));
    EXPECT_THAT(error.what(), HasSubstr("ranks 0 and 1"));
  }
}

// Without local domains there is no clique to satisfy, so sharing a GPU is
// none of topology discovery's business.
TEST(CanonicalTopologyDiscoveryTest, NoLocalPermitsTwoRanksOnOneDevice) {
  constexpr int kNRanks = 2;
  const CanonicalTopologyConfig config{
      .domainMode = TopologyDomainMode::kNoLocal};
  auto first = makeCanonicalRank(0, "host-a", config);
  auto second = makeCanonicalRank(1, "host-a", config);
  second.cudaDevice = first.cudaDevice;
  TopologyDiscovery discovery;

  const auto result = discovery.classifyCanonical(
      /*myRank=*/0,
      kNRanks,
      {first, second},
      identityReachability(kNRanks),
      config);

  EXPECT_EQ(result.ranksByDomain, (std::vector<std::vector<int>>{{0}, {1}}));
}

} // namespace
} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
