// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdlib>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "comms/ctran/commstate/CommStateX.h"
#include "comms/ctran/tests/CtranXPlatUtUtils.h"

namespace ncclx {

constexpr auto kHost0 = "twshared0100.01.nha1";
constexpr auto kHost1 = "twshared0101.01.nha1";
constexpr auto kHost2 = "twshared0102.01.nha1";
constexpr auto kHost3 = "twshared0103.01.nha1";

constexpr auto kDc = "nha1";
constexpr auto kZone = "c084";

constexpr auto kRtsw0 = "rtsw098.c084.f00.nha1";
constexpr auto kRtsw1 = "rtsw099.c084.f00.nha1";

constexpr auto kSuDomain1 = "nha1.c084.u001";
constexpr auto kSuDomain2 = "nha1.c084.u002";

constexpr auto kNvlFabricClusterId1 = "1";
constexpr auto kNvlFabricClusterId2 = "2";
constexpr int64_t kNvlFabricCliqueId1 = 1;
constexpr int64_t kNvlFabricCliqueId2 = 2;
constexpr int64_t kNvlFabricCliqueId3 = 3;
constexpr int64_t kNvlFabricCliqueId4 = 4;

RankTopology createRankTopology(
    int rank,
    const std::string& dc,
    const std::string& zone,
    const std::string& su,
    const std::string& rtsw,
    const std::string& host,
    int rackSerial = -1) {
  RankTopology topo;
  topo.rank = rank;
  std::strcpy(topo.host, host.c_str());
  std::strcpy(topo.rtsw, rtsw.c_str());
  std::strcpy(topo.su, su.c_str());
  std::strcpy(topo.dc, dc.c_str());
  std::strcpy(topo.zone, zone.c_str());
  topo.rackSerial = rackSerial;
  return topo;
}

NvlFabricTopology createNvlFabricTopology(
    int rank,
    const std::string& clusterId,
    const int64_t cliqueId) {
  NvlFabricTopology topo;
  topo.supportNvlFabric = true;
  topo.rank = rank;
  topo.clusterId = clusterId;
  topo.cliqueId = cliqueId;
  return topo;
}

class CommStateXTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ncclCvarInit();
  }

  void TearDown() override {}
};

TEST(CommStateXTest, Topology) {
  // format dc/zone//rtsw
  std::vector<RankTopology> rankTopologies{};
  const std::string kSu;
  rankTopologies.emplace_back(
      createRankTopology(0, kDc, kZone, kSu, kRtsw0, kHost0));
  rankTopologies.emplace_back(
      createRankTopology(1, kDc, kZone, kSu, kRtsw0, kHost0));
  rankTopologies.emplace_back(
      createRankTopology(2, kDc, kZone, kSu, kRtsw0, kHost1));
  rankTopologies.emplace_back(
      createRankTopology(3, kDc, kZone, kSu, kRtsw0, kHost1));

  rankTopologies.emplace_back(
      createRankTopology(4, kDc, kZone, kSu, kRtsw1, kHost2));
  rankTopologies.emplace_back(
      createRankTopology(5, kDc, kZone, kSu, kRtsw1, kHost2));
  rankTopologies.emplace_back(
      createRankTopology(6, kDc, kZone, kSu, kRtsw1, kHost3));
  rankTopologies.emplace_back(
      createRankTopology(7, kDc, kZone, kSu, kRtsw1, kHost3));

  std::unordered_map<int, std::vector<std::string_view>> expectedHostRtsws{
      {0, {kHost0, kRtsw0, kDc, kZone}},
      {1, {kHost0, kRtsw0, kDc, kZone}},
      {2, {kHost1, kRtsw0, kDc, kZone}},
      {3, {kHost1, kRtsw0, kDc, kZone}},
      {4, {kHost2, kRtsw1, kDc, kZone}},
      {5, {kHost2, kRtsw1, kDc, kZone}},
      {6, {kHost3, kRtsw1, kDc, kZone}},
      {7, {kHost3, kRtsw1, kDc, kZone}}};

  auto verify = [&](CommStateX* s, int rank) {
    EXPECT_EQ(s->cudaDev(), 0);
    EXPECT_EQ(s->cudaArch(), 90);
    EXPECT_EQ(s->busId(), 25);
    EXPECT_EQ(s->commHash(), 0);

    // node APIs
    int expectedNode = rank / 2;
    EXPECT_EQ(s->rank(), rank);
    EXPECT_EQ(s->nRanks(), 8);
    EXPECT_EQ(s->nNodes(), 4);
    EXPECT_EQ(s->node(), expectedNode);
    EXPECT_EQ(s->node(0), 0);
    EXPECT_EQ(s->node(1), 0);
    EXPECT_EQ(s->node(2), 1);
    EXPECT_EQ(s->node(3), 1);
    EXPECT_EQ(s->node(4), 2);
    EXPECT_EQ(s->node(5), 2);
    EXPECT_EQ(s->node(6), 3);
    EXPECT_EQ(s->node(7), 3);

    // localRank APIs
    int expectedLocalRank = rank % 2;
    EXPECT_EQ(s->localRank(), expectedLocalRank);
    EXPECT_EQ(s->localRank(0), 0);
    EXPECT_EQ(s->localRank(1), 1);
    EXPECT_EQ(s->localRank(2), 0);
    EXPECT_EQ(s->localRank(3), 1);
    EXPECT_EQ(s->localRank(4), 0);
    EXPECT_EQ(s->localRank(5), 1);
    EXPECT_EQ(s->localRank(6), 0);
    EXPECT_EQ(s->localRank(7), 1);

    EXPECT_EQ(s->nLocalRanks(), 2);
    EXPECT_EQ(s->nLocalRanks(0), 2);
    EXPECT_EQ(s->nLocalRanks(1), 2);
    EXPECT_EQ(s->nLocalRanks(2), 2);
    EXPECT_EQ(s->nLocalRanks(3), 2);
    EXPECT_EQ(s->nLocalRanks(4), 2);
    EXPECT_EQ(s->nLocalRanks(5), 2);
    EXPECT_EQ(s->nLocalRanks(6), 2);
    EXPECT_EQ(s->nLocalRanks(7), 2);

    int expectedStartRank = rank % 2 == 0 ? rank : (rank - 1);
    EXPECT_EQ(s->localRankToRank(0), expectedStartRank);
    EXPECT_EQ(s->localRankToRank(1), expectedStartRank + 1);

    for (int nodeId = 0; nodeId < s->nNodes(); ++nodeId) {
      for (int i = 0; i < s->nLocalRanks(); ++i) {
        EXPECT_EQ(s->localRankToRank(i, nodeId), nodeId * 2 + i);
      }
    }
    // host/rtsw
    EXPECT_EQ(s->host(), expectedHostRtsws.at(rank).at(0));
    EXPECT_EQ(s->rtsw(), expectedHostRtsws.at(rank).at(1));
    EXPECT_EQ(s->dc(), expectedHostRtsws.at(rank).at(2));
    EXPECT_EQ(s->zone(), expectedHostRtsws.at(rank).at(3));
  };

  // verify all ranks
  for (int rank = 0; rank < 8; ++rank) {
    auto commState = std::make_unique<CommStateX>(
        rank,
        8,
        0,
        90, // H100
        25, // busId
        0,
        rankTopologies,
        std::vector<int>{},
        "");
    verify(commState.get(), rank);
  }
}

TEST(CommStateXTest, ValidEorTopology) {
  std::vector<RankTopology> rankTopologies{};
  const std::string kRtsw;
  rankTopologies.emplace_back(
      createRankTopology(0, kDc, kZone, kSuDomain1, kRtsw, kHost0));
  rankTopologies.emplace_back(
      createRankTopology(1, kDc, kZone, kSuDomain1, kRtsw, kHost0));
  rankTopologies.emplace_back(
      createRankTopology(2, kDc, kZone, kSuDomain1, kRtsw, kHost1));
  rankTopologies.emplace_back(
      createRankTopology(3, kDc, kZone, kSuDomain2, kRtsw, kHost2));

  std::unordered_map<int, std::vector<std::string_view>> expectedTopology{
      {0, {kHost0, kSuDomain1, kDc, kZone}}};

  int myRank = 0;
  auto commState = std::make_unique<CommStateX>(
      myRank,
      4,
      0,
      90, // H100
      25, // busId
      0,
      rankTopologies,
      std::vector<int>{},
      "");

  EXPECT_EQ(commState->cudaDev(), 0);
  EXPECT_EQ(commState->cudaArch(), 90);
  EXPECT_EQ(commState->busId(), 25);
  EXPECT_EQ(commState->commHash(), 0);
  EXPECT_EQ(commState->nRanks(), 4);
  EXPECT_EQ(commState->nNodes(), 3);

  EXPECT_EQ(commState->host(), expectedTopology.at(myRank).at(0));
  EXPECT_EQ(commState->su(), expectedTopology.at(myRank).at(1));
  EXPECT_EQ(commState->dc(), expectedTopology.at(myRank).at(2));
  EXPECT_EQ(commState->zone(), expectedTopology.at(myRank).at(3));

  for (int peer = 1; peer < 4; ++peer) {
    if (peer == 1) {
      EXPECT_TRUE(commState->isSameNode(myRank, peer));
      EXPECT_TRUE(commState->isSameRack(myRank, peer));
    } else if (peer == 2) {
      EXPECT_FALSE(commState->isSameNode(myRank, peer));
      EXPECT_TRUE(commState->isSameRack(myRank, peer));
    } else if (peer == 3) {
      EXPECT_FALSE(commState->isSameRack(myRank, peer));
      EXPECT_TRUE(commState->isSameZone(myRank, peer));
    }
  }
}
TEST(CommStateXTest, multiRackTest) {
  EnvRAII env1(NCCL_CTRAN_NVL_FABRIC_ENABLE, true);
  EnvRAII env2(NCCL_MNNVL_TRUNK_DISABLE, true);
  const int rank = 0;
  const int nRanks = 3;
  const int cudaDev = 0;
  const int cudaArch = 90;
  const int64_t busId = 25;
  const uint64_t commHash = 0;
  const std::string kRtsw;
  std::vector<RankTopology> rankTopologies{};
  rankTopologies.emplace_back(
      createRankTopology(0, kDc, kZone, kSuDomain1, kRtsw, kHost0, 100));
  rankTopologies.emplace_back(
      createRankTopology(1, kDc, kZone, kSuDomain1, kRtsw, kHost0, 100));
  rankTopologies.emplace_back(
      createRankTopology(2, kDc, kZone, kSuDomain1, kRtsw, kHost1, 101));

  auto commState = std::make_unique<CommStateX>(
      rank,
      nRanks,
      cudaDev,
      cudaArch,
      busId,
      commHash,
      rankTopologies,
      std::vector<int>{});

  EXPECT_TRUE(commState->isSameDeviceRack(rank, 1));
  EXPECT_FALSE(commState->isSameDeviceRack(rank, 2));
}

TEST(CommStateXTest, nvlFabricTest) {
  // set NCCL_CTRAN_NVL_FABRIC_ENABLE to true
  EnvRAII env(NCCL_CTRAN_NVL_FABRIC_ENABLE, true);
  const int rank = 0;
  const int nRanks = 8;
  const int cudaDev = 0;
  const int cudaArch = 90;
  const int64_t busId = 25;
  const uint64_t commHash = 0;
  std::vector<NvlFabricTopology> nvlFabricTopologies{};

  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(0, kNvlFabricClusterId1, kNvlFabricCliqueId1));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(1, kNvlFabricClusterId1, kNvlFabricCliqueId1));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(2, kNvlFabricClusterId1, kNvlFabricCliqueId1));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(3, kNvlFabricClusterId1, kNvlFabricCliqueId1));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(4, kNvlFabricClusterId2, kNvlFabricCliqueId1));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(5, kNvlFabricClusterId2, kNvlFabricCliqueId1));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(6, kNvlFabricClusterId2, kNvlFabricCliqueId1));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(7, kNvlFabricClusterId2, kNvlFabricCliqueId1));

  std::vector<RankTopology> rankTopologies{};
  const std::string kSu = "";
  rankTopologies.emplace_back(
      createRankTopology(0, kDc, kZone, kSu, kRtsw0, kHost0));
  rankTopologies.emplace_back(
      createRankTopology(1, kDc, kZone, kSu, kRtsw0, kHost0));
  rankTopologies.emplace_back(
      createRankTopology(2, kDc, kZone, kSu, kRtsw0, kHost1));
  rankTopologies.emplace_back(
      createRankTopology(3, kDc, kZone, kSu, kRtsw0, kHost1));

  rankTopologies.emplace_back(
      createRankTopology(4, kDc, kZone, kSu, kRtsw1, kHost2));
  rankTopologies.emplace_back(
      createRankTopology(5, kDc, kZone, kSu, kRtsw1, kHost2));
  rankTopologies.emplace_back(
      createRankTopology(6, kDc, kZone, kSu, kRtsw1, kHost3));
  rankTopologies.emplace_back(
      createRankTopology(7, kDc, kZone, kSu, kRtsw1, kHost3));

  auto commState = std::make_unique<CommStateX>(
      rank,
      nRanks,
      cudaDev,
      cudaArch,
      busId,
      commHash,
      rankTopologies,
      std::vector<int>{});
  commState->setNvlFabricTopos(nvlFabricTopologies);

  for (int i = 0; i < nRanks; ++i) {
    if (i < 4) {
      EXPECT_TRUE(commState->isSameNvlFabric(0, i));
      EXPECT_EQ(commState->localRank(i), i);
      EXPECT_EQ(commState->nLocalRanks(i), 4);
      EXPECT_EQ(commState->localRankToRank(commState->localRank(i)), i);
      EXPECT_EQ(commState->node(i), 0);
    } else {
      EXPECT_FALSE(commState->isSameNvlFabric(0, i));
      EXPECT_EQ(commState->localRank(i), i - 4);
      EXPECT_EQ(commState->nLocalRanks(i), 4);
      EXPECT_EQ(commState->node(i), 1);
    }
    EXPECT_EQ(commState->nNodes(), 2);
    EXPECT_EQ(commState->localRankToRanks().size(), 4);
  }
  EXPECT_TRUE(commState->nvlFabricEnabled());
  // reset NCCL_CTRAN_NVL_FABRIC_ENABLE to false
  {
    EnvRAII env(NCCL_CTRAN_NVL_FABRIC_ENABLE, false);
    // reload nvlFabricTopologies
    commState->setNvlFabricTopos(std::move(nvlFabricTopologies));
    EXPECT_FALSE(commState->nvlFabricEnabled());
    for (int i = 0; i < nRanks; ++i) {
      EXPECT_FALSE(commState->isSameNvlFabric(0, i));
    }
  }
}

TEST(CommStateXTest, nvlFabricCliqueTest) {
  // enable clique and NVL software partioning mode
  EnvRAII env1(NCCL_CTRAN_NVL_FABRIC_ENABLE, true);
  EnvRAII env2(NCCL_MNNVL_DETERMINISTIC_COLLECTIVE_ENABLE, true);
  EnvRAII env3(NCCL_MNNVL_CLIQUE_SIZE, 2);
  const int rank = 0;
  const int nRanks = 8;
  const int cudaDev = 0;
  const int cudaArch = 90;
  const int64_t busId = 25;
  const uint64_t commHash = 0;
  std::vector<NvlFabricTopology> nvlFabricTopologies{};

  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(0, kNvlFabricClusterId1, kNvlFabricCliqueId1));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(1, kNvlFabricClusterId1, kNvlFabricCliqueId1));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(2, kNvlFabricClusterId1, kNvlFabricCliqueId2));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(3, kNvlFabricClusterId1, kNvlFabricCliqueId2));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(4, kNvlFabricClusterId2, kNvlFabricCliqueId3));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(5, kNvlFabricClusterId2, kNvlFabricCliqueId3));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(6, kNvlFabricClusterId2, kNvlFabricCliqueId4));
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(7, kNvlFabricClusterId2, kNvlFabricCliqueId4));

  std::vector<RankTopology> rankTopologies{};
  const std::string kSu = "";
  rankTopologies.emplace_back(
      createRankTopology(0, kDc, kZone, kSu, kRtsw0, kHost0));
  rankTopologies.emplace_back(
      createRankTopology(1, kDc, kZone, kSu, kRtsw0, kHost0));
  rankTopologies.emplace_back(
      createRankTopology(2, kDc, kZone, kSu, kRtsw0, kHost1));
  rankTopologies.emplace_back(
      createRankTopology(3, kDc, kZone, kSu, kRtsw0, kHost1));

  rankTopologies.emplace_back(
      createRankTopology(4, kDc, kZone, kSu, kRtsw1, kHost2));
  rankTopologies.emplace_back(
      createRankTopology(5, kDc, kZone, kSu, kRtsw1, kHost2));
  rankTopologies.emplace_back(
      createRankTopology(6, kDc, kZone, kSu, kRtsw1, kHost3));
  rankTopologies.emplace_back(
      createRankTopology(7, kDc, kZone, kSu, kRtsw1, kHost3));

  auto commState = std::make_unique<CommStateX>(
      rank,
      nRanks,
      cudaDev,
      cudaArch,
      busId,
      commHash,
      rankTopologies,
      std::vector<int>{});
  commState->setNvlFabricTopos(nvlFabricTopologies);
  EXPECT_TRUE(commState->nvlFabricEnabled());
  EXPECT_TRUE(commState->nvlFabricCliqueEnabled());

  for (int i = 0; i < nRanks; ++i) {
    if (i < 4) {
      EXPECT_TRUE(commState->isSameNvlFabric(0, i));
      if (i < 2) {
        EXPECT_EQ(commState->localRankToRank(commState->localRank(i)), i);
      }
    } else {
      EXPECT_FALSE(commState->isSameNvlFabric(0, i));
    }
    EXPECT_EQ(commState->localRank(i), i % 2);
    EXPECT_EQ(commState->nLocalRanks(i), 2);
    EXPECT_EQ(commState->nNodes(), 4);
    EXPECT_EQ(commState->node(i), i / 2);
    EXPECT_EQ(commState->localRankToRanks().size(), 2);
  }
}

TEST(CommStateXTest, TopologyFailure) {
  const int rank = 0;
  const int nRanks = 8;
  const int cudaDev = 0;
  const int cudaArch = 90;
  const int64_t busId = 25;
  const uint64_t commHash = 0;
  auto commState = std::make_unique<CommStateX>(
      rank,
      nRanks,
      cudaDev,
      cudaArch,
      busId,
      commHash,
      std::vector<RankTopology>{},
      std::vector<int>{});

  // no rank topologies
  EXPECT_DEATH(commState->node(0), "");
}

TEST(CommStateXTest, CommRankToWorldRanks) {
  const int rank = 0;
  const int nRanks = 4;
  const int cudaDev = 0;
  const int cudaArch = 90;
  const int64_t busId = 25;
  const uint64_t commHash = 0;

  auto commState = std::make_unique<CommStateX>(
      rank,
      nRanks,
      cudaDev,
      cudaArch,
      busId,
      commHash,
      std::vector<RankTopology>{},
      std::vector<int>{4, 5, 6, 7});

  EXPECT_EQ(commState->gRank(), 4);
  EXPECT_EQ(commState->gRank(0), 4);
  EXPECT_EQ(commState->gRank(1), 5);
  EXPECT_EQ(commState->gRank(2), 6);
  EXPECT_EQ(commState->gRank(3), 7);
}

TEST(CommStateXTest, TopologySetInvalidNvlFabricTopos) {
  const int rank = 0;
  const int nRanks = 4;
  const int cudaDev = 0;
  const int cudaArch = 90;
  const int64_t busId = 25;
  const uint64_t commHash = 0;

  std::vector<RankTopology> rankTopologies{};
  const std::string kSu;
  rankTopologies.emplace_back(
      createRankTopology(0, kDc, kZone, kSu, kRtsw0, kHost0));
  rankTopologies.emplace_back(
      createRankTopology(1, kDc, kZone, kSu, kRtsw0, kHost0));
  rankTopologies.emplace_back(
      createRankTopology(2, kDc, kZone, kSu, kRtsw0, kHost1));
  rankTopologies.emplace_back(
      createRankTopology(3, kDc, kZone, kSu, kRtsw0, kHost1));

  rankTopologies.emplace_back(
      createRankTopology(4, kDc, kZone, kSu, kRtsw1, kHost2));
  rankTopologies.emplace_back(
      createRankTopology(5, kDc, kZone, kSu, kRtsw1, kHost2));
  rankTopologies.emplace_back(
      createRankTopology(6, kDc, kZone, kSu, kRtsw1, kHost3));
  rankTopologies.emplace_back(
      createRankTopology(7, kDc, kZone, kSu, kRtsw1, kHost3));

  auto commState = std::make_unique<CommStateX>(
      rank,
      nRanks,
      cudaDev,
      cudaArch,
      busId,
      commHash,
      rankTopologies,
      std::vector<int>{});

  std::vector<NvlFabricTopology> nvlFabricTopologies{};
  nvlFabricTopologies.emplace_back(
      createNvlFabricTopology(0, kNvlFabricClusterId1, kNvlFabricCliqueId1));
  // Skip EXPECT_DEATH to check log before abort
  // commState->setNvlFabricTopos(nvlFabricTopologies);
  EXPECT_DEATH(commState->setNvlFabricTopos(nvlFabricTopologies), "");
}

} // namespace ncclx
