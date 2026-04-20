// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "comms/ctran/commstate/CommStateX.h"
#include "comms/testinfra/TestXPlatUtils.h"

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
    int rackSerial = -1,
    int pid = -1) {
  RankTopology topo;
  topo.rank = rank;
  topo.pid = pid;
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
  EnvRAII env(NCCL_MNNVL_TRUNK_DISABLE, true);
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
  commState->setNvlFabricTopos(nvlFabricTopologies, true);

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
  // Test with fabricHwSupported = false
  {
    // reload nvlFabricTopologies with fabric HW not supported
    commState->setNvlFabricTopos(std::move(nvlFabricTopologies), false);
    EXPECT_FALSE(commState->nvlFabricEnabled());
    for (int i = 0; i < nRanks; ++i) {
      EXPECT_FALSE(commState->isSameNvlFabric(0, i));
    }
  }
}

TEST(CommStateXTest, nvlFabricCliqueTest) {
  // enable clique and NVL software partioning mode
  EnvRAII env1(NCCL_MNNVL_DETERMINISTIC_COLLECTIVE_ENABLE, true);
  EnvRAII env2(NCCL_MNNVL_CLIQUE_SIZE, 2);
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
  commState->setNvlFabricTopos(nvlFabricTopologies, true);
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

TEST(CommStateXTest, nvlFabricVCliqueSizeHint) {
  const int rank = 0;
  const int nRanks = 8;
  const int cudaDev = 0;
  const int cudaArch = 90;
  const int64_t busId = 25;
  const uint64_t commHash = 0;

  std::vector<NvlFabricTopology> nvlFabricTopologies{};
  for (int i = 0; i < nRanks; ++i) {
    nvlFabricTopologies.emplace_back(
        createNvlFabricTopology(i, kNvlFabricClusterId1, 0));
  }

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

  auto makeCommState = [&](int vCliqueSize = 0) {
    return std::make_unique<CommStateX>(
        rank,
        nRanks,
        cudaDev,
        cudaArch,
        busId,
        commHash,
        rankTopologies,
        std::vector<int>{},
        "" /* commDesc */,
        false /* noLocal */,
        vCliqueSize);
  };

  // vCliqueSize=4 partitions 8 ranks into 2 virtual domains of 4
  {
    auto cs = makeCommState(4);
    cs->setNvlFabricTopos(nvlFabricTopologies, true);
    EXPECT_TRUE(cs->nvlFabricCliqueEnabled());
    EXPECT_EQ(cs->nNodes(), 2);
    for (int i = 0; i < nRanks; ++i) {
      EXPECT_EQ(cs->localRank(i), i % 4);
      EXPECT_EQ(cs->nLocalRanks(i), 4);
    }
  }

  // vCliqueSize=0 does not override (no CVARs set either)
  {
    auto cs = makeCommState(0);
    cs->setNvlFabricTopos(nvlFabricTopologies, true);
    EXPECT_FALSE(cs->nvlFabricCliqueEnabled());
  }

  // vCliqueSize=3 is invalid (8 ranks not divisible by 3)
  {
    auto cs = makeCommState(3);
    EXPECT_DEATH(
        cs->setNvlFabricTopos(nvlFabricTopologies, true),
        "nRanks.*must be evenly divisible by effectiveVCliqueSize");
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

// Parameterized test: verify host()/gPid() preserve real values and node
// grouping is correct across all virtual topology modes.
enum class TopoMode { kSystem, kNolocal, kVnode, kVClique };

struct TopoTestParam {
  TopoMode mode;
  int vCliqueSize; // only used for kVClique
  int expectedNNodes;
  int expectedNLocalRanks; // for rank 0
};

std::string topoTestName(const testing::TestParamInfo<TopoTestParam>& info) {
  switch (info.param.mode) {
    case TopoMode::kSystem:
      return "system";
    case TopoMode::kNolocal:
      return "nolocal";
    case TopoMode::kVnode:
      return "vnode4";
    case TopoMode::kVClique:
      return "vclique" + std::to_string(info.param.vCliqueSize);
  }
  return "unknown";
}

class GpidTopoTest : public ::testing::TestWithParam<TopoTestParam> {
 protected:
  static constexpr int kRanksPerHost = 4;
  static constexpr int kNumHosts = 2;
  static constexpr int kNRanks = kRanksPerHost * kNumHosts;
  static constexpr int kCudaDev = 0;
  static constexpr int kCudaArch = 90;
  static constexpr int64_t kBusId = 25;
  static constexpr uint64_t kCommHash = 0;

  std::vector<RankTopology> makeRankTopologies() {
    const std::string kSu;
    const std::array<std::pair<const char*, const char*>, kNumHosts> hostInfo =
        {{{kHost0, kRtsw0}, {kHost1, kRtsw1}}};
    std::vector<RankTopology> topos;
    topos.reserve(kNRanks);
    for (int r = 0; r < kNRanks; r++) {
      const auto& [host, rtsw] = hostInfo[r / kRanksPerHost];
      const int pid = 1000 * (r / kRanksPerHost + 1) + r % kRanksPerHost;
      topos.emplace_back(
          createRankTopology(r, kDc, kZone, kSu, rtsw, host, -1, pid));
    }
    return topos;
  }
};

TEST_P(GpidTopoTest, HostAndGpidPreserved) {
  const auto& [mode, vCliqueSize, expectedNNodes, expectedNLocalRanks] =
      GetParam();
  auto rankTopologies = makeRankTopologies();

  // Set CVAR for vnode mode (needs ncclx EnvRAII, not SysEnvRAII)
  // nolocal and vClique are passed via constructor params instead.
  std::unique_ptr<SysEnvRAII> envTopo, envPpn;
  if (mode == TopoMode::kVnode) {
    envTopo =
        std::make_unique<SysEnvRAII>("NCCL_COMM_STATE_DEBUG_TOPO", "vnode");
    envPpn = std::make_unique<SysEnvRAII>(
        "NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS", "4");
  }

  const bool noLocal = (mode == TopoMode::kNolocal);
  auto commState = std::make_unique<CommStateX>(
      0,
      kNRanks,
      kCudaDev,
      kCudaArch,
      kBusId,
      kCommHash,
      rankTopologies,
      std::vector<int>{},
      "" /* commDesc */,
      noLocal,
      vCliqueSize);

  // Node grouping matches expected virtual topology
  EXPECT_EQ(commState->nNodes(), expectedNNodes);
  EXPECT_EQ(commState->nLocalRanks(0), expectedNLocalRanks);

  // Real hostname and gPid preserved for all ranks
  std::set<std::string> gPids;
  for (int r = 0; r < kNRanks; ++r) {
    EXPECT_EQ(commState->host(r), std::string(rankTopologies[r].host))
        << "rank " << r << " should have real host";

    std::string expected = std::string(rankTopologies[r].host) + ":" +
        std::to_string(rankTopologies[r].pid);
    EXPECT_EQ(commState->gPid(r), expected)
        << "rank " << r << " should have real gPid";

    auto [it, inserted] = gPids.insert(commState->gPid(r));
    EXPECT_TRUE(inserted) << "gPid collision at rank " << r;
  }
}

INSTANTIATE_TEST_SUITE_P(
    CommStateXTest,
    GpidTopoTest,
    ::testing::Values(
        // system: 2 real hosts × 4 ranks each
        TopoTestParam{TopoMode::kSystem, 0, 2, 4},
        // nolocal: each rank is its own node
        TopoTestParam{TopoMode::kNolocal, 0, 8, 1},
        // vnode with nLocalRanks=4: 2 virtual nodes
        TopoTestParam{TopoMode::kVnode, 0, 2, 4},
        // vClique size=2: 4 virtual nodes of 2 ranks each
        TopoTestParam{TopoMode::kVClique, 2, 4, 2}),
    topoTestName);

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

TEST(CommStateXTest, nvlFabricWithNoLocal) {
  const int rank = 0;
  const int nRanks = 8;
  const int cudaDev = 0;
  const int cudaArch = 90;
  const int64_t busId = 25;
  const uint64_t commHash = 0;

  // Create CommStateX with empty rank topologies (will be set by
  // initRankTopologyNolocal)
  auto commState = std::make_unique<CommStateX>(
      rank,
      nRanks,
      cudaDev,
      cudaArch,
      busId,
      commHash,
      std::vector<RankTopology>{},
      std::vector<int>{},
      "" /* commDesc */,
      true /* noLocal */);

  commState->initRankTopologyNolocal();

  // Set up NVL fabric with 2 clusters of 4 ranks each (e.g. GB200 2-GPU trays)
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
  commState->setNvlFabricTopos(nvlFabricTopologies, true);

  // NVL fabric should still be enabled (for transport)
  EXPECT_TRUE(commState->nvlFabricEnabled());

  // But topology getters should respect noLocal: each rank is its own node
  for (int i = 0; i < nRanks; ++i) {
    EXPECT_EQ(commState->nLocalRanks(i), 1);
    EXPECT_EQ(commState->localRank(i), 0);
    EXPECT_EQ(commState->node(i), i);
  }
  EXPECT_EQ(commState->nNodes(), nRanks);
}

TEST(CommStateXTest, nvlFabricWithNoLocalCvar) {
  const int rank = 0;
  const int nRanks = 8;
  const int cudaDev = 0;
  const int cudaArch = 90;
  const int64_t busId = 25;
  const uint64_t commHash = 0;

  // Set the CVAR to nolocal but do NOT set the noLocal_ hint bool
  EnvRAII noLocalCvar(
      NCCL_COMM_STATE_DEBUG_TOPO, NCCL_COMM_STATE_DEBUG_TOPO::nolocal);

  auto commState = std::make_unique<CommStateX>(
      rank,
      nRanks,
      cudaDev,
      cudaArch,
      busId,
      commHash,
      std::vector<RankTopology>{},
      std::vector<int>{},
      "" /* commDesc */,
      false /* noLocal - hint is NOT set, only CVAR */);

  // initRankTopologyNolocal sets host-based nLocalRanks=1 for all ranks
  commState->initRankTopologyNolocal();

  // Set up NVL fabric with 2 clusters of 4 ranks each (e.g. GB200 2-GPU trays)
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
  commState->setNvlFabricTopos(nvlFabricTopologies, true);

  // NVL fabric should still be enabled (for transport)
  EXPECT_TRUE(commState->nvlFabricEnabled());

  // The CVAR should be respected: each rank is its own node with nLocalRanks=1
  for (int i = 0; i < nRanks; ++i) {
    EXPECT_EQ(commState->nLocalRanks(i), 1);
    EXPECT_EQ(commState->localRank(i), 0);
    EXPECT_EQ(commState->node(i), i);
  }
  EXPECT_EQ(commState->nNodes(), nRanks);
}

} // namespace ncclx
