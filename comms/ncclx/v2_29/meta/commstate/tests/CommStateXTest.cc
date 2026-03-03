// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdlib>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "comms/testinfra/TestUtils.h"

#include "comms/ctran/commstate/CommStateX.h"
#include "param.h" // @manual

#include "comms/utils/cvars/nccl_cvars.h"

namespace ncclx {

RankTopology
createRankTopology(int rank, const std::string& host, const std::string& rtsw) {
  RankTopology topo;
  topo.rank = rank;
  std::strcpy(topo.host, host.c_str());
  std::strcpy(topo.rtsw, rtsw.c_str());
  return topo;
}

class CommStateXTest : public ::testing::Test {
 protected:
  void SetUp() override {
    initEnv();
  }

  void TearDown() override {}
};

static void fillDummyComm(ncclComm& comm, int numNvlDomain = 1) {
  comm.rank = 0;
  comm.nRanks = 16;
  comm.cudaDev = 0;
  comm.commHash = 123456789;
  comm.config.commDesc = "default_pg:0";

  comm.localRank = 0;
  comm.localRanks = 8 / numNvlDomain; // local ranks in the same NVL domain
  comm.nNodes = comm.nRanks / comm.localRanks;
  comm.rankToNode = new int[comm.nRanks];
  comm.localRankToRank = new int[comm.localRanks];
  comm.rankToLocalRank = new int[comm.nRanks];
  comm.peerInfo = new struct ncclPeerInfo[comm.nRanks];

  for (int i = 0; i < comm.nRanks; ++i) {
    comm.rankToNode[i] = i / comm.localRanks;
    comm.rankToLocalRank[i] = i % comm.localRanks;
    snprintf(
        comm.peerInfo[i].hostname,
        kMaxHostNameLen,
        "host%d",
        // assign the same physical node for every numNvlDomain number of
        // logical nodes
        comm.rankToNode[i] / numNvlDomain);
  }

  for (int i = 0; i < comm.localRanks; ++i) {
    comm.localRankToRank[i] = i;
  }

  comm.nChannels = 4;
  for (int i = 0; i < comm.nChannels; ++i) {
    // create a dummy channel
    comm.channels[i] = ncclChannel();
    comm.channels[i].ring = ncclRing();
    comm.channels[i].ring.userRanks = new int[comm.nRanks];
    for (int j = 0; j < comm.nRanks; ++j) {
      comm.channels[i].ring.userRanks[j] = j;
    }
  }
}

TEST(CommStateXTest, CreateVNodeFromNcclComm) {
  // Create a dummy ncclComm
  ncclComm comm;
  fillDummyComm(comm);

  // Create CommStateX from ncclComm with noLocal mode
  // Expect CommStateX to generate nLocalRanks=1 topo
  int nLocalRanks = 2;
  auto env1 =
      EnvRAII(NCCL_COMM_STATE_DEBUG_TOPO, NCCL_COMM_STATE_DEBUG_TOPO::vnode);
  auto env2 =
      EnvRAII(NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS, nLocalRanks);

  auto state = ncclx::createCommStateXFromNcclComm(&comm);

  EXPECT_EQ(state->rank(), 0);
  EXPECT_EQ(state->nRanks(), comm.nRanks);
  EXPECT_EQ(state->cudaDev(), comm.cudaDev);
  EXPECT_EQ(state->cudaArch(), comm.cudaArch);
  EXPECT_EQ(state->busId(), comm.busId);
  EXPECT_EQ(state->localRank(), state->rank() % nLocalRanks);
  EXPECT_EQ(state->nLocalRanks(), nLocalRanks);
  EXPECT_EQ(state->nNodes(), comm.nRanks / nLocalRanks);
  EXPECT_EQ(state->commHash(), comm.commHash);
  EXPECT_EQ(state->commDesc(), comm.config.commDesc);
  for (int i = 0; i < state->nRanks(); ++i) {
    EXPECT_EQ(state->node(i), i / nLocalRanks);
  }
  for (int i = 0; i < state->nRanks(); ++i) {
    EXPECT_EQ(state->localRank(i), i % nLocalRanks);
  }
  for (int nodeId = 0; nodeId < state->nNodes(); ++nodeId) {
    for (int localRankId = 0; localRankId < nLocalRanks; ++localRankId) {
      EXPECT_EQ(
          state->localRankToRank(localRankId, nodeId),
          nodeId * nLocalRanks + localRankId);
    }
  }

  delete[] comm.rankToNode;
  delete[] comm.localRankToRank;
  delete[] comm.peerInfo;
  delete[] comm.rankToLocalRank;
  for (int i = 0; i < comm.nChannels; ++i) {
    delete[] comm.channels[i].ring.userRanks;
  }
}

TEST(CommStateXTest, CreateNoLocalFromNcclComm) {
  // Create a dummy ncclComm
  ncclComm comm;
  fillDummyComm(comm);

  // Create CommStateX from ncclComm with noLocal mode
  // Expect CommStateX to generate nLocalRanks=1 topo
  EnvRAII env(NCCL_COMM_STATE_DEBUG_TOPO, NCCL_COMM_STATE_DEBUG_TOPO::nolocal);

  auto state = ncclx::createCommStateXFromNcclComm(&comm);

  EXPECT_EQ(state->rank(), 0);
  EXPECT_EQ(state->nRanks(), comm.nRanks);
  EXPECT_EQ(state->cudaDev(), comm.cudaDev);
  EXPECT_EQ(state->cudaArch(), comm.cudaArch);
  EXPECT_EQ(state->busId(), comm.busId);
  EXPECT_EQ(state->localRank(), 0);
  EXPECT_EQ(state->nLocalRanks(), 1);
  EXPECT_EQ(state->nNodes(), comm.nRanks);
  EXPECT_EQ(state->commHash(), comm.commHash);
  EXPECT_EQ(state->commDesc(), comm.config.commDesc);
  for (int i = 0; i < state->nRanks(); ++i) {
    EXPECT_EQ(state->node(i), i);
  }
  for (int i = 0; i < state->nRanks(); ++i) {
    EXPECT_EQ(state->localRank(i), 0);
  }
  for (int nodeId = 0; nodeId < state->nNodes(); ++nodeId) {
    EXPECT_EQ(state->localRankToRank(0, nodeId), nodeId);
  }

  delete[] comm.rankToNode;
  delete[] comm.localRankToRank;
  delete[] comm.peerInfo;
  delete[] comm.rankToLocalRank;
  for (int i = 0; i < comm.nChannels; ++i) {
    delete[] comm.channels[i].ring.userRanks;
  }
}

class CommStateXNcclCommTestParamFixture
    : public CommStateXTest,
      public ::testing::WithParamInterface<int> {};

TEST_P(CommStateXNcclCommTestParamFixture, CreateFromNcclComm) {
  auto numNvlDomain = GetParam();
  // create a dummy ncclComm
  ncclComm comm;
  fillDummyComm(comm, numNvlDomain);

  // create CommStateX from ncclComm
  auto state = ncclx::createCommStateXFromNcclComm(&comm);
  EXPECT_EQ(state->rank(), 0);
  EXPECT_EQ(state->nRanks(), comm.nRanks);
  EXPECT_EQ(state->cudaDev(), comm.cudaDev);
  EXPECT_EQ(state->commHash(), comm.commHash);
  EXPECT_EQ(state->commDesc(), comm.config.commDesc);
  EXPECT_EQ(state->localRank(), comm.localRank);
  EXPECT_EQ(state->nLocalRanks(), comm.localRanks);
  for (int i = 0; i < state->nRanks(); ++i) {
    EXPECT_EQ(state->node(i), comm.rankToNode[i]);
  }
  for (int i = 0; i < state->nRanks(); ++i) {
    EXPECT_EQ(state->localRank(i), comm.rankToLocalRank[i]);
  }
  for (int i = 0; i < state->nLocalRanks(); ++i) {
    EXPECT_EQ(state->localRankToRank(i), comm.localRankToRank[i]);
  }
  for (int nodeId = 0; nodeId < state->nNodes(); ++nodeId) {
    for (int i = 0; i < state->nLocalRanks(); ++i) {
      EXPECT_EQ(
          state->localRankToRank(i, nodeId), nodeId * comm.localRanks + i);
    }
  }
  EXPECT_EQ(state->nNodes(), comm.nNodes);

  delete[] comm.rankToNode;
  delete[] comm.localRankToRank;
  delete[] comm.peerInfo;
  delete[] comm.rankToLocalRank;
  for (int i = 0; i < comm.nChannels; ++i) {
    delete[] comm.channels[i].ring.userRanks;
  }
}

INSTANTIATE_TEST_SUITE_P(
    CommStateXNcclCommTest,
    CommStateXNcclCommTestParamFixture,
    ::testing::Values(1, 2),
    [&](const testing::TestParamInfo<
        CommStateXNcclCommTestParamFixture::ParamType>& info) {
      return std::to_string(info.param) + "nvlDomains";
    });

} // namespace ncclx
