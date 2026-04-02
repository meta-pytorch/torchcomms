// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Verify CTran topology and backend selection when NCCL_TOPO_FILE_PATH is not
// set. Uses vnode topology mode (8 local ranks per node) with
// CtranDistTestFixture to avoid any NCCL dependency.
//
// Two test configurations (see BUCK):
//   nvl_test (1x8): 1 node, 8 local ranks — topology + NVL + IB tests.
//   ib_test  (2x8): 2 nodes, 8 local ranks each — topology + IB tests.
// NvlBackendForLocalPeers is excluded via GTEST_FILTER because RE container
// isolation prevents NVLink P2P (different /dev/shm per rank).

#include <folly/init/Init.h>

#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"

class NoMetaTopoFileEnvironment : public ctran::CtranDistEnvironment {
 public:
  void SetUp() override {
    ctran::CtranDistEnvironment::SetUp();
    setenv("NCCL_DEBUG", "WARN", 0);
  }
};

class NoMetaTopoFileTest : public ctran::CtranDistTestFixture {
 protected:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
    setenv("NCCL_TOPO_FILE_PATH", "", 1);
    setenv("NCCL_COMM_STATE_DEBUG_TOPO", "vnode", 1);
    setenv("NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS", "8", 1);
    CtranDistTestFixture::SetUp();

    ctranComm = makeCtranComm();
    ASSERT_NE(ctranComm, nullptr);
  }

  void TearDown() override {
    ctranComm.reset();
    CtranDistTestFixture::TearDown();
  }

  std::unique_ptr<CtranComm> ctranComm;
};

TEST_F(NoMetaTopoFileTest, TopologyCorrectWithoutTopoFile) {
  ASSERT_NE(ctranComm->statex_, nullptr);

  auto* statex = ctranComm->statex_.get();

  EXPECT_EQ(statex->rank(), globalRank);
  EXPECT_EQ(statex->nRanks(), numRanks);

  // vnode mode with nLocalRanks=8
  EXPECT_EQ(statex->nLocalRanks(), 8);
  EXPECT_EQ(statex->nNodes(), numRanks / 8);

  // Same-node peers should be detected as such
  for (int j = 0; j < numRanks; ++j) {
    if (statex->node(j) == statex->node(globalRank)) {
      EXPECT_TRUE(statex->isSameNode(globalRank, j))
          << "rank " << globalRank << " and " << j << " should be on same node";
    }
  }

  // Without topo file: datacenter topology fields are empty
  if (numRanks > 1) {
    int peer = (globalRank + 1) % numRanks;
    // isSameRack returns false (empty rtsw and su)
    EXPECT_FALSE(statex->isSameRack(globalRank, peer));
    // isSameZone/isSameDc return true (empty == empty)
    EXPECT_TRUE(statex->isSameZone(globalRank, peer));
    EXPECT_TRUE(statex->isSameDc(globalRank, peer));
  }
}

TEST_F(NoMetaTopoFileTest, NvlBackendForLocalPeers) {
  ASSERT_NE(ctranComm->ctran_, nullptr);

  auto* mapper = ctranComm->ctran_->mapper.get();
  ASSERT_NE(mapper, nullptr);

  auto* statex = ctranComm->statex_.get();

  ASSERT_EQ(statex->nLocalRanks(), 8)
      << "Need 8 local ranks for NVL topology (nLocalRanks="
      << statex->nLocalRanks() << "). Requires shared /dev/shm.";

  int nvlPeerCt = 0;
  for (int peer = 0; peer < numRanks; ++peer) {
    if (peer != globalRank &&
        mapper->hasBackend(peer, CtranMapperBackend::NVL)) {
      nvlPeerCt += 1;
    }
  }

  EXPECT_EQ(nvlPeerCt, 7)
      << "rank " << globalRank << " has " << nvlPeerCt
      << " NVL peers (nLocalRanks=" << statex->nLocalRanks()
      << ", nNodes=" << statex->nNodes()
      << "). NCCL topology may not detect NVLink on this hardware.";
}

TEST_F(NoMetaTopoFileTest, IbBackendWithDefaultTopology) {
  ASSERT_GE(numRanks, 2) << "Need at least 2 ranks for IB test";
  ASSERT_NE(ctranComm->ctran_, nullptr);

  auto* mapper = ctranComm->ctran_->mapper.get();
  ASSERT_NE(mapper, nullptr);

  auto* statex = ctranComm->statex_.get();

  bool ibAvailable = false;
  for (int peer = 0; peer < numRanks; ++peer) {
    if (peer != globalRank &&
        mapper->hasBackend(peer, CtranMapperBackend::IB)) {
      ibAvailable = true;
      EXPECT_FALSE(statex->isSameRack(globalRank, peer));
      EXPECT_TRUE(statex->isSameZone(globalRank, peer));
      EXPECT_TRUE(statex->isSameDc(globalRank, peer));
    }
  }

  EXPECT_TRUE(ibAvailable) << "IB hardware not available on any peer";
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new NoMetaTopoFileEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
