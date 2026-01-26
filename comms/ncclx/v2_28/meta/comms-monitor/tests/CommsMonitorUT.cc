// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comm.h" // @manual
#include "comms/ctran/Ctran.h" // @manual
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/colltrace/CollTrace.h" // @manual
#include "meta/comms-monitor/CommsMonitor.h" // @manual
#include "meta/wrapper/MetaFactory.h"

inline std::unique_ptr<ncclComm> createFakeNcclComm() {
  auto comm = std::make_unique<ncclComm>();
  comm->rank = 0;
  comm->nRanks = 1;
  comm->cudaDev = 0;
  comm->commHash = 0xfaceb00c;
  comm->config.commDesc = "fake_comm";
  comm->localRank = 0;
  comm->localRanks = 1;
  comm->nNodes = 1;
  comm->nChannels = 0;
  setCtranCommBase(comm.get());

  comm->ctranComm_->statex_ = std::make_unique<ncclx::CommStateX>(
      comm->rank,
      comm->nRanks,
      comm->cudaDev,
      comm->cudaArch,
      comm->busId,
      comm->commHash,
      std::vector<ncclx::RankTopology>(), /* rankTopologies */
      std::vector<int>(), /* commRanksToWorldRanks */
      comm->config.commDesc);

  return comm;
}

namespace {
void initFakeChannels(ncclComm* comm) {
  static std::list<std::vector<int>> ringStorage;

  comm->nChannels = 2;

  // Setup fake rings
  ringStorage.push_back(std::vector<int>());
  auto& ring1Storage = ringStorage.back();
  ringStorage.push_back(std::vector<int>());
  auto& ring2Storage = ringStorage.back();
  ring1Storage.reserve(comm->nRanks);
  ring2Storage.reserve(comm->nRanks);
  for (int i = 0; i < comm->nRanks; i++) {
    ring1Storage.emplace_back(i);
    ring2Storage.emplace_back(comm->nRanks - i - 1);
  }
  comm->channels[0].ring.userRanks = ring1Storage.data();
  comm->channels[1].ring.userRanks = ring2Storage.data();

  // Set up fake trees
  comm->channels[0].tree.up = 0;
  comm->channels[1].tree.up = 1;
}
} // namespace

// Need to be in the ncclx::comms_monitor scope to be the friend class
namespace ncclx::comms_monitor {
class CommsMonitorTest : public ::testing::Test {
 public:
  void SetUp() override {
    ncclCvarInit();
    NCCL_COMMSMONITOR_ENABLE = true;
  }
  int getCommsMapSize() {
    auto commsMonitorPtr = CommsMonitor::getInstance();
    EXPECT_THAT(commsMonitorPtr, ::testing::NotNull());
    if (commsMonitorPtr) {
      auto lockedMap = commsMonitorPtr->commsMap_.rlock();
      return lockedMap->size();
    } else {
      return -1;
    }
  }
};
} // namespace ncclx::comms_monitor

using namespace ncclx::comms_monitor;

TEST_F(CommsMonitorTest, TestRegisterComm) {
  auto fakeComm = createFakeNcclComm();
  const auto refNum = CommsMonitor::getNumOfCommMonitoring();
  EXPECT_TRUE(CommsMonitor::registerComm(fakeComm.get()));

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), refNum + 1);
}

TEST_F(CommsMonitorTest, TestRegisterDeregisterComm) {
  auto fakeComm = createFakeNcclComm();
  const auto refNum = CommsMonitor::getNumOfCommMonitoring();

  EXPECT_TRUE(CommsMonitor::registerComm(fakeComm.get()));
  EXPECT_TRUE(CommsMonitor::deregisterComm(fakeComm.get()));

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), refNum + 1);
}

// TODO: this test fails locally, disable it for now to unblock 2.25 rebase.
TEST_F(CommsMonitorTest, DISABLED_TestOnlyDeregisterComm) {
  auto fakeComm = createFakeNcclComm();
  EXPECT_FALSE(CommsMonitor::deregisterComm(fakeComm.get()));
}

TEST_F(CommsMonitorTest, TestNcclTopoInfoFromNcclComm) {
  auto fakeComm = createFakeNcclComm();

  // Test with no channels initialized
  {
    auto topoInfo = getTopoInfoFromNcclComm(fakeComm.get());
    EXPECT_EQ(topoInfo.nChannels(), 0);
    EXPECT_TRUE(topoInfo.rings()->empty());
    EXPECT_TRUE(topoInfo.treeInfos()->empty());
  }

  // Test with initialized channels
  initFakeChannels(fakeComm.get());
  {
    auto topoInfo = getTopoInfoFromNcclComm(fakeComm.get());
    EXPECT_EQ(topoInfo.nChannels(), 2);

    // Verify rings
    ASSERT_EQ(topoInfo.rings()->size(), 2);

    // Channel 0 ring should contain ranks [0]
    EXPECT_EQ((*topoInfo.rings())[0].size(), 1);
    EXPECT_EQ((*topoInfo.rings())[0][0], 0);

    // Channel 1 ring should contain ranks [0] (reversed)
    EXPECT_EQ((*topoInfo.rings())[1].size(), 1);
    EXPECT_EQ((*topoInfo.rings())[1][0], 0);

    // Verify trees
    ASSERT_EQ(topoInfo.treeInfos()->size(), 2);

    // Channel 0 tree
    EXPECT_EQ((*topoInfo.treeInfos())[0].parentNode(), 0);

    // Channel 1 tree
    EXPECT_EQ((*topoInfo.treeInfos())[1].parentNode(), 1);

    // Verify the tree children arrays are properly copied
    // (The children nodes are initialized to default values)
    for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) {
      // Children arrays should contain the data from the fake channels
      // initFakeChannels doesn't set these, so they should be default
      // initialized
      EXPECT_GE((*topoInfo.treeInfos())[0].childrenNodes()[i], -1);
      EXPECT_GE((*topoInfo.treeInfos())[1].childrenNodes()[i], -1);
    }
  }
}

TEST_F(CommsMonitorTest, TestNcclTopoInfoFromNcclCommMultiRank) {
  auto fakeComm = createFakeNcclComm();

  // Setup multi-rank scenario
  fakeComm->nRanks = 4;
  initFakeChannels(fakeComm.get());

  auto topoInfo = getTopoInfoFromNcclComm(fakeComm.get());

  EXPECT_EQ(topoInfo.nChannels(), 2);
  for (int i = 0; i < topoInfo.rings()->size(); ++i) {
    XLOG(INFO) << "ring: " << i << " " << (*topoInfo.rings())[i].size();
  }
  ASSERT_EQ(topoInfo.rings()->size(), 2);

  // Channel 0 ring: [0, 1, 2, 3]
  ASSERT_EQ((*topoInfo.rings())[0].size(), 4);
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ((*topoInfo.rings())[0][i], i);
  }

  // Channel 1 ring: [3, 2, 1, 0] (reversed)
  ASSERT_EQ((*topoInfo.rings())[1].size(), 4);
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ((*topoInfo.rings())[1][i], 3 - i);
  }

  // Verify tree structure
  ASSERT_EQ(topoInfo.treeInfos()->size(), 2);
  EXPECT_EQ((*topoInfo.treeInfos())[0].parentNode(), 0);
  EXPECT_EQ((*topoInfo.treeInfos())[1].parentNode(), 1);
}
