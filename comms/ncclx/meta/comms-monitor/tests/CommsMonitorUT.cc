// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "comm.h" // @manual
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/testinfra/DistEnvironmentBase.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/comms-monitor/CommsMonitor.h" // @manual

// Need to be in the ncclx::comms_monitor scope to be the friend class
namespace ncclx::comms_monitor {
class CommsMonitorTest : public NcclxBaseTestFixture {
 public:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp({{"NCCL_COMMSMONITOR_ENABLE", "1"}});
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
  const auto refNum = CommsMonitor::getNumOfCommMonitoring();
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  // Real comm is auto-registered during init
  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), refNum + 1);
}

TEST_F(CommsMonitorTest, TestRegisterDeregisterComm) {
  const auto refNum = CommsMonitor::getNumOfCommMonitoring();
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), refNum + 1);
  EXPECT_TRUE(CommsMonitor::deregisterComm(comm.get()));
  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), refNum + 1);
}

TEST_F(CommsMonitorTest, TestNcclTopoInfoFromNcclComm) {
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  auto topoInfo = getTopoInfoFromNcclComm(comm.get());
  EXPECT_GE(topoInfo.nChannels(), 1);
  ASSERT_GE(topoInfo.rings()->size(), 1);
  ASSERT_GE(topoInfo.treeInfos()->size(), 1);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
