// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "meta/comms-monitor/CommsMonitor.h"
#include "nccl.h"

using namespace ncclx::comms_monitor;
using ncclx::test::NcclCommRAII;

namespace ncclx::comms_monitor {
class CommsMonitorTest {
 public:
  static void resetCommsMap() {
    auto commsMonitorPtr = CommsMonitor::getInstance();
    EXPECT_THAT(commsMonitorPtr, ::testing::NotNull());
    if (commsMonitorPtr) {
      auto lockedMap = commsMonitorPtr->commsMap_.wlock();
      lockedMap->clear();
    }
  }
};
} // namespace ncclx::comms_monitor

class CommsMonitorDist : public NcclxBaseTestFixture {
 public:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp();

    ncclCvarInit();
    NCCL_COMMSMONITOR_ENABLE = true;

    CUDACHECK_TEST(cudaStreamCreate(&this->stream));

    ncclx::comms_monitor::CommsMonitorTest::resetCommsMap();
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
    NcclxBaseTestFixture::TearDown();
  }

  void prepareAllreduce(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
  }

 protected:
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  void* sendHandle{nullptr};
  void* recvHandle{nullptr};
  cudaStream_t stream;
};

TEST_F(CommsMonitorDist, testNotEnable) {
  NCCL_COMMSMONITOR_ENABLE = false;
  NcclCommRAII comm{globalRank, numRanks, localRank, bootstrap_.get()};

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 0);
}

TEST_F(CommsMonitorDist, testOneComm) {
  NcclCommRAII comm{globalRank, numRanks, localRank, bootstrap_.get()};

  auto count = 1 << 20;
  auto nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 1);
}

TEST_F(CommsMonitorDist, testOneCommDeregister) {
  {
    NcclCommRAII comm{globalRank, numRanks, localRank, bootstrap_.get()};
    EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 1);
  }
  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 1);
}

TEST_F(CommsMonitorDist, testMultipleComms) {
  NcclCommRAII comm1{globalRank, numRanks, localRank, bootstrap_.get()};
  NcclCommRAII comm2{globalRank, numRanks, localRank, bootstrap_.get()};
  NcclCommRAII comm3{globalRank, numRanks, localRank, bootstrap_.get()};
  NcclCommRAII comm4{globalRank, numRanks, localRank, bootstrap_.get()};

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 4);
}

TEST_F(CommsMonitorDist, testMultipleCommsDeregister) {
  {
    NcclCommRAII comm1{globalRank, numRanks, localRank, bootstrap_.get()};
    NcclCommRAII comm2{globalRank, numRanks, localRank, bootstrap_.get()};
    NcclCommRAII comm3{globalRank, numRanks, localRank, bootstrap_.get()};
    NcclCommRAII comm4{globalRank, numRanks, localRank, bootstrap_.get()};

    EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 4);
  }
  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 4);
}

TEST_F(CommsMonitorDist, testCommSplit) {
  NcclCommRAII origComm{globalRank, numRanks, localRank, bootstrap_.get()};

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 1);

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints splitHints({{"commDesc", "split_comm"}});
  config.hints = &splitHints;

  ncclComm_t splitComm;

  ncclCommSplit(origComm, globalRank % 2, globalRank, &splitComm, &config);

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 2);

  ncclCommDestroy(splitComm);

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 2);
}

TEST_F(CommsMonitorDist, testCommSplitNoColor) {
  NcclCommRAII origComm{globalRank, numRanks, localRank, bootstrap_.get()};

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 1);

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints noColorHints({{"commDesc", "split_comm"}});
  config.hints = &noColorHints;

  ncclComm_t splitComm;

  if (globalRank % 2 == 0) {
    ncclCommSplit(origComm, 1, globalRank, &splitComm, &config);

    EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 2);

    ncclCommDestroy(splitComm);
  } else {
    ncclCommSplit(
        origComm, NCCL_SPLIT_NOCOLOR, globalRank, &splitComm, &config);

    EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 1);
  }
}

TEST_F(CommsMonitorDist, testOneCommDump) {
  NcclCommRAII comm{globalRank, numRanks, localRank, bootstrap_.get()};

  auto count = 1 << 20;
  auto nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  auto commDumpsMaybe = CommsMonitor::commDumpAll();
  ASSERT_TRUE(commDumpsMaybe.has_value());
  auto& commDumps = commDumpsMaybe.value();

  EXPECT_EQ(commDumps.size(), 1);
  const auto& [commHash, commDump] = *commDumps.cbegin();
  EXPECT_EQ(commHash, hashToHexStr(comm->commHash));
  EXPECT_GT(commDump.size(), 0);
}

TEST_F(CommsMonitorDist, testMultipleCommDump) {
  // TODO: Change it to use vector. Currently NcclCommRAII has some
  // compatibility issue with vector.
  NcclCommRAII comm1{globalRank, numRanks, localRank, bootstrap_.get()};
  NcclCommRAII comm2{globalRank, numRanks, localRank, bootstrap_.get()};
  NcclCommRAII comm3{globalRank, numRanks, localRank, bootstrap_.get()};
  NcclCommRAII comm4{globalRank, numRanks, localRank, bootstrap_.get()};

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 4);

  auto count = 1 << 20;
  auto nColl = 10;

  prepareAllreduce(count);

  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(ncclAllReduce(
        sendBuf, recvBuf, count, ncclInt, ncclSum, comm1, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(ncclAllReduce(
        sendBuf, recvBuf, count, ncclInt, ncclSum, comm2, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(ncclAllReduce(
        sendBuf, recvBuf, count, ncclInt, ncclSum, comm3, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(ncclAllReduce(
        sendBuf, recvBuf, count, ncclInt, ncclSum, comm4, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  auto commDumpsMaybe = CommsMonitor::commDumpAll();
  ASSERT_TRUE(commDumpsMaybe.has_value());
  auto& commDumps = commDumpsMaybe.value();

  std::unordered_set<std::string> commHashes{
      hashToHexStr(comm1->commHash),
      hashToHexStr(comm2->commHash),
      hashToHexStr(comm3->commHash),
      hashToHexStr(comm4->commHash)};

  EXPECT_EQ(commDumps.size(), 4);

  for (const auto& [commHash, commDump] : commDumps) {
    EXPECT_TRUE(commHashes.contains(commHash));
    EXPECT_GT(commDump.size(), 0);
    commHashes.erase(commHash);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
