// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/comms-monitor/CommsMonitor.h"

using namespace ncclx::comms_monitor;

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

class CommsMonitorDist : public NcclxBaseTest {
 public:
  void SetUp() override {
    NcclxBaseTest::SetUp();
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    ncclCvarInit();
    NCCL_COMMSMONITOR_ENABLE = true;

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));

    ncclx::comms_monitor::CommsMonitorTest::resetCommsMap();
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
  }

  void prepareAllreduce(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  void* sendHandle{nullptr};
  void* recvHandle{nullptr};
  cudaStream_t stream;
};

TEST_F(CommsMonitorDist, testNotEnable) {
  NCCL_COMMSMONITOR_ENABLE = false;
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 0);
}

TEST_F(CommsMonitorDist, testOneComm) {
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};

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
    NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
    EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 1);
  }
  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 1);
}

TEST_F(CommsMonitorDist, testMultipleComms) {
  NcclCommRAII comm1{this->globalRank, this->numRanks, this->localRank};
  NcclCommRAII comm2{this->globalRank, this->numRanks, this->localRank};
  NcclCommRAII comm3{this->globalRank, this->numRanks, this->localRank};
  NcclCommRAII comm4{this->globalRank, this->numRanks, this->localRank};

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 4);
}

TEST_F(CommsMonitorDist, testMultipleCommsDeregister) {
  {
    NcclCommRAII comm1{this->globalRank, this->numRanks, this->localRank};
    NcclCommRAII comm2{this->globalRank, this->numRanks, this->localRank};
    NcclCommRAII comm3{this->globalRank, this->numRanks, this->localRank};
    NcclCommRAII comm4{this->globalRank, this->numRanks, this->localRank};

    EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 4);
  }
  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 4);
}

TEST_F(CommsMonitorDist, testNcclTopoInfoFromNcclComm) {
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};

  // Call fromNcclComm on the real communicator
  auto topoInfo = NcclTopoInfo::fromNcclComm(comm);

  // Verify basic structure
  EXPECT_GE(topoInfo.nChannels, 0);
  EXPECT_EQ(topoInfo.rings.size(), topoInfo.nChannels);
  EXPECT_EQ(topoInfo.trees.size(), topoInfo.nChannels);

  // If channels are present, verify their structure
  if (topoInfo.nChannels > 0) {
    for (size_t i = 0; i < topoInfo.nChannels; i++) {
      // Each ring should have numRanks elements
      EXPECT_EQ(topoInfo.rings[i].size(), static_cast<size_t>(this->numRanks));

      // Verify all ranks are present in the ring
      std::set<int> ringRanks(
          topoInfo.rings[i].begin(), topoInfo.rings[i].end());
      EXPECT_EQ(ringRanks.size(), static_cast<size_t>(this->numRanks));

      // Verify all rank values are valid (0 to numRanks-1)
      for (int rank : topoInfo.rings[i]) {
        EXPECT_GE(rank, 0);
        EXPECT_LT(rank, this->numRanks);
      }

      // Tree parent should be a valid rank or -1 (no parent)
      EXPECT_GE(topoInfo.trees[i].parentNode, -1);
      if (topoInfo.trees[i].parentNode >= 0) {
        EXPECT_LT(topoInfo.trees[i].parentNode, this->numRanks);
      }

      // Tree children should be valid ranks or -1 (no child)
      for (int j = 0; j < NCCL_MAX_TREE_ARITY; j++) {
        int child = topoInfo.trees[i].childrenNodes[j];
        EXPECT_GE(child, -1);
        if (child >= 0) {
          EXPECT_LT(child, this->numRanks);
        }
      }
    }
  }
}

TEST_F(CommsMonitorDist, testNcclTopoInfoConsistencyAcrossOperations) {
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};

  // Get topology info before any operations
  auto topoInfoBefore = NcclTopoInfo::fromNcclComm(comm);

  // Perform some collective operations
  auto count = 1 << 10;
  prepareAllreduce(count);

  for (int i = 0; i < 5; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Get topology info after operations
  auto topoInfoAfter = NcclTopoInfo::fromNcclComm(comm);

  // Topology should remain consistent
  EXPECT_EQ(topoInfoBefore.nChannels, topoInfoAfter.nChannels);
  EXPECT_EQ(topoInfoBefore.rings.size(), topoInfoAfter.rings.size());
  EXPECT_EQ(topoInfoBefore.trees.size(), topoInfoAfter.trees.size());

  // Compare ring structures
  for (size_t i = 0; i < topoInfoBefore.rings.size(); i++) {
    EXPECT_EQ(topoInfoBefore.rings[i], topoInfoAfter.rings[i]);
  }

  // Compare tree structures
  for (size_t i = 0; i < topoInfoBefore.trees.size(); i++) {
    EXPECT_EQ(
        topoInfoBefore.trees[i].parentNode, topoInfoAfter.trees[i].parentNode);
    for (int j = 0; j < NCCL_MAX_TREE_ARITY; j++) {
      EXPECT_EQ(
          topoInfoBefore.trees[i].childrenNodes[j],
          topoInfoAfter.trees[i].childrenNodes[j]);
    }
  }
}

TEST_F(CommsMonitorDist, testNcclTopoInfoFromSplitComm) {
  NcclCommRAII origComm{this->globalRank, this->numRanks, this->localRank};

  // Get topology info from original communicator
  auto origTopoInfo = NcclTopoInfo::fromNcclComm(origComm);

  // Create a split communicator
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  const std::string commDest = "split_comm_topo_test";
  config.commDesc = commDest.c_str();

  ncclComm_t splitComm;
  ncclCommSplit(origComm, globalRank % 2, globalRank, &splitComm, &config);

  // Get topology info from split communicator
  auto splitTopoInfo = NcclTopoInfo::fromNcclComm(splitComm);

  // Split communicator should have valid topology
  EXPECT_GE(splitTopoInfo.nChannels, 0);
  EXPECT_EQ(splitTopoInfo.rings.size(), splitTopoInfo.nChannels);
  EXPECT_EQ(splitTopoInfo.trees.size(), splitTopoInfo.nChannels);

  // If split communicator has channels, verify structure
  if (splitTopoInfo.nChannels > 0) {
    for (size_t i = 0; i < splitTopoInfo.nChannels; i++) {
      // Split communicator should have fewer ranks than original
      EXPECT_LE(splitTopoInfo.rings[i].size(), origTopoInfo.rings[i].size());

      // Verify all ranks in split are valid
      for (int rank : splitTopoInfo.rings[i]) {
        EXPECT_GE(rank, 0);
        // Split comm has its own rank space, so we can't directly compare
        // with original comm's numRanks
      }
    }
  }

  ncclCommDestroy(splitComm);
}

TEST_F(CommsMonitorDist, testNcclTopoInfoFromMultipleComms) {
  NcclCommRAII comm1{this->globalRank, this->numRanks, this->localRank};
  NcclCommRAII comm2{this->globalRank, this->numRanks, this->localRank};

  // Get topology info from both communicators
  auto topoInfo1 = NcclTopoInfo::fromNcclComm(comm1);
  auto topoInfo2 = NcclTopoInfo::fromNcclComm(comm2);

  // Both communicators should have identical topology since they're created
  // with the same parameters
  EXPECT_EQ(topoInfo1.nChannels, topoInfo2.nChannels);
  EXPECT_EQ(topoInfo1.rings.size(), topoInfo2.rings.size());
  EXPECT_EQ(topoInfo1.trees.size(), topoInfo2.trees.size());

  // Compare ring structures
  for (size_t i = 0; i < topoInfo1.rings.size(); i++) {
    EXPECT_EQ(topoInfo1.rings[i], topoInfo2.rings[i]);
  }

  // Compare tree structures
  for (size_t i = 0; i < topoInfo1.trees.size(); i++) {
    EXPECT_EQ(topoInfo1.trees[i].parentNode, topoInfo2.trees[i].parentNode);
    for (int j = 0; j < NCCL_MAX_TREE_ARITY; j++) {
      EXPECT_EQ(
          topoInfo1.trees[i].childrenNodes[j],
          topoInfo2.trees[i].childrenNodes[j]);
    }
  }
}

TEST_F(CommsMonitorDist, testNcclTopoInfoRingConnectivity) {
  // Only run this test if we have at least 2 ranks
  if (this->numRanks < 2) {
    GTEST_SKIP() << "Skipping ring connectivity test with less than 2 ranks";
  }

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  auto topoInfo = NcclTopoInfo::fromNcclComm(comm);

  if (topoInfo.nChannels > 0) {
    for (size_t i = 0; i < topoInfo.nChannels; i++) {
      const auto& ring = topoInfo.rings[i];

      // Ring should form a proper cycle where each rank appears exactly once
      std::set<int> uniqueRanks(ring.begin(), ring.end());
      EXPECT_EQ(uniqueRanks.size(), ring.size())
          << "Ring " << i << " contains duplicate ranks";

      // All ranks from 0 to numRanks-1 should be present
      for (int rank = 0; rank < this->numRanks; rank++) {
        EXPECT_TRUE(uniqueRanks.count(rank) > 0)
            << "Rank " << rank << " missing from ring " << i;
      }
    }
  }
}

TEST_F(CommsMonitorDist, testNcclTopoInfoTreeStructure) {
  // Only run this test if we have at least 2 ranks
  if (this->numRanks < 2) {
    GTEST_SKIP() << "Skipping tree structure test with less than 2 ranks";
  }

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  auto topoInfo = NcclTopoInfo::fromNcclComm(comm);

  if (topoInfo.nChannels > 0) {
    for (size_t i = 0; i < topoInfo.nChannels; i++) {
      const auto& tree = topoInfo.trees[i];

      // Collect all valid child nodes
      std::vector<int> children;
      for (int j = 0; j < NCCL_MAX_TREE_ARITY; j++) {
        if (tree.childrenNodes[j] >= 0) {
          children.push_back(tree.childrenNodes[j]);
          // Child should be a valid rank
          EXPECT_LT(tree.childrenNodes[j], this->numRanks)
              << "Invalid child rank " << tree.childrenNodes[j] << " in tree "
              << i;
        }
      }

      // No duplicate children
      std::set<int> uniqueChildren(children.begin(), children.end());
      EXPECT_EQ(uniqueChildren.size(), children.size())
          << "Tree " << i << " has duplicate children";

      // If comm has more than 1 rank, at least one of parentNode or one child
      // should be valid (not -1) to form a connected tree
      if (this->numRanks > 1) {
        bool hasValidParent = (tree.parentNode >= 0);
        bool hasValidChild = std::ranges::any_of(
            tree.childrenNodes, [](auto num) { return num >= 0; });
        EXPECT_TRUE(hasValidParent || hasValidChild)
            << "Tree " << i << " with " << this->numRanks
            << " ranks must have at least one valid parent or child connection";
      }
    }
  }
}

TEST_F(CommsMonitorDist, testCommSplit) {
  NcclCommRAII origComm{this->globalRank, this->numRanks, this->localRank};

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 1);

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  const std::string commDest = "split_comm";
  config.commDesc = commDest.c_str();

  ncclComm_t splitComm;

  ncclCommSplit(origComm, globalRank % 2, globalRank, &splitComm, &config);

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 2);

  ncclCommDestroy(splitComm);

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 2);
}

TEST_F(CommsMonitorDist, testCommSplitNoColor) {
  NcclCommRAII origComm{this->globalRank, this->numRanks, this->localRank};

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 1);

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  const std::string commDest = "split_comm";
  config.commDesc = commDest.c_str();

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
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};

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
  NcclCommRAII comm1{this->globalRank, this->numRanks, this->localRank};
  NcclCommRAII comm2{this->globalRank, this->numRanks, this->localRank};
  NcclCommRAII comm3{this->globalRank, this->numRanks, this->localRank};
  NcclCommRAII comm4{this->globalRank, this->numRanks, this->localRank};

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

TEST_F(CommsMonitorDist, testMultipleCommsWithLazySetupChannels) {
  auto lazyGuard = EnvRAII{NCCL_LAZY_SETUP_CHANNELS, true};

  NcclCommRAII comm1{this->globalRank, this->numRanks, this->localRank};
  NcclCommRAII comm2{this->globalRank, this->numRanks, this->localRank};
  NcclCommRAII comm3{this->globalRank, this->numRanks, this->localRank};
  NcclCommRAII comm4{this->globalRank, this->numRanks, this->localRank};

  EXPECT_EQ(CommsMonitor::getNumOfCommMonitoring(), 4);
}
TEST_F(CommsMonitorDist, testNcclTopoInfoFromSplitCommWithLazySetupChannel) {
  auto lazyGuard = EnvRAII{NCCL_LAZY_SETUP_CHANNELS, true};

  NcclCommRAII origComm{this->globalRank, this->numRanks, this->localRank};

  // Get topology info from original communicator
  auto origTopoInfo = NcclTopoInfo::fromNcclComm(origComm);

  // Create a split communicator
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  const std::string commDest = "split_comm_topo_test";
  config.commDesc = commDest.c_str();

  ncclComm_t splitComm;
  ncclCommSplit(origComm, globalRank % 2, globalRank, &splitComm, &config);

  // Get topology info from split communicator
  auto splitTopoInfo = NcclTopoInfo::fromNcclComm(splitComm);

  // Split communicator should have valid topology
  EXPECT_GE(splitTopoInfo.nChannels, 0);
  EXPECT_EQ(splitTopoInfo.rings.size(), splitTopoInfo.nChannels);
  EXPECT_EQ(splitTopoInfo.trees.size(), splitTopoInfo.nChannels);

  // If split communicator has channels, verify structure
  if (splitTopoInfo.nChannels > 0) {
    for (size_t i = 0; i < splitTopoInfo.nChannels; i++) {
      // Split communicator should have fewer ranks than original
      EXPECT_LE(splitTopoInfo.rings[i].size(), origTopoInfo.rings[i].size());

      // Verify all ranks in split are valid
      for (int rank : splitTopoInfo.rings[i]) {
        EXPECT_GE(rank, 0);
        // Split comm has its own rank space, so we can't directly compare
        // with original comm's numRanks
      }
    }
  }

  ncclCommDestroy(splitComm);
}

TEST_F(CommsMonitorDist, testNcclTopoInfoTreeStructureWithLazySetupChannels) {
  auto lazyGuard = EnvRAII{NCCL_LAZY_SETUP_CHANNELS, true};

  // Only run this test if we have at least 2 ranks
  if (this->numRanks < 2) {
    GTEST_SKIP() << "Skipping tree structure test with less than 2 ranks";
  }

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  auto topoInfo = NcclTopoInfo::fromNcclComm(comm);

  if (topoInfo.nChannels > 0) {
    for (size_t i = 0; i < topoInfo.nChannels; i++) {
      const auto& tree = topoInfo.trees[i];

      // Collect all valid child nodes
      std::vector<int> children;
      for (int j = 0; j < NCCL_MAX_TREE_ARITY; j++) {
        if (tree.childrenNodes[j] >= 0) {
          children.push_back(tree.childrenNodes[j]);
          // Child should be a valid rank
          EXPECT_LT(tree.childrenNodes[j], this->numRanks)
              << "Invalid child rank " << tree.childrenNodes[j] << " in tree "
              << i;
        }
      }

      // No duplicate children
      std::set<int> uniqueChildren(children.begin(), children.end());
      EXPECT_EQ(uniqueChildren.size(), children.size())
          << "Tree " << i << " has duplicate children";

      // If comm has more than 1 rank, at least one of parentNode or one child
      // should be valid (not -1) to form a connected tree
      if (this->numRanks > 1) {
        bool hasValidParent = (tree.parentNode >= 0);
        bool hasValidChild = std::ranges::any_of(
            tree.childrenNodes, [](auto num) { return num >= 0; });
        EXPECT_TRUE(hasValidParent || hasValidChild)
            << "Tree " << i << " with " << this->numRanks
            << " ranks must have at least one valid parent or child connection";
      }
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
