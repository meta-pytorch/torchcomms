// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <gtest/gtest.h>
#include "comms/ctran/window/CtranWin.h"
#include "comms/ctran/window/Types.h"

using ctran::CtranWin;
using ctran::window::RemWinInfo;

// MockCtranComm is needed to test CtranWin, which requires a comm object.
// CtranWin accesses comm->stateX to initialize signal counters.
class MockCtranComm : public CtranComm {
 public:
  MockCtranComm() {
    std::vector<ncclx::RankTopology> rankTopologies;
    std::vector<int> commRanksToWorldRanks = {0};
    statex_ = std::make_unique<ncclx::CommStateX>(
        0, 1, 0, 0, 0, 0, rankTopologies, commRanksToWorldRanks, "mock_comm");
  }
};

// Multi-rank mock for testing asymmetric scenarios
class MockCtranCommMultiRank : public CtranComm {
 public:
  explicit MockCtranCommMultiRank(int rank, int nRanks) {
    std::vector<ncclx::RankTopology> rankTopologies;
    std::vector<int> commRanksToWorldRanks;
    for (int i = 0; i < nRanks; i++) {
      commRanksToWorldRanks.push_back(i);
    }
    statex_ = std::make_unique<ncclx::CommStateX>(
        rank,
        nRanks,
        0,
        0,
        0,
        0,
        rankTopologies,
        commRanksToWorldRanks,
        "mock_comm_multi");
  }
};

TEST(CtranWinUT, OpCount) {
  auto dummyComm = std::make_unique<MockCtranComm>();

  const size_t size = 8192;
  auto win = std::make_unique<CtranWin>(dummyComm.get(), size);

  ctran::window::OpCountType opType = ctran::window::OpCountType::kPut;
  auto opCount = win->updateOpCount(8, opType);
  EXPECT_EQ(opCount, 0);

  // Expect increased opCount per query for a given rank
  for (int x = 0; x < 5; x++) {
    opCount = win->updateOpCount(8, opType);
    EXPECT_EQ(opCount, 1 + x);
  }

  // Expect opCount starts from 0 for another rank
  opCount = win->updateOpCount(9, opType);
  EXPECT_EQ(opCount, 0);

  // Expect opCount starts from 0 for another OpType
  opType = ctran::window::OpCountType::kWaitSignal;
  opCount = win->updateOpCount(8, opType);
  EXPECT_EQ(opCount, 0);

  // Expect winScope opCount being tracked separately, and starts from 0
  auto winOpCount = win->updateOpCount(8);
  EXPECT_EQ(winOpCount, 0);
}

// Test RemWinInfo struct initialization and dataBytes field
TEST(CtranWinUT, RemWinInfoDataBytes) {
  RemWinInfo info;

  // Verify default initialization
  EXPECT_EQ(info.dataAddr, nullptr);
  EXPECT_EQ(info.signalAddr, nullptr);
  EXPECT_EQ(info.dataBytes, 0);

  // Set and verify dataBytes
  info.dataBytes = 4096;
  EXPECT_EQ(info.dataBytes, 4096);

  info.dataBytes = 1024 * 1024; // 1MB
  EXPECT_EQ(info.dataBytes, 1024 * 1024);
}

// Test CtranWin dataBytes initialization
TEST(CtranWinUT, WindowDataBytesInit) {
  auto dummyComm = std::make_unique<MockCtranComm>();

  const size_t size = 8192;
  auto win = std::make_unique<CtranWin>(dummyComm.get(), size);

  // Verify dataBytes is set correctly in constructor
  EXPECT_EQ(win->dataBytes, size);
}

// Test CtranWin with different sizes
TEST(CtranWinUT, WindowDataBytesDifferentSizes) {
  auto dummyComm = std::make_unique<MockCtranComm>();

  // Test with various sizes
  std::vector<size_t> testSizes = {
      1024, // 1KB
      4096, // 4KB
      1024 * 1024, // 1MB
      16 * 1024 * 1024, // 16MB
  };

  for (size_t size : testSizes) {
    auto win = std::make_unique<CtranWin>(dummyComm.get(), size);
    EXPECT_EQ(win->dataBytes, size) << "Failed for size: " << size;
  }
}

// Test getDataSize with empty remWinInfo (before exchange)
TEST(CtranWinUT, GetDataSizeBeforeExchange) {
  auto dummyComm = std::make_unique<MockCtranComm>();

  const size_t size = 8192;
  auto win = std::make_unique<CtranWin>(dummyComm.get(), size);

  // Before exchange, remWinInfo is empty, getDataSize should return 0
  EXPECT_EQ(win->getDataSize(0), 0);
  EXPECT_EQ(win->getDataSize(1), 0);
  EXPECT_EQ(win->getDataSize(-1), 0); // invalid rank
}

// Test getDataSize with manually populated remWinInfo
TEST(CtranWinUT, GetDataSizeWithRemWinInfo) {
  auto dummyComm = std::make_unique<MockCtranCommMultiRank>(0, 4);

  const size_t size = 8192;
  auto win = std::make_unique<CtranWin>(dummyComm.get(), size);

  // Manually populate remWinInfo to simulate post-exchange state
  win->remWinInfo.resize(4);
  win->remWinInfo[0].dataBytes = 1024;
  win->remWinInfo[1].dataBytes = 2048;
  win->remWinInfo[2].dataBytes = 4096;
  win->remWinInfo[3].dataBytes = 8192;

  // Verify getDataSize returns correct values
  EXPECT_EQ(win->getDataSize(0), 1024);
  EXPECT_EQ(win->getDataSize(1), 2048);
  EXPECT_EQ(win->getDataSize(2), 4096);
  EXPECT_EQ(win->getDataSize(3), 8192);

  // Invalid ranks should return 0
  EXPECT_EQ(win->getDataSize(-1), 0);
  EXPECT_EQ(win->getDataSize(4), 0);
  EXPECT_EQ(win->getDataSize(100), 0);
}

// Test asymmetric window scenario with arbitrary sizes per rank (from vector)
TEST(CtranWinUT, AsymmetricWindowSizesFromVector) {
  const int nRanks = 8;
  auto dummyComm = std::make_unique<MockCtranCommMultiRank>(0, nRanks);

  // Define arbitrary sizes for each rank (not following any formula)
  const std::vector<size_t> rankSizes = {
      4096, // rank 0: 4KB
      16384, // rank 1: 16KB
      8192, // rank 2: 8KB
      32768, // rank 3: 32KB
      2048, // rank 4: 2KB
      65536, // rank 5: 64KB
      12288, // rank 6: 12KB
      24576, // rank 7: 24KB
  };

  // Simulate rank 0's local size from the vector
  const size_t localSize = rankSizes[0];
  auto win = std::make_unique<CtranWin>(dummyComm.get(), localSize);

  // Simulate post-exchange state: populate remWinInfo from the vector
  win->remWinInfo.resize(nRanks);
  for (int r = 0; r < nRanks; r++) {
    win->remWinInfo[r].dataBytes = rankSizes[r];
  }

  // Verify each rank has correct size from the vector
  for (int r = 0; r < nRanks; r++) {
    EXPECT_EQ(win->getDataSize(r), rankSizes[r]) << "Mismatch for rank " << r;
  }

  // Verify sizes are actually different (asymmetric)
  EXPECT_NE(win->getDataSize(0), win->getDataSize(1));
  EXPECT_NE(win->getDataSize(1), win->getDataSize(2));
  EXPECT_NE(win->getDataSize(3), win->getDataSize(4));
}

// Test RemWinInfo vector operations (simulating exchange)
TEST(CtranWinUT, RemWinInfoVectorOperations) {
  std::vector<RemWinInfo> remWinInfo;

  // Resize and populate (similar to exchange())
  const int nRanks = 4;
  remWinInfo.resize(nRanks);

  std::vector<size_t> sizes = {1024, 2048, 3072, 4096};
  for (int r = 0; r < nRanks; r++) {
    remWinInfo[r].dataBytes = sizes[r];
    remWinInfo[r].dataAddr = reinterpret_cast<void*>(0x1000 + r * 0x1000);
  }

  // Verify all entries
  for (int r = 0; r < nRanks; r++) {
    EXPECT_EQ(remWinInfo[r].dataBytes, sizes[r]);
    EXPECT_NE(remWinInfo[r].dataAddr, nullptr);
  }
}
