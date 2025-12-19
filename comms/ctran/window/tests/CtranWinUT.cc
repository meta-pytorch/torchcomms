// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <gtest/gtest.h>
#include "comms/ctran/window/CtranWin.h"

using ctran::CtranWin;

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
