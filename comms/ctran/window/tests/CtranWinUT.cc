// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <gtest/gtest.h>
#include "comms/ctran/window/CtranWin.h"

using ctran::CtranWin;
TEST(CtranWinUT, OpCount) {
  CtranComm* dummyComm = nullptr;
  const size_t size = 8192;
  auto win =
      std::make_unique<CtranWin>(dummyComm, size, NCCL_CTRAN_WIN_SIGNAL_SIZE);

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
