// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "BatchSendRecvTest.hpp"

#include <gtest/gtest.h>

TEST_P(BatchSendRecvTest, SyncBatchSendRecv) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncBatchSendRecv(count, dtype);
}

TEST_P(BatchSendRecvTest, SyncBatchSendRecvNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncBatchSendRecvNoWork(count, dtype);
}

TEST_P(BatchSendRecvTest, AsyncBatchSendRecv) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncBatchSendRecv(count, dtype);
}

TEST_P(BatchSendRecvTest, AsyncBatchSendRecvEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncBatchSendRecvEarlyReset(count, dtype);
}

TEST_P(BatchSendRecvTest, BatchSendRecvInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testBatchSendRecvInputDeleted(count, dtype);
}

TEST_P(BatchSendRecvTest, GraphBatchSendRecv) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphBatchSendRecv(count, dtype);
}

TEST_P(BatchSendRecvTest, GraphBatchSendRecvInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphBatchSendRecvInputDeleted(count, dtype);
}

INSTANTIATE_TEST_SUITE_P(
    BatchSendRecvTestParams,
    BatchSendRecvTest,
    ::testing::Combine(
        ::testing::Values(4),
        ::testing::Values(at::kFloat, at::kInt, at::kChar)),
    [](const ::testing::TestParamInfo<std::tuple<int, at::ScalarType>>& info) {
      int count = std::get<0>(info.param);
      at::ScalarType dtype = std::get<1>(info.param);
      return "Count_" + std::to_string(count) + "_" + getDtypeName(dtype);
    });

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
