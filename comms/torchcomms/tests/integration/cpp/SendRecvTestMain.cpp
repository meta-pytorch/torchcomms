// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "SendRecvTest.hpp"

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

TEST_P(SendRecvTest, SyncSendRecv) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncSendRecv(count, dtype);
}

TEST_P(SendRecvTest, SyncSendRecvNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncSendRecvNoWork(count, dtype);
}

TEST_P(SendRecvTest, AsyncSendRecv) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncSendRecv(count, dtype);
}

TEST_P(SendRecvTest, AsyncSendRecvEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncSendRecvEarlyReset(count, dtype);
}

TEST_P(SendRecvTest, SendRecvInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSendRecvInputDeleted(count, dtype);
}

TEST_P(SendRecvTest, GraphSendRecv) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphSendRecv(count, dtype);
}

TEST_P(SendRecvTest, GraphSendRecvInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphSendRecvInputDeleted(count, dtype);
}

INSTANTIATE_TEST_SUITE_P(
    SendRecvTestParams,
    SendRecvTest,
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
