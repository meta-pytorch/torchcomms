// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllReduceTest.hpp"

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

TEST_P(AllReduceTest, SyncAllReduce) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSyncAllReduce(count, dtype, op);
}

TEST_P(AllReduceTest, SyncAllReduceNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSyncAllReduceNoWork(count, dtype, op);
}

TEST_P(AllReduceTest, AsyncAllReduce) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testAsyncAllReduce(count, dtype, op);
}

TEST_P(AllReduceTest, AsyncAllReduceEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testAsyncAllReduceEarlyReset(count, dtype, op);
}

TEST_P(AllReduceTest, AllReduceInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testAllReduceInputDeleted(count, dtype, op);
}

TEST_P(AllReduceTest, GraphAllReduce) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testGraphAllReduce(count, dtype, op);
}

TEST_P(AllReduceTest, GraphAllReduceInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testGraphAllReduceInputDeleted(count, dtype, op);
}

INSTANTIATE_TEST_SUITE_P(
    AllReduceTestParams,
    AllReduceTest,
    ::testing::Combine(
        ::testing::Values(0, 4, 1024, 1024 * 1024),
        ::testing::Values(at::kFloat, at::kInt, at::kChar),
        ::testing::Values(
            torch::comms::ReduceOp::SUM,
            torch::comms::ReduceOp::MAX,
            torch::comms::ReduceOp::AVG)),
    [](const ::testing::TestParamInfo<
        std::tuple<int, at::ScalarType, torch::comms::ReduceOp>>& info) {
      int count = std::get<0>(info.param);
      at::ScalarType dtype = std::get<1>(info.param);
      torch::comms::ReduceOp op = std::get<2>(info.param);
      return "Count_" + std::to_string(count) + "_" + getDtypeName(dtype) +
          "_" + getOpName(op);
    });

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
