// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ReduceTest.hpp"

#include <gtest/gtest.h>
#include <vector>
#include "TorchCommTestHelpers.h"

TEST_P(ReduceTest, SyncReduce) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSyncReduce(count, dtype, op);
}

TEST_P(ReduceTest, SyncReduceNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSyncReduceNoWork(count, dtype, op);
}

TEST_P(ReduceTest, AsyncReduce) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testAsyncReduce(count, dtype, op);
}

TEST_P(ReduceTest, AsyncReduceEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testAsyncReduceEarlyReset(count, dtype, op);
}

TEST_P(ReduceTest, ReduceInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testReduceInputDeleted(count, dtype, op);
}

TEST_P(ReduceTest, GraphReduce) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testGraphReduce(count, dtype, op);
}

TEST_P(ReduceTest, GraphReduceInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testGraphReduceInputDeleted(count, dtype, op);
}

INSTANTIATE_TEST_SUITE_P(
    ReduceTestParams,
    ReduceTest,
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
