// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ReduceScatterTest.hpp"

#include <gtest/gtest.h>
#include <vector>
#include "TorchCommTestHelpers.h"

TEST_P(ReduceScatterTest, SyncReduceScatter) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSyncReduceScatter(count, dtype, op);
}

TEST_P(ReduceScatterTest, SyncReduceScatterNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSyncReduceScatterNoWork(count, dtype, op);
}

TEST_P(ReduceScatterTest, AsyncReduceScatter) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testAsyncReduceScatter(count, dtype, op);
}

TEST_P(ReduceScatterTest, AsyncReduceScatterEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testAsyncReduceScatterEarlyReset(count, dtype, op);
}

TEST_P(ReduceScatterTest, ReduceScatterInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testReduceScatterInputDeleted(count, dtype, op);
}

TEST_P(ReduceScatterTest, GraphReduceScatter) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testGraphReduceScatter(count, dtype, op);
}

TEST_P(ReduceScatterTest, GraphReduceScatterInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testGraphReduceScatterInputDeleted(count, dtype, op);
}

INSTANTIATE_TEST_SUITE_P(
    ReduceScatterTestParams,
    ReduceScatterTest,
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
