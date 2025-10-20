// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ReduceScatterSingleTest.hpp"

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

TEST_P(ReduceScatterSingleTest, SyncReduceScatterSingle) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSyncReduceScatterSingle(count, dtype, op);
}

TEST_P(ReduceScatterSingleTest, SyncReduceScatterSingleNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSyncReduceScatterSingleNoWork(count, dtype, op);
}

TEST_P(ReduceScatterSingleTest, AsyncReduceScatterSingle) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testAsyncReduceScatterSingle(count, dtype, op);
}

TEST_P(ReduceScatterSingleTest, AsyncReduceScatterSingleEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testAsyncReduceScatterSingleEarlyReset(count, dtype, op);
}

TEST_P(ReduceScatterSingleTest, ReduceScatterSingleInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testReduceScatterSingleInputDeleted(count, dtype, op);
}

TEST_P(ReduceScatterSingleTest, GraphReduceScatterSingle) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testGraphReduceScatterSingle(count, dtype, op);
}

TEST_P(ReduceScatterSingleTest, GraphReduceScatterSingleInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testGraphReduceScatterSingleInputDeleted(count, dtype, op);
}

INSTANTIATE_TEST_SUITE_P(
    ReduceScatterSingleTestParams,
    ReduceScatterSingleTest,
    ::testing::Combine(
        ::testing::Values(0, 4, 1024, 1024 * 1024),
        ::testing::Values(at::kFloat, at::kInt, at::kChar),
        ::testing::Values(
            torch::comms::ReduceOp::SUM,
            torch::comms::ReduceOp::MAX)),
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
