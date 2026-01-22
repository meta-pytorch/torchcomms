// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ReduceScatterVTest.hpp"

#include <gtest/gtest.h>
#include <vector>
#include "TorchCommTestHelpers.h"

TEST_P(ReduceScatterVTest, SyncReduceScatterV) {
  auto backend = std::string(getenv("TEST_BACKEND"));
  if (backend != "ncclx") {
    GTEST_SKIP() << "skip reduce_scatter_v test for non-NCCLX backends";
  }
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSyncReduceScatterV(count, dtype, op);
}

INSTANTIATE_TEST_SUITE_P(
    ReduceScatterVTestParams,
    ReduceScatterVTest,
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
