// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "GatherTest.hpp"

#include <gtest/gtest.h>

TEST_P(GatherTest, SyncGather) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncGather(count, dtype);
}

TEST_P(GatherTest, SyncGatherNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncGatherNoWork(count, dtype);
}

TEST_P(GatherTest, AsyncGather) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncGather(count, dtype);
}

TEST_P(GatherTest, AsyncGatherEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncGatherEarlyReset(count, dtype);
}

TEST_P(GatherTest, GatherInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGatherInputDeleted(count, dtype);
}

TEST_P(GatherTest, GraphGather) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphGather(count, dtype);
}

TEST_P(GatherTest, GraphGatherInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphGatherInputDeleted(count, dtype);
}

INSTANTIATE_TEST_SUITE_P(
    GatherTestParams,
    GatherTest,
    ::testing::Combine(
        ::testing::Values(0, 4, 1024, 1024 * 1024),
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
