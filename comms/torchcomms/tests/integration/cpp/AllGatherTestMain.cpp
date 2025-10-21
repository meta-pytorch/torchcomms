// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllGatherTest.hpp"

#include <gtest/gtest.h>

TEST_P(AllGatherTest, SyncAllGather) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncAllGather(count, dtype);
}

TEST_P(AllGatherTest, SyncAllGatherNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncAllGatherNoWork(count, dtype);
}

TEST_P(AllGatherTest, AsyncAllGather) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncAllGather(count, dtype);
}

TEST_P(AllGatherTest, AsyncAllGatherEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncAllGatherEarlyReset(count, dtype);
}

TEST_P(AllGatherTest, AllGatherInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAllGatherInputDeleted(count, dtype);
}

TEST_P(AllGatherTest, GraphAllGather) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphAllGather(count, dtype);
}

TEST_P(AllGatherTest, GraphAllGatherInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphAllGatherInputDeleted(count, dtype);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherTestParams,
    AllGatherTest,
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
