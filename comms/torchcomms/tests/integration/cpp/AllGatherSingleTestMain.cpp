// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllGatherSingleTest.hpp"

#include <gtest/gtest.h>

TEST_P(AllGatherSingleTest, SyncAllGatherSingle) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncAllGatherSingle(count, dtype);
}

TEST_P(AllGatherSingleTest, SyncAllGatherSingleNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncAllGatherSingleNoWork(count, dtype);
}

TEST_P(AllGatherSingleTest, AsyncAllGatherSingle) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncAllGatherSingle(count, dtype);
}

TEST_P(AllGatherSingleTest, AsyncAllGatherSingleEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncAllGatherSingleEarlyReset(count, dtype);
}

TEST_P(AllGatherSingleTest, AllGatherSingleInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAllGatherSingleInputDeleted(count, dtype);
}

TEST_P(AllGatherSingleTest, GraphAllGatherSingle) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphAllGatherSingle(count, dtype);
}

TEST_P(AllGatherSingleTest, GraphAllGatherSingleInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphAllGatherSingleInputDeleted(count, dtype);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherSingleTestParams,
    AllGatherSingleTest,
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
