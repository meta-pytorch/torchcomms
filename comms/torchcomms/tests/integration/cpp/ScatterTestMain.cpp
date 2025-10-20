// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ScatterTest.hpp"

#include <gtest/gtest.h>
#include <vector>
#include "TorchCommTestHelpers.h"

TEST_P(ScatterTest, SyncScatter) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncScatter(count, dtype);
}

TEST_P(ScatterTest, SyncScatterNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncScatterNoWork(count, dtype);
}

TEST_P(ScatterTest, AsyncScatter) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncScatter(count, dtype);
}

TEST_P(ScatterTest, AsyncScatterEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncScatterEarlyReset(count, dtype);
}

TEST_P(ScatterTest, ScatterInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testScatterInputDeleted(count, dtype);
}

TEST_P(ScatterTest, GraphScatter) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphScatter(count, dtype);
}

TEST_P(ScatterTest, GraphScatterInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphScatterInputDeleted(count, dtype);
}

INSTANTIATE_TEST_SUITE_P(
    ScatterTestParams,
    ScatterTest,
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
