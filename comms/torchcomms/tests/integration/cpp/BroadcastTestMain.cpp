// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "BroadcastTest.hpp"

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

TEST_P(BroadcastTest, SyncBroadcast) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncBroadcast(count, dtype);
}

TEST_P(BroadcastTest, SyncBroadcastNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncBroadcastNoWork(count, dtype);
}

TEST_P(BroadcastTest, AsyncBroadcast) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncBroadcast(count, dtype);
}

TEST_P(BroadcastTest, AsyncBroadcastEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncBroadcastEarlyReset(count, dtype);
}

TEST_P(BroadcastTest, BroadcastInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testBroadcastInputDeleted(count, dtype);
}

TEST_P(BroadcastTest, GraphBroadcast) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphBroadcast(count, dtype);
}

TEST_P(BroadcastTest, GraphBroadcastInputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testGraphBroadcastInputDeleted(count, dtype);
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastTestParams,
    BroadcastTest,
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
