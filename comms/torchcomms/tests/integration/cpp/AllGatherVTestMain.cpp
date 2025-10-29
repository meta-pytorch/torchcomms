// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllGatherVTest.hpp"

#include <gtest/gtest.h>

TEST_P(AllGatherVTest, SyncAllGatherV) {
  auto backend = std::string(getenv("TEST_BACKEND"));
  if (backend != "ncclx") {
    GTEST_SKIP() << "Skipping all_gather_v test for non-NCCLX backends";
  }
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncAllGatherV(count, dtype);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherVTestParams,
    AllGatherVTest,
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
