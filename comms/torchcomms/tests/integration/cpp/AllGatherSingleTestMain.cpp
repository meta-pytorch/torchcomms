// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllGatherSingleTest.hpp"

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

using Eager = AllGatherSingleTest<EagerTestFixture<AllGatherSingleParams>>;
using SingleGraph =
    AllGatherSingleTest<GraphTestFixture<AllGatherSingleParams, 1>>;
using MultiGraph =
    AllGatherSingleTest<GraphTestFixture<AllGatherSingleParams, 2>>;

TEST_P(Eager, Sync) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSync(count, dtype);
}

TEST_P(Eager, SyncNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncNoWork(count, dtype);
}

TEST_P(Eager, Async) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsync(count, dtype);
}

TEST_P(Eager, AsyncEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsyncEarlyReset(count, dtype);
}

TEST_P(Eager, InputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testInputDeleted(count, dtype);
}

TEST_P(SingleGraph, Sync) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSync(count, dtype);
}

TEST_P(SingleGraph, SyncNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncNoWork(count, dtype);
}

TEST_P(SingleGraph, Async) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsync(count, dtype);
}

TEST_P(SingleGraph, InputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testInputDeleted(count, dtype);
}

TEST_P(MultiGraph, Sync) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSync(count, dtype);
}

TEST_P(MultiGraph, SyncNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testSyncNoWork(count, dtype);
}

TEST_P(MultiGraph, Async) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testAsync(count, dtype);
}

TEST_P(MultiGraph, InputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  testInputDeleted(count, dtype);
}

auto allGatherSingleParamValues() {
  return ::testing::Combine(
      ::testing::Values(0, 4, 1024, 1024 * 1024),
      ::testing::Values(at::kFloat, at::kInt, at::kChar));
}

auto allGatherSingleGraphParamValues() {
  return ::testing::Combine(
      ::testing::Values(0, 1000, 1024 * 1024), ::testing::Values(at::kFloat));
}

auto allGatherSingleParamNamer(
    const ::testing::TestParamInfo<AllGatherSingleParams>& info) {
  int count = std::get<0>(info.param);
  at::ScalarType dtype = std::get<1>(info.param);
  return "Count_" + std::to_string(count) + "_" + getDtypeName(dtype);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherSingle,
    Eager,
    allGatherSingleParamValues(),
    allGatherSingleParamNamer);

INSTANTIATE_TEST_SUITE_P(
    AllGatherSingle,
    SingleGraph,
    allGatherSingleGraphParamValues(),
    allGatherSingleParamNamer);

INSTANTIATE_TEST_SUITE_P(
    AllGatherSingle,
    MultiGraph,
    allGatherSingleGraphParamValues(),
    allGatherSingleParamNamer);

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
