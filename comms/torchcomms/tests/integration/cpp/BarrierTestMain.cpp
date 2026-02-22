// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "BarrierTest.hpp"

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

using Eager = BarrierTest<EagerTestFixture<BarrierParams>>;
using SingleGraph = BarrierTest<GraphTestFixture<BarrierParams, 1>>;
using MultiGraph = BarrierTest<GraphTestFixture<BarrierParams, 2>>;

TEST_P(Eager, Sync) {
  testSync();
}

TEST_P(Eager, SyncNoWork) {
  testSyncNoWork();
}

TEST_P(Eager, Async) {
  testAsync();
}

TEST_P(Eager, AsyncEarlyReset) {
  testAsyncEarlyReset();
}

TEST_P(SingleGraph, Sync) {
  testSync();
}

TEST_P(SingleGraph, SyncNoWork) {
  testSyncNoWork();
}

TEST_P(SingleGraph, Async) {
  testAsync();
}

TEST_P(MultiGraph, Sync) {
  testSync();
}

TEST_P(MultiGraph, SyncNoWork) {
  testSyncNoWork();
}

TEST_P(MultiGraph, Async) {
  testAsync();
}

auto barrierParamNamer(const ::testing::TestParamInfo<BarrierParams>&) {
  return std::string("Default");
}

INSTANTIATE_TEST_SUITE_P(
    Barrier,
    Eager,
    ::testing::Values(0),
    barrierParamNamer);

INSTANTIATE_TEST_SUITE_P(
    Barrier,
    SingleGraph,
    ::testing::Values(0),
    barrierParamNamer);

INSTANTIATE_TEST_SUITE_P(
    Barrier,
    MultiGraph,
    ::testing::Values(0),
    barrierParamNamer);

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
