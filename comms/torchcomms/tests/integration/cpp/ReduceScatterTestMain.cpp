// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ReduceScatterTest.hpp"

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

using Eager = ReduceScatterTest<EagerTestFixture<ReduceScatterParams>>;
using SingleGraph = ReduceScatterTest<GraphTestFixture<ReduceScatterParams, 1>>;
using MultiGraph = ReduceScatterTest<GraphTestFixture<ReduceScatterParams, 2>>;

TEST_P(Eager, Sync) {
  auto [count, dtype, op] = GetParam();
  testSync(count, dtype, op);
}

TEST_P(Eager, SyncNoWork) {
  auto [count, dtype, op] = GetParam();
  testSyncNoWork(count, dtype, op);
}

TEST_P(Eager, Async) {
  auto [count, dtype, op] = GetParam();
  testAsync(count, dtype, op);
}

TEST_P(Eager, AsyncEarlyReset) {
  auto [count, dtype, op] = GetParam();
  testAsyncEarlyReset(count, dtype, op);
}

TEST_P(Eager, InputDeleted) {
  auto [count, dtype, op] = GetParam();
  testInputDeleted(count, dtype, op);
}

TEST_P(SingleGraph, Sync) {
  auto [count, dtype, op] = GetParam();
  testSync(count, dtype, op);
}

TEST_P(SingleGraph, SyncNoWork) {
  auto [count, dtype, op] = GetParam();
  testSyncNoWork(count, dtype, op);
}

TEST_P(SingleGraph, Async) {
  auto [count, dtype, op] = GetParam();
  testAsync(count, dtype, op);
}

TEST_P(SingleGraph, InputDeleted) {
  auto [count, dtype, op] = GetParam();
  testInputDeleted(count, dtype, op);
}

TEST_P(MultiGraph, Sync) {
  auto [count, dtype, op] = GetParam();
  testSync(count, dtype, op);
}

TEST_P(MultiGraph, SyncNoWork) {
  auto [count, dtype, op] = GetParam();
  testSyncNoWork(count, dtype, op);
}

TEST_P(MultiGraph, Async) {
  auto [count, dtype, op] = GetParam();
  testAsync(count, dtype, op);
}

TEST_P(MultiGraph, InputDeleted) {
  auto [count, dtype, op] = GetParam();
  testInputDeleted(count, dtype, op);
}

auto reduceScatterParamValues() {
  return ::testing::Combine(
      ::testing::Values(0, 4, 1024, 1024 * 1024),
      ::testing::Values(at::kFloat, at::kInt, at::kChar),
      ::testing::Values(
          torch::comms::ReduceOp::SUM,
          torch::comms::ReduceOp::MAX,
          torch::comms::ReduceOp::AVG));
}

auto reduceScatterGraphParamValues() {
  return ::testing::Combine(
      ::testing::Values(0, 1000, 1024 * 1024),
      ::testing::Values(at::kFloat),
      ::testing::Values(torch::comms::ReduceOp::SUM));
}

auto reduceScatterParamNamer(
    const ::testing::TestParamInfo<ReduceScatterParams>& info) {
  auto [count, dtype, op] = info.param;
  return "Count_" + std::to_string(count) + "_" + getDtypeName(dtype) + "_" +
      getOpName(op);
}

INSTANTIATE_TEST_SUITE_P(
    ReduceScatter,
    Eager,
    reduceScatterParamValues(),
    reduceScatterParamNamer);

INSTANTIATE_TEST_SUITE_P(
    ReduceScatter,
    SingleGraph,
    reduceScatterGraphParamValues(),
    reduceScatterParamNamer);

INSTANTIATE_TEST_SUITE_P(
    ReduceScatter,
    MultiGraph,
    reduceScatterGraphParamValues(),
    reduceScatterParamNamer);

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
