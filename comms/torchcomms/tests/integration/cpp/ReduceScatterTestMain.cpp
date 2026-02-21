// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ReduceScatterTest.hpp"

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

using Eager = ReduceScatterTest<EagerTestFixture<ReduceScatterParams>>;
using SingleGraph = ReduceScatterTest<GraphTestFixture<ReduceScatterParams, 1>>;
using MultiGraph = ReduceScatterTest<GraphTestFixture<ReduceScatterParams, 2>>;

TEST_P(Eager, Sync) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSync(count, dtype, op);
}

TEST_P(Eager, SyncNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSyncNoWork(count, dtype, op);
}

TEST_P(Eager, Async) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testAsync(count, dtype, op);
}

TEST_P(Eager, AsyncEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testAsyncEarlyReset(count, dtype, op);
}

TEST_P(Eager, InputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testInputDeleted(count, dtype, op);
}

TEST_P(SingleGraph, Sync) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSync(count, dtype, op);
}

TEST_P(SingleGraph, SyncNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSyncNoWork(count, dtype, op);
}

TEST_P(SingleGraph, Async) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testAsync(count, dtype, op);
}

TEST_P(SingleGraph, InputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testInputDeleted(count, dtype, op);
}

TEST_P(MultiGraph, Sync) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSync(count, dtype, op);
}

TEST_P(MultiGraph, SyncNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testSyncNoWork(count, dtype, op);
}

TEST_P(MultiGraph, Async) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
  testAsync(count, dtype, op);
}

TEST_P(MultiGraph, InputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  torch::comms::ReduceOp op = std::get<2>(GetParam());
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
  int count = std::get<0>(info.param);
  at::ScalarType dtype = std::get<1>(info.param);
  torch::comms::ReduceOp op = std::get<2>(info.param);
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

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
