// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllReduceTest.hpp"

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

using Eager = AllReduceTest<EagerTestFixture<PreMulSumParams>>;
using SingleGraph = AllReduceTest<GraphTestFixture<PreMulSumParams, 1>>;
using MultiGraph = AllReduceTest<GraphTestFixture<PreMulSumParams, 2>>;

#define MAKE_PREMUL_OP(opType, dtype)                  \
  ((opType) == PreMulSumOpType::kTensor                \
       ? torch::comms::ReduceOp::make_nccl_premul_sum( \
             createPreMulFactorTensor((dtype)))        \
       : torch::comms::ReduceOp::make_nccl_premul_sum(2.0))

TEST_P(Eager, Sync) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  PreMulSumOpType opType = std::get<2>(GetParam());
  testSync(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(Eager, SyncNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  PreMulSumOpType opType = std::get<2>(GetParam());
  testSyncNoWork(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(Eager, Async) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  PreMulSumOpType opType = std::get<2>(GetParam());
  testAsync(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(Eager, AsyncEarlyReset) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  PreMulSumOpType opType = std::get<2>(GetParam());
  testAsyncEarlyReset(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(Eager, InputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  PreMulSumOpType opType = std::get<2>(GetParam());
  testInputDeleted(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(SingleGraph, Sync) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  PreMulSumOpType opType = std::get<2>(GetParam());
  testSync(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(SingleGraph, SyncNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  PreMulSumOpType opType = std::get<2>(GetParam());
  testSyncNoWork(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(SingleGraph, Async) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  PreMulSumOpType opType = std::get<2>(GetParam());
  testAsync(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(SingleGraph, InputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  PreMulSumOpType opType = std::get<2>(GetParam());
  testInputDeleted(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(MultiGraph, Sync) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  PreMulSumOpType opType = std::get<2>(GetParam());
  testSync(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(MultiGraph, SyncNoWork) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  PreMulSumOpType opType = std::get<2>(GetParam());
  testSyncNoWork(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(MultiGraph, Async) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  PreMulSumOpType opType = std::get<2>(GetParam());
  testAsync(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(MultiGraph, InputDeleted) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  PreMulSumOpType opType = std::get<2>(GetParam());
  testInputDeleted(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

#undef MAKE_PREMUL_OP

auto allReducePreMulSumScalarParams() {
  return ::testing::Combine(
      ::testing::Values(0, 4, 1024, 1024 * 1024),
      ::testing::Values(at::kHalf, at::kFloat, at::kDouble),
      ::testing::Values(PreMulSumOpType::kScalar));
}

auto allReducePreMulSumBf16TensorParams() {
  return ::testing::Combine(
      ::testing::Values(0, 4, 1024, 1024 * 1024),
      ::testing::Values(at::kBFloat16),
      ::testing::Values(PreMulSumOpType::kTensor));
}

auto allReducePreMulSumGraphScalarParams() {
  return ::testing::Combine(
      ::testing::Values(0, 1024, 1024 * 1024),
      ::testing::Values(at::kFloat),
      ::testing::Values(PreMulSumOpType::kScalar));
}

auto allReducePreMulSumGraphBf16TensorParams() {
  return ::testing::Combine(
      ::testing::Values(0, 1024, 1024 * 1024),
      ::testing::Values(at::kBFloat16),
      ::testing::Values(PreMulSumOpType::kTensor));
}

std::string getPreMulSumOpTypeName(PreMulSumOpType opType) {
  switch (opType) {
    case PreMulSumOpType::kScalar:
      return "ScalarPreMulSum";
    case PreMulSumOpType::kTensor:
      return "TensorPreMulSum";
  }
  return "Unknown";
}

auto allReducePreMulSumParamNamer(
    const ::testing::TestParamInfo<PreMulSumParams>& info) {
  int count = std::get<0>(info.param);
  at::ScalarType dtype = std::get<1>(info.param);
  PreMulSumOpType opType = std::get<2>(info.param);
  return "Count_" + std::to_string(count) + "_" + getDtypeName(dtype) + "_" +
      getPreMulSumOpTypeName(opType);
}

INSTANTIATE_TEST_SUITE_P(
    AllReducePreMulSum,
    Eager,
    allReducePreMulSumScalarParams(),
    allReducePreMulSumParamNamer);

INSTANTIATE_TEST_SUITE_P(
    AllReducePreMulSumBf16Tensor,
    Eager,
    allReducePreMulSumBf16TensorParams(),
    allReducePreMulSumParamNamer);

INSTANTIATE_TEST_SUITE_P(
    AllReducePreMulSum,
    SingleGraph,
    allReducePreMulSumGraphScalarParams(),
    allReducePreMulSumParamNamer);

INSTANTIATE_TEST_SUITE_P(
    AllReducePreMulSumBf16Tensor,
    SingleGraph,
    allReducePreMulSumGraphBf16TensorParams(),
    allReducePreMulSumParamNamer);

INSTANTIATE_TEST_SUITE_P(
    AllReducePreMulSum,
    MultiGraph,
    allReducePreMulSumGraphScalarParams(),
    allReducePreMulSumParamNamer);

INSTANTIATE_TEST_SUITE_P(
    AllReducePreMulSumBf16Tensor,
    MultiGraph,
    allReducePreMulSumGraphBf16TensorParams(),
    allReducePreMulSumParamNamer);

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
