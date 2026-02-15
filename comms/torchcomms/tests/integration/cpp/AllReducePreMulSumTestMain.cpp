// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "AllReduceTest.hpp"

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

using Eager = AllReduceTest<EagerTestFixture<PreMulSumParams>>;
using SingleGraph = AllReduceTest<GraphTestFixture<PreMulSumParams, 1>>;
using MultiGraph = AllReduceTest<GraphTestFixture<PreMulSumParams, 2>>;

#define MAKE_PREMUL_OP(opType, dtype)                                \
  ((opType) == PreMulSumOpType::kTensor                              \
       ? torch::comms::ReduceOp::make_nccl_premul_sum(               \
             helper_.createPreMulFactorTensor((dtype), deviceType_)) \
       : torch::comms::ReduceOp::make_nccl_premul_sum(2.0))

TEST_P(Eager, Sync) {
  auto [count, dtype, opType] = GetParam();
  testSync(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(Eager, SyncNoWork) {
  auto [count, dtype, opType] = GetParam();
  testSyncNoWork(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(Eager, Async) {
  auto [count, dtype, opType] = GetParam();
  testAsync(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(Eager, AsyncEarlyReset) {
  auto [count, dtype, opType] = GetParam();
  testAsyncEarlyReset(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(Eager, InputDeleted) {
  auto [count, dtype, opType] = GetParam();
  testInputDeleted(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(SingleGraph, Sync) {
  auto [count, dtype, opType] = GetParam();
  testSync(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(SingleGraph, SyncNoWork) {
  auto [count, dtype, opType] = GetParam();
  testSyncNoWork(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(SingleGraph, Async) {
  auto [count, dtype, opType] = GetParam();
  testAsync(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(SingleGraph, InputDeleted) {
  auto [count, dtype, opType] = GetParam();
  testInputDeleted(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(MultiGraph, Sync) {
  auto [count, dtype, opType] = GetParam();
  testSync(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(MultiGraph, SyncNoWork) {
  auto [count, dtype, opType] = GetParam();
  testSyncNoWork(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(MultiGraph, Async) {
  auto [count, dtype, opType] = GetParam();
  testAsync(count, dtype, MAKE_PREMUL_OP(opType, dtype));
}

TEST_P(MultiGraph, InputDeleted) {
  auto [count, dtype, opType] = GetParam();
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
  auto [count, dtype, opType] = info.param;
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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
