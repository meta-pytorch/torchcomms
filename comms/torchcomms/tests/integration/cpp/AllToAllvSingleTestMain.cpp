// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllToAllvSingleTest.hpp"

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "TorchCommTestHelpers.h"

using Eager = AllToAllvSingleTest<EagerTestFixture<AllToAllvSingleParams>>;
using SingleGraph =
    AllToAllvSingleTest<GraphTestFixture<AllToAllvSingleParams, 1>>;
using MultiGraph =
    AllToAllvSingleTest<GraphTestFixture<AllToAllvSingleParams, 2>>;

TEST_P(Eager, Sync) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testSync(pattern, count, dtype);
}

TEST_P(Eager, SyncNoWork) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testSyncNoWork(pattern, count, dtype);
}

TEST_P(Eager, Async) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testAsync(pattern, count, dtype);
}

TEST_P(Eager, AsyncEarlyReset) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testAsyncEarlyReset(pattern, count, dtype);
}

TEST_P(Eager, InputDeleted) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testInputDeleted(pattern, count, dtype);
}

TEST_P(Eager, MultiDimTensor) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testMultiDimTensor(pattern, count, dtype);
}

TEST_P(SingleGraph, Sync) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testSync(pattern, count, dtype);
}

TEST_P(SingleGraph, SyncNoWork) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testSyncNoWork(pattern, count, dtype);
}

TEST_P(SingleGraph, Async) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testAsync(pattern, count, dtype);
}

TEST_P(SingleGraph, InputDeleted) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testInputDeleted(pattern, count, dtype);
}

TEST_P(MultiGraph, Sync) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testSync(pattern, count, dtype);
}

TEST_P(MultiGraph, SyncNoWork) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testSyncNoWork(pattern, count, dtype);
}

TEST_P(MultiGraph, Async) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testAsync(pattern, count, dtype);
}

TEST_P(MultiGraph, InputDeleted) {
  AllToAllvSizePattern pattern = std::get<0>(GetParam());
  int count = std::get<1>(GetParam());
  at::ScalarType dtype = std::get<2>(GetParam());
  testInputDeleted(pattern, count, dtype);
}

auto allToAllvSingleParamValues() {
  static std::vector<AllToAllvSingleParams> params = []() {
    std::vector<AllToAllvSingleParams> p;
    std::vector<at::ScalarType> dtypes = {at::kFloat, at::kInt, at::kChar};

    // Uniform pattern: full dtype coverage with two counts
    for (int count : {4, 1024}) {
      for (auto dtype : dtypes) {
        p.emplace_back(AllToAllvSizePattern::Uniform, count, dtype);
      }
    }
    // Other patterns: minimal config (count=4, Float only) for pattern coverage
    p.emplace_back(AllToAllvSizePattern::Variable, 4, at::kFloat);
    p.emplace_back(AllToAllvSizePattern::ZeroSizes, 4, at::kFloat);
    p.emplace_back(AllToAllvSizePattern::AllZero, 0, at::kFloat);
    p.emplace_back(AllToAllvSizePattern::Asymmetric, 4, at::kFloat);
    return p;
  }();
  return ::testing::ValuesIn(params);
}

auto allToAllvSingleGraphParamValues() {
  return ::testing::Values(
      AllToAllvSingleParams{AllToAllvSizePattern::Uniform, 4, at::kFloat});
}

auto allToAllvSingleParamNamer(
    const ::testing::TestParamInfo<AllToAllvSingleParams>& info) {
  AllToAllvSizePattern pattern = std::get<0>(info.param);
  int count = std::get<1>(info.param);
  at::ScalarType dtype = std::get<2>(info.param);
  return Eager::getPatternName(pattern) + "_Count_" + std::to_string(count) +
      "_" + getDtypeName(dtype);
}

INSTANTIATE_TEST_SUITE_P(
    AllToAllvSingle,
    Eager,
    allToAllvSingleParamValues(),
    allToAllvSingleParamNamer);

INSTANTIATE_TEST_SUITE_P(
    AllToAllvSingle,
    SingleGraph,
    allToAllvSingleGraphParamValues(),
    allToAllvSingleParamNamer);

INSTANTIATE_TEST_SUITE_P(
    AllToAllvSingle,
    MultiGraph,
    allToAllvSingleGraphParamValues(),
    allToAllvSingleParamNamer);

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
