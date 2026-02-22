// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "comms/torchcomms/tests/integration/cpp/GraphTestFixtures.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

enum class AllToAllvSizePattern {
  Uniform,
  Variable,
  ZeroSizes,
  AllZero,
  Asymmetric,
};

using AllToAllvSingleParams =
    std::tuple<AllToAllvSizePattern, int, at::ScalarType>;

template <typename Fixture>
class AllToAllvSingleTest : public Fixture {
 protected:
  using Fixture::device_type_;
  using Fixture::num_ranks_;
  using Fixture::rank_;
  using Fixture::run;
  using Fixture::torchcomm_;

  void testSync(AllToAllvSizePattern pattern, int count, at::ScalarType dtype);
  void
  testSyncNoWork(AllToAllvSizePattern pattern, int count, at::ScalarType dtype);
  void testAsync(AllToAllvSizePattern pattern, int count, at::ScalarType dtype);
  void testAsyncEarlyReset(
      AllToAllvSizePattern pattern,
      int count,
      at::ScalarType dtype);
  void testInputDeleted(
      AllToAllvSizePattern pattern,
      int count,
      at::ScalarType dtype);
  void testMultiDimTensor(
      AllToAllvSizePattern pattern,
      int count,
      at::ScalarType dtype);

 public:
  static std::string getPatternName(AllToAllvSizePattern pattern);

 protected:
  std::pair<std::vector<uint64_t>, std::vector<uint64_t>> computeSplitSizes(
      AllToAllvSizePattern pattern,
      int count);
  virtual at::Tensor createInputTensor(
      const std::vector<uint64_t>& input_split_sizes,
      at::ScalarType dtype);
  virtual at::Tensor createOutputTensor(
      const std::vector<uint64_t>& output_split_sizes,
      at::ScalarType dtype);
  void verifyResults(
      const at::Tensor& output,
      const std::vector<uint64_t>& output_split_sizes);
};
