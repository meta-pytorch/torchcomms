// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <gtest/gtest.h>
#include <memory>
#include <tuple>
#include "comms/torchcomms/tests/integration/cpp/GraphTestFixtures.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

using AllReduceParams = std::tuple<int, at::ScalarType, torch::comms::ReduceOp>;

enum class PreMulSumOpType { kScalar, kTensor };
using PreMulSumParams = std::tuple<int, at::ScalarType, PreMulSumOpType>;

template <typename Fixture>
class AllReduceTest : public Fixture {
 protected:
  using Fixture::device_type_;
  using Fixture::num_ranks_;
  using Fixture::rank_;
  using Fixture::run;
  using Fixture::torchcomm_;

  void
  testSync(int count, at::ScalarType dtype, const torch::comms::ReduceOp& op);
  void testSyncNoWork(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void
  testAsync(int count, at::ScalarType dtype, const torch::comms::ReduceOp& op);
  void testAsyncEarlyReset(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void testInputDeleted(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);

  // Helper function declarations with parameters
  virtual at::Tensor createInputTensor(int count, at::ScalarType dtype);
  virtual at::Tensor createPreMulFactorTensor(at::ScalarType dtype);
  double calculateExpectedResult(const torch::comms::ReduceOp& op);
  void verifyResults(const at::Tensor& input, const torch::comms::ReduceOp& op);
};
