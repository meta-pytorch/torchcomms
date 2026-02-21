// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <gtest/gtest.h>
#include <memory>
#include <tuple>
#include <vector>
#include "comms/torchcomms/tests/integration/cpp/GraphTestFixtures.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

using ReduceScatterVParams =
    std::tuple<int, at::ScalarType, torch::comms::ReduceOp>;

template <typename Fixture>
class ReduceScatterVTest : public Fixture {
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

  std::vector<int> getCounts(int count);

  virtual std::vector<at::Tensor> createInputTensors(
      const std::vector<int>& counts,
      at::ScalarType dtype);

  virtual at::Tensor createOutputTensor(int count, at::ScalarType dtype);

  int calculateExpectedResult(const torch::comms::ReduceOp& op);

  void verifyResults(
      const at::Tensor& output,
      const torch::comms::ReduceOp& op);
};
