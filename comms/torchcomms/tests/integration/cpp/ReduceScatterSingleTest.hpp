// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <gtest/gtest.h>
#include <memory>
#include <tuple>
#include "comms/torchcomms/tests/integration/cpp/GraphTestFixtures.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

using ReduceScatterSingleParams =
    std::tuple<int, at::ScalarType, torch::comms::ReduceOp>;

template <typename Fixture>
class ReduceScatterSingleTest : public Fixture {
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

  virtual at::Tensor createInputTensor(int count, at::ScalarType dtype);
  virtual at::Tensor createOutputTensor(int count, at::ScalarType dtype);
  void verifyResults(
      const at::Tensor& output,
      const torch::comms::ReduceOp& op);
};
