// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <tuple>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/tests/integration/cpp/GraphTestFixtures.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

using ReduceScatterSingleParams =
    std::tuple<int, at::ScalarType, torch::comms::ReduceOp>;

class ReduceScatterSingleHelper {
 public:
  at::Tensor createInputTensor(
      int count,
      at::ScalarType dtype,
      c10::DeviceType deviceType,
      int numRanks);

  at::Tensor createOutputTensor(
      int count,
      at::ScalarType dtype,
      c10::DeviceType deviceType);

  int calculateExpectedResult(
      const torch::comms::ReduceOp& op,
      int rank,
      int numRanks);

  void verifyResults(
      const at::Tensor& output,
      const torch::comms::ReduceOp& op,
      int rank,
      int numRanks);
};

template <typename Fixture>
class ReduceScatterSingleTest : public Fixture {
 protected:
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

  ReduceScatterSingleHelper helper_;
};
