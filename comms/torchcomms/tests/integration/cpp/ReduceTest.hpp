// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <tuple>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/tests/integration/cpp/GraphTestFixtures.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

using ReduceParams = std::tuple<int, at::ScalarType, torch::comms::ReduceOp>;

// Helper class for Reduce test operations.
// Methods are non-static to allow subclass override if needed.
class ReduceHelper {
 public:
  at::Tensor createInputTensor(
      int count,
      at::ScalarType dtype,
      c10::DeviceType deviceType,
      int rank);

  void synchronizeStream(c10::DeviceType deviceType);

  double calculateExpectedResult(
      const torch::comms::ReduceOp& op,
      int numRanks);

  void verifyResults(
      const at::Tensor& output,
      const torch::comms::ReduceOp& op,
      int rank,
      int numRanks,
      c10::DeviceType deviceType);
};

// Reduce test class template. Fixture determines execution mode
// (Eager, SingleGraph, MultiGraph).
template <typename Fixture>
class ReduceTest : public Fixture {
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

  ReduceHelper helper_;
};
