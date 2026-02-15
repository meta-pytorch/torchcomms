// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <tuple>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/tests/integration/cpp/GraphTestFixtures.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

using AllReduceParams = std::tuple<int, at::ScalarType, torch::comms::ReduceOp>;

enum class PreMulSumOpType { kScalar, kTensor };
using PreMulSumParams = std::tuple<int, at::ScalarType, PreMulSumOpType>;

// Helper class for AllReduce test operations.
// Methods are non-static to allow subclass override if needed.
class AllReduceHelper {
 public:
  at::Tensor createInputTensor(
      int count,
      at::ScalarType dtype,
      c10::DeviceType deviceType,
      int rank);

  at::Tensor createPreMulFactorTensor(
      at::ScalarType dtype,
      c10::DeviceType deviceType);

  double calculateExpectedResult(
      const torch::comms::ReduceOp& op,
      int numRanks);

  void verifyResults(
      const at::Tensor& input,
      const torch::comms::ReduceOp& op,
      int numRanks);
};

// AllReduce test class template. Fixture determines execution mode
// (Eager, SingleGraph, MultiGraph).
template <typename Fixture>
class AllReduceTest : public Fixture {
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

  AllReduceHelper helper_;
};
