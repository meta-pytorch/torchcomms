// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <gtest/gtest.h>
#include <memory>
#include <tuple>
#include "comms/torchcomms/tests/integration/cpp/GraphTestFixtures.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

using BroadcastParams = std::tuple<int, at::ScalarType>;

template <typename Fixture>
class BroadcastTest : public Fixture {
 protected:
  using Fixture::device_type_;
  using Fixture::num_ranks_;
  using Fixture::rank_;
  using Fixture::run;
  using Fixture::torchcomm_;

  void testSync(int count, at::ScalarType dtype);
  void testSyncNoWork(int count, at::ScalarType dtype);
  void testAsync(int count, at::ScalarType dtype);
  void testAsyncEarlyReset(int count, at::ScalarType dtype);
  void testInputDeleted(int count, at::ScalarType dtype);

  at::Tensor
  createTensor(int rootRank, int value, int count, at::ScalarType dtype);
  void verifyResults(const at::Tensor& tensor, int value);
};
