// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class BarrierTest : public ::testing::Test {
 public:
  BarrierTest() {}

  // Test function declarations
  void testSyncBarrier();
  void testSyncBarrierNoWork();
  void testAsyncBarrier();
  void testAsyncBarrierEarlyReset();
  void testGraphBarrier();

 protected:
  virtual std::unique_ptr<TorchCommTestWrapper> createWrapper();

  virtual void SetUp() override;

  virtual void TearDown() override;

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_{};
  int num_ranks_{};

  static constexpr int num_replays = 4;
};
