// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class BroadcastTest
    : public ::testing::TestWithParam<std::tuple<int, at::ScalarType>> {
 public:
  BroadcastTest() : BroadcastTest(c10::DeviceType::CUDA) {}
  explicit BroadcastTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_type_(device_type) {}

  // Test function declarations with parameters
  void testSyncBroadcast(int count, at::ScalarType dtype);
  void testSyncBroadcastNoWork(int count, at::ScalarType dtype);
  void testAsyncBroadcast(int count, at::ScalarType dtype);
  void testAsyncBroadcastEarlyReset(int count, at::ScalarType dtype);
  void testBroadcastInputDeleted(int count, at::ScalarType dtype);
  void testGraphBroadcast(int count, at::ScalarType dtype);
  void testGraphBroadcastInputDeleted(int count, at::ScalarType dtype);

 protected:
  virtual std::unique_ptr<TorchCommTestWrapper> createWrapper();

  virtual void SetUp() override;

  virtual void TearDown() override;

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_;
  int num_ranks_;
  c10::DeviceType device_type_;

  static constexpr int num_replays = 4;

  // Helper function declarations with parameters
  at::Tensor createBroadcastTensor(
      int root_rank,
      int value,
      int count,
      at::ScalarType dtype);
  void verifyBroadcastResults(const at::Tensor& tensor, int value);
};
