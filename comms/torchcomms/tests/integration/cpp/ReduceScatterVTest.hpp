// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class ReduceScatterVTest
    : public ::testing::TestWithParam<
          std::tuple<int, at::ScalarType, torch::comms::ReduceOp>> {
 public:
  ReduceScatterVTest() : ReduceScatterVTest(c10::DeviceType::CUDA) {}
  explicit ReduceScatterVTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_type_(device_type) {}

  // Test function declarations with parameters
  void testSyncReduceScatterV(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void testSyncReduceScatterVNoWork(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void testAsyncReduceScatterV(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void testAsyncReduceScatterVEarlyReset(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void testReduceScatterVInputDeleted(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void testGraphReduceScatterV(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void testGraphReduceScatterVInputDeleted(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);

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
