// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class ReduceScatterSingleTest
    : public ::testing::TestWithParam<
          std::tuple<int, at::ScalarType, torch::comms::ReduceOp>> {
 public:
  ReduceScatterSingleTest() : ReduceScatterSingleTest(c10::DeviceType::CUDA) {}
  explicit ReduceScatterSingleTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_type_(device_type) {}

  // Test function declarations with parameters
  void testSyncReduceScatterSingle(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void testSyncReduceScatterSingleNoWork(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void testAsyncReduceScatterSingle(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void testAsyncReduceScatterSingleEarlyReset(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void testReduceScatterSingleInputDeleted(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void testGraphReduceScatterSingle(
      int count,
      at::ScalarType dtype,
      const torch::comms::ReduceOp& op);
  void testGraphReduceScatterSingleInputDeleted(
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
  virtual at::Tensor createInputTensor(int count, at::ScalarType dtype);
  virtual at::Tensor createOutputTensor(int count, at::ScalarType dtype);
  void verifyResults(
      const at::Tensor& output,
      const torch::comms::ReduceOp& op);
};
