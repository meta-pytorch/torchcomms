// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class ReduceScatterTest
    : public ::testing::TestWithParam<
          std::tuple<int, at::ScalarType, torch::comms::ReduceOp>> {
 public:
  ReduceScatterTest() : ReduceScatterTest(c10::DeviceType::CUDA) {}
  explicit ReduceScatterTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_type_(device_type) {}

  // Test function declarations with parameters
  void testSyncReduceScatter(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void testSyncReduceScatterNoWork(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void testAsyncReduceScatter(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void testAsyncReduceScatterEarlyReset(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void testReduceScatterInputDeleted(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void testGraphReduceScatter(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void testGraphReduceScatterInputDeleted(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);

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
  virtual std::vector<at::Tensor> createInputTensors(
      int count,
      at::ScalarType dtype);
  virtual at::Tensor createOutputTensor(int count, at::ScalarType dtype);
  int calculateExpectedResult(torch::comms::ReduceOp op);
  void verifyResults(const at::Tensor& output, torch::comms::ReduceOp op);
};
