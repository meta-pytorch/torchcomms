// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class AllReduceTest
    : public ::testing::TestWithParam<
          std::tuple<int, at::ScalarType, torch::comms::ReduceOp>> {
 public:
  AllReduceTest() : AllReduceTest(c10::DeviceType::CUDA) {}
  explicit AllReduceTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_type_(device_type) {}

  // Test function declarations with parameters
  void
  testSyncAllReduce(int count, at::ScalarType dtype, torch::comms::ReduceOp op);
  void testSyncAllReduceNoWork(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void testAsyncAllReduce(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void testAsyncAllReduceEarlyReset(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void testAllReduceInputDeleted(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void testGraphAllReduce(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void testGraphAllReduceInputDeleted(
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
  virtual at::Tensor createInputTensor(int count, at::ScalarType dtype);
  virtual at::Tensor createPreMulFactorTensor(at::ScalarType dtype);
  double calculateExpectedResult(torch::comms::ReduceOp op);
  void verifyResults(const at::Tensor& input, torch::comms::ReduceOp op);
};
