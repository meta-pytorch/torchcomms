// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class ReduceTest
    : public ::testing::TestWithParam<
          std::tuple<int, at::ScalarType, torch::comms::ReduceOp>> {
 public:
  ReduceTest() : ReduceTest(c10::DeviceType::CUDA) {}
  explicit ReduceTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_index_(0), device_type_(device_type) {}

  // Test function declarations with parameters
  void
  testSyncReduce(int count, at::ScalarType dtype, torch::comms::ReduceOp op);
  void testSyncReduceNoWork(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void
  testAsyncReduce(int count, at::ScalarType dtype, torch::comms::ReduceOp op);
  void testAsyncReduceEarlyReset(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void testReduceInputDeleted(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);
  void
  testGraphReduce(int count, at::ScalarType dtype, torch::comms::ReduceOp op);
  void testGraphReduceInputDeleted(
      int count,
      at::ScalarType dtype,
      torch::comms::ReduceOp op);

 protected:
  virtual void synchronizeStream();

  virtual std::unique_ptr<TorchCommTestWrapper> createWrapper();

  virtual void SetUp() override;

  virtual void TearDown() override;

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_;
  int num_ranks_;
  int device_index_;
  c10::DeviceType device_type_;

  static constexpr int num_replays = 4;

  // Helper function declarations with parameters
  virtual at::Tensor createInputTensor(int count, at::ScalarType dtype);
  int calculateExpectedResult(torch::comms::ReduceOp op);
  void verifyResults(
      const at::Tensor& output,
      torch::comms::ReduceOp op,
      int root_rank);
};
