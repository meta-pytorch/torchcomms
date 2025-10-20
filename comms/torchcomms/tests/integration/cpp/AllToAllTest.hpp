// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class AllToAllTest : public ::testing::Test {
 public:
  AllToAllTest() : AllToAllTest(c10::DeviceType::CUDA) {}
  explicit AllToAllTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_type_(device_type) {}

  // Test function declarations
  void testSyncAllToAll(int count, at::ScalarType dtype);
  void testSyncAllToAllNoWork(int count, at::ScalarType dtype);
  void testAsyncAllToAll(int count, at::ScalarType dtype);
  void testAsyncAllToAllEarlyReset(int count, at::ScalarType dtype);
  void testAllToAllInputDeleted(int count, at::ScalarType dtype);
  void testGraphAllToAll(int count, at::ScalarType dtype);
  void testGraphAllToAllInputDeleted(int count, at::ScalarType dtype);

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

  // Helper function declarations
  virtual std::vector<at::Tensor> createInputTensors(
      int count,
      at::ScalarType dtype);
  virtual std::vector<at::Tensor> createOutputTensors(
      int count,
      at::ScalarType dtype);
  std::vector<int> createExpectedOutput();
  void verifyResults(
      const std::vector<at::Tensor>& output_tensors,
      const std::vector<int>& expected_output);
};
