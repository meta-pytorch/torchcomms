// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class AllGatherTest
    : public ::testing::TestWithParam<std::tuple<int, at::ScalarType>> {
 public:
  AllGatherTest() : AllGatherTest(c10::DeviceType::CUDA) {}
  explicit AllGatherTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_type_(device_type) {}

  // Test function declarations with parameters
  void testSyncAllGather(int count, at::ScalarType dtype);
  void testSyncAllGatherNoWork(int count, at::ScalarType dtype);
  void testAsyncAllGather(int count, at::ScalarType dtype);
  void testAsyncAllGatherEarlyReset(int count, at::ScalarType dtype);
  void testAllGatherInputDeleted(int count, at::ScalarType dtype);
  void testGraphAllGather(int count, at::ScalarType dtype);
  void testGraphAllGatherInputDeleted(int count, at::ScalarType dtype);

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
  virtual std::vector<at::Tensor> createOutputTensors(
      int count,
      at::ScalarType dtype);
  void verifyResults(const std::vector<at::Tensor>& outputs);
};
