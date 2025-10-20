// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class ScatterTest
    : public ::testing::TestWithParam<std::tuple<int, at::ScalarType>> {
 public:
  ScatterTest() : ScatterTest(c10::DeviceType::CUDA) {}
  explicit ScatterTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_type_(device_type) {}

  // Test function declarations with parameters
  void testSyncScatter(int count, at::ScalarType dtype);
  void testSyncScatterNoWork(int count, at::ScalarType dtype);
  void testAsyncScatter(int count, at::ScalarType dtype);
  void testAsyncScatterEarlyReset(int count, at::ScalarType dtype);
  void testScatterInputDeleted(int count, at::ScalarType dtype);
  void testGraphScatter(int count, at::ScalarType dtype);
  void testGraphScatterInputDeleted(int count, at::ScalarType dtype);

 protected:
  virtual std::unique_ptr<TorchCommTestWrapper> createWrapper();

  virtual void SetUp() override;

  virtual void TearDown() override;

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_;
  int num_ranks_;
  c10::DeviceType device_type_;

 private:
  static constexpr int num_replays = 4;

  // Helper function declarations with parameters
  virtual std::vector<at::Tensor> createInputTensors(
      int count,
      at::ScalarType dtype);
  virtual at::Tensor createOutputTensor(int count, at::ScalarType dtype);
  void verifyResults(const at::Tensor& output);
};
