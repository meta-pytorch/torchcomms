// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class AllToAllvSingleTest : public ::testing::Test {
 public:
  AllToAllvSingleTest() : AllToAllvSingleTest(c10::DeviceType::CUDA) {}
  explicit AllToAllvSingleTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_type_(device_type) {}

  // Test function declarations with parameters
  void testSyncAllToAllvSingle(
      const std::vector<uint64_t>& input_split_sizes,
      const std::vector<uint64_t>& output_split_sizes,
      at::ScalarType dtype);
  void testSyncAllToAllvSingleNoWork(
      const std::vector<uint64_t>& input_split_sizes,
      const std::vector<uint64_t>& output_split_sizes,
      at::ScalarType dtype);
  void testAsyncAllToAllvSingle(
      const std::vector<uint64_t>& input_split_sizes,
      const std::vector<uint64_t>& output_split_sizes,
      at::ScalarType dtype);
  void testAsyncAllToAllvSingleEarlyReset(
      const std::vector<uint64_t>& input_split_sizes,
      const std::vector<uint64_t>& output_split_sizes,
      at::ScalarType dtype);
  void testAllToAllvSingleInputDeleted(
      const std::vector<uint64_t>& input_split_sizes,
      const std::vector<uint64_t>& output_split_sizes,
      at::ScalarType dtype);
  void testGraphAllToAllvSingle(
      const std::vector<uint64_t>& input_split_sizes,
      const std::vector<uint64_t>& output_split_sizes,
      at::ScalarType dtype);
  void testGraphAllToAllvSingleInputDeleted(
      const std::vector<uint64_t>& input_split_sizes,
      const std::vector<uint64_t>& output_split_sizes,
      at::ScalarType dtype);
  void testSyncAllToAllvSingleMultiDimTensor(
      const std::vector<uint64_t>& input_split_sizes,
      const std::vector<uint64_t>& output_split_sizes,
      at::ScalarType dtype);

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
  virtual at::Tensor createInputTensor(
      const std::vector<uint64_t>& input_split_sizes,
      at::ScalarType dtype);
  virtual at::Tensor createOutputTensor(
      const std::vector<uint64_t>& output_split_sizes,
      at::ScalarType dtype);
  void verifyResults(
      const at::Tensor& output,
      const std::vector<uint64_t>& output_split_sizes);
};
