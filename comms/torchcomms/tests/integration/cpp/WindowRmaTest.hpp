// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class WindowRmaTest : public ::testing::TestWithParam<
                          std::tuple<int, at::ScalarType, bool, bool, bool>> {
 public:
  WindowRmaTest() : WindowRmaTest(c10::DeviceType::CUDA) {}
  explicit WindowRmaTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_index_(0), device_type_(device_type) {}

  // Test function declarations with parameters
  void testWindowPutBasic(
      int count,
      at::ScalarType dtype,
      bool async_op,
      bool signal,
      bool async_signal);
  void testWindowCpuPut(
      int count,
      at::ScalarType dtype,
      bool async_op,
      bool signal,
      bool async_signal);

  bool checkIfSkip();

 protected:
  std::unique_ptr<TorchCommTestWrapper> createWrapper();
  void SetUp() override;

  void TearDown() override;

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_;
  int num_ranks_;
  int device_index_;
  c10::DeviceType device_type_;

  static constexpr int num_replays = 4;

  // Helper function declarations with parameters
  at::Tensor createWindowRmaTensor(int value, int count, at::ScalarType dtype);
  void verifyWindowRmaResults(const at::Tensor& tensor, int value);
};
