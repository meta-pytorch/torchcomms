// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <string>
#include <tuple>

#include <ATen/ATen.h>
#include <ATen/cuda/MemPool.h>
#include <gtest/gtest.h>

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchCommWindow.hpp"
#include "comms/torchcomms/tests/integration/cpp/TimeoutTestHelpers.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class WindowRmaTimeoutTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {
 public:
  enum class RmaType {
    kWaitSignal,
    kPutWaitSignal,
  };

  static std::string rmaTypeName(RmaType type);

 protected:
  static constexpr int kNumWarmup = 2;
  static constexpr std::chrono::milliseconds kTimeout{std::chrono::seconds(2)};
  static constexpr std::chrono::seconds kRank0Sleep{5};

  void testTimeout(
      RmaType type,
      torch::comms::test::TimeoutTestHelper::ExecMode mode);

 private:
  void childSetUp();
  void childTearDown();

  void execute(
      RmaType type,
      bool asyncOp = true,
      std::chrono::milliseconds timeout = torch::comms::kNoTimeout);

  torch::comms::test::TimeoutTestHelper helper_;
  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_{0};
  int num_ranks_{2};
  c10::DeviceType device_type_{c10::DeviceType::CUDA};

  std::shared_ptr<torch::comms::TorchCommWindow> window_;
  std::unique_ptr<at::cuda::MemPool> mem_pool_;
  at::Tensor window_tensor_;
};
