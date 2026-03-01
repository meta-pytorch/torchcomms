// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/tests/integration/cpp/TimeoutTestHelpers.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class CollectiveTimeoutTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {
 public:
  enum class CollectiveType {
    kSendRecv,
    kBatchSendRecv,
    kBroadcast,
    kAllReduce,
    kReduce,
    kAllGather,
    kAllGatherSingle,
    kAllGatherV,
    kReduceScatter,
    kReduceScatterSingle,
    kReduceScatterV,
    kAllToAllSingle,
    kAllToAll,
    kBarrier,
    kScatter,
    kGather,
  };

  static std::string collectiveTypeName(CollectiveType type);

 protected:
  static constexpr int kNumWarmup = 2;
  static constexpr std::chrono::milliseconds kTimeout{std::chrono::seconds(2)};
  static constexpr std::chrono::seconds kRank0Sleep{5};

  void testTimeout(
      CollectiveType type,
      torch::comms::test::TimeoutTestHelper::ExecMode mode);

 private:
  void childSetUp();
  void childTearDown();

  void execute(
      CollectiveType type,
      bool asyncOp = true,
      std::chrono::milliseconds timeout = torch::comms::kNoTimeout);

  torch::comms::test::TimeoutTestHelper helper_;
  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_{0};
  int num_ranks_{2};
  c10::DeviceType device_type_{c10::DeviceType::CUDA};
};
