// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <gtest/gtest.h>
#include <unordered_map>
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class CommDumpTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  std::shared_ptr<torch::comms::TorchCommNCCLX> ncclx_comm_;
  int rank_{0};
  int num_ranks_{0};
};
