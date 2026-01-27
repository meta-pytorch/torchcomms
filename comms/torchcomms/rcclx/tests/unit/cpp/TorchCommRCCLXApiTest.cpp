// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "comms/torchcomms/rcclx/RcclxApi.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;

namespace torch::comms::test {

class TorchcommRCCLXApiTest : public ::testing::Test {};

TEST_F(TorchcommRCCLXApiTest, UnsupportedWindowApiTest) {
  auto rcclx_api = std::make_unique<DefaultRcclxApi>();
  ncclComm_t nccl_comm = nullptr;
  EXPECT_THROW(
      rcclx_api->winAllocate(0, nccl_comm, nullptr, nullptr, true, 0),
      std::runtime_error);
  EXPECT_THROW(rcclx_api->winFree(nccl_comm, nullptr), std::runtime_error);
  EXPECT_THROW(
      rcclx_api->winPut(nullptr, 0, ncclFloat, 0, 0, nullptr, nullptr),
      std::runtime_error);
  EXPECT_THROW(
      rcclx_api->winSharedQuery(0, nccl_comm, nullptr, nullptr),
      std::runtime_error);
  EXPECT_THROW(
      rcclx_api->winWaitSignal(0, 0, 0, nullptr, nullptr), std::runtime_error);
  EXPECT_THROW(
      rcclx_api->winSignal(0, 0, 0, nullptr, nullptr), std::runtime_error);
}

} // namespace torch::comms::test
