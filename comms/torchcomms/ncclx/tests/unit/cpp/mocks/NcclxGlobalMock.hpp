// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>
#include <nccl.h> // @manual
#include "comms/torchcomms/ncclx/NcclxGlobalApi.hpp"

namespace torch::comms::test {

/**
 * Mock implementation of NcclxGlobalApi using Google Mock.
 */
class NcclxGlobalMock : public NcclxGlobalApi {
 public:
  ~NcclxGlobalMock() override = default;

  MOCK_METHOD(const char*, getErrorString, (ncclResult_t result), (override));

  void setupDefaultBehaviors() {
    using ::testing::_;
    using ::testing::Return;

    ON_CALL(*this, getErrorString(_))
        .WillByDefault(Return("mock nccl error string"));
  }

  void reset() {
    ::testing::Mock::VerifyAndClearExpectations(this);
    setupDefaultBehaviors();
  }
};

} // namespace torch::comms::test
