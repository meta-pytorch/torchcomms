// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <unordered_map>

#include <gmock/gmock.h>
#include <nccl.h> // @manual
#include "comms/torchcomms/ncclx/NcclxGlobalApi.hpp"

namespace torch::comms::test {

// Type alias to avoid preprocessor comma issues inside MOCK_METHOD macros.
using NcclxCommDumpAllMap = std::
    unordered_map<std::string, std::unordered_map<std::string, std::string>>;

/**
 * Mock implementation of NcclxGlobalApi using Google Mock.
 */
class NcclxGlobalMock : public NcclxGlobalApi {
 public:
  ~NcclxGlobalMock() override = default;

  MOCK_METHOD(const char*, getErrorString, (ncclResult_t result), (override));
  MOCK_METHOD(
      ncclResult_t,
      commDumpAll,
      (NcclxCommDumpAllMap & map),
      (override));

  void setupDefaultBehaviors() {
    using ::testing::_;
    using ::testing::Return;

    ON_CALL(*this, getErrorString(_))
        .WillByDefault(Return("mock nccl error string"));
    ON_CALL(*this, commDumpAll(_)).WillByDefault(Return(ncclSuccess));
  }

  void reset() {
    ::testing::Mock::VerifyAndClearExpectations(this);
    setupDefaultBehaviors();
  }
};

} // namespace torch::comms::test
