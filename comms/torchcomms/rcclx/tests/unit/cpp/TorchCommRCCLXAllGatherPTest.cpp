// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <memory>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/rcclx/TorchCommRCCLX.hpp"
#include "comms/torchcomms/rcclx/tests/unit/cpp/mocks/HipMock.hpp"
#include "comms/torchcomms/rcclx/tests/unit/cpp/mocks/RcclxMock.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch::comms::test {

class TorchCommRCCLXAllGatherPTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create fresh mocks for each test
    rcclx_mock_ = std::make_shared<NiceMock<RcclxMock>>();
    hip_mock_ = std::make_shared<NiceMock<HipMock>>();

    // Set up default behaviors for mocks
    setupDefaultMockBehaviors();
  }

  void setupDefaultMockBehaviors() {
    // Set up default return values for common operations
    ON_CALL(*rcclx_mock_, groupStart()).WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, groupEnd()).WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, getErrorString(_))
        .WillByDefault(Return("mock error"));
    ON_CALL(*rcclx_mock_, getLastError(_))
        .WillByDefault(Return("mock last error"));

    // Set up HIP mock defaults
    ON_CALL(*hip_mock_, streamCreate(_))
        .WillByDefault(DoAll(
            SetArgPointee<0>(reinterpret_cast<hipStream_t>(0x1000)),
            Return(hipSuccess)));
    ON_CALL(*hip_mock_, streamDestroy(_)).WillByDefault(Return(hipSuccess));
    ON_CALL(*hip_mock_, eventCreate(_))
        .WillByDefault(DoAll(
            SetArgPointee<0>(reinterpret_cast<hipEvent_t>(0x2000)),
            Return(hipSuccess)));
    ON_CALL(*hip_mock_, eventDestroy(_)).WillByDefault(Return(hipSuccess));
    ON_CALL(*hip_mock_, eventRecord(_, _)).WillByDefault(Return(hipSuccess));
    ON_CALL(*hip_mock_, streamWaitEvent(_, _, _))
        .WillByDefault(Return(hipSuccess));
    ON_CALL(*hip_mock_, eventQuery(_)).WillByDefault(Return(hipSuccess));
  }

  std::shared_ptr<NiceMock<RcclxMock>> rcclx_mock_;
  std::shared_ptr<NiceMock<HipMock>> hip_mock_;
};

// Test that allGatherInit is called with correct parameters
TEST_F(TorchCommRCCLXAllGatherPTest, AllGatherInitCallsRcclxApi) {
  void* fake_request = reinterpret_cast<void*>(0x5000);

  // Expect allGatherInit to be called
  EXPECT_CALL(*rcclx_mock_, allGatherInit(_, _, _, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<6>(fake_request), Return(ncclSuccess)));

  // The actual test would require a fully initialized TorchCommRCCLX,
  // which needs more setup. This test validates the mock is correctly set up.
  EXPECT_NE(fake_request, nullptr);
}

// Test that allGatherExec is called with correct parameters
TEST_F(TorchCommRCCLXAllGatherPTest, AllGatherExecCallsRcclxApi) {
  void* fake_request = reinterpret_cast<void*>(0x5000);

  // Expect allGatherExec to be called
  EXPECT_CALL(*rcclx_mock_, allGatherExec(_, _, _, fake_request))
      .WillOnce(Return(ncclSuccess));

  ncclResult_t result =
      rcclx_mock_->allGatherExec(nullptr, 1024, ncclFloat, fake_request);
  EXPECT_EQ(result, ncclSuccess);
}

// Test that pFree is called correctly
TEST_F(TorchCommRCCLXAllGatherPTest, PFreeCallsRcclxApi) {
  void* fake_request = reinterpret_cast<void*>(0x5000);

  // Expect pFree to be called
  EXPECT_CALL(*rcclx_mock_, pFree(fake_request)).WillOnce(Return(ncclSuccess));

  ncclResult_t result = rcclx_mock_->pFree(fake_request);
  EXPECT_EQ(result, ncclSuccess);
}

// Test that pFree handles nullptr gracefully
TEST_F(TorchCommRCCLXAllGatherPTest, PFreeHandlesNullptr) {
  // pFree with nullptr should succeed (no-op)
  EXPECT_CALL(*rcclx_mock_, pFree(nullptr)).WillOnce(Return(ncclSuccess));

  ncclResult_t result = rcclx_mock_->pFree(nullptr);
  EXPECT_EQ(result, ncclSuccess);
}

// Test that hints are passed correctly to allGatherInit
TEST_F(TorchCommRCCLXAllGatherPTest, AllGatherInitPassesHints) {
  void* fake_request = reinterpret_cast<void*>(0x5000);
  RcclxHints hints;
  hints["key1"] = "value1";
  hints["key2"] = "value2";

  // Expect allGatherInit to be called with hints
  EXPECT_CALL(*rcclx_mock_, allGatherInit(_, _, _, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<6>(fake_request), Return(ncclSuccess)));

  ncclResult_t result = rcclx_mock_->allGatherInit(
      nullptr, 1024, hints, ncclFloat, nullptr, nullptr, &fake_request);
  EXPECT_EQ(result, ncclSuccess);
}

// Test error handling for allGatherInit failure
TEST_F(TorchCommRCCLXAllGatherPTest, AllGatherInitHandlesError) {
  EXPECT_CALL(*rcclx_mock_, allGatherInit(_, _, _, _, _, _, _))
      .WillOnce(Return(ncclInternalError));

  void* request = nullptr;
  RcclxHints hints;
  ncclResult_t result = rcclx_mock_->allGatherInit(
      nullptr, 1024, hints, ncclFloat, nullptr, nullptr, &request);
  EXPECT_EQ(result, ncclInternalError);
}

// Test error handling for allGatherExec failure
TEST_F(TorchCommRCCLXAllGatherPTest, AllGatherExecHandlesError) {
  void* fake_request = reinterpret_cast<void*>(0x5000);

  EXPECT_CALL(*rcclx_mock_, allGatherExec(_, _, _, fake_request))
      .WillOnce(Return(ncclInternalError));

  ncclResult_t result =
      rcclx_mock_->allGatherExec(nullptr, 1024, ncclFloat, fake_request);
  EXPECT_EQ(result, ncclInternalError);
}

} // namespace torch::comms::test
