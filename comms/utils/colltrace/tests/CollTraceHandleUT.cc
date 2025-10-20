// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/logging/xlog.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/colltrace/tests/MockTypes.h"

using namespace meta::comms;
using namespace meta::comms::colltrace;
using ::testing::_;
using ::testing::Return;

// Test fixture for CollTraceHandle tests
class CollTraceHandleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mockCollTrace = std::make_unique<MockCollTrace>();
    emptyEvent = std::make_unique<CollTraceEvent>(nullptr, nullptr);
    handle = std::make_unique<CollTraceHandle>(
        mockCollTrace.get(), emptyEvent.get());
  }

  std::unique_ptr<MockCollTrace> mockCollTrace;
  std::unique_ptr<CollTraceEvent> emptyEvent;
  std::unique_ptr<CollTraceHandle> handle;
};

// Test constructor
TEST_F(CollTraceHandleTest, Constructor) {
  // A newly constructed handle should not be invalidated
  EXPECT_CALL(
      *mockCollTrace,
      triggerEventState(
          testing::Ref(*emptyEvent),
          CollTraceHandleTriggerState::BeforeEnqueueKernel))
      .WillOnce(Return(folly::unit));
  auto result =
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  if (!result.hasValue()) {
    XLOG(ERR) << "Received Error: " << result.error().message << std::endl;
  }
  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result.value(), folly::unit);
}

// Test trigger method with valid state sequence
TEST_F(CollTraceHandleTest, TriggerValidStateSequence) {
  // Set up expectations for the mock
  EXPECT_CALL(
      *mockCollTrace,
      triggerEventState(
          testing::Ref(*emptyEvent),
          CollTraceHandleTriggerState::BeforeEnqueueKernel))
      .WillOnce(Return(folly::unit));
  EXPECT_CALL(
      *mockCollTrace,
      triggerEventState(
          testing::Ref(*emptyEvent),
          CollTraceHandleTriggerState::AfterEnqueueKernel))
      .WillOnce(Return(folly::unit));
  EXPECT_CALL(
      *mockCollTrace,
      triggerEventState(
          testing::Ref(*emptyEvent),
          CollTraceHandleTriggerState::KernelStarted))
      .WillOnce(Return(folly::unit));
  EXPECT_CALL(
      *mockCollTrace,
      triggerEventState(
          testing::Ref(*emptyEvent),
          CollTraceHandleTriggerState::KernelFinished))
      .WillOnce(Return(folly::unit));

  // Trigger states in the correct sequence
  auto result1 =
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  if (!result1.hasValue()) {
    XLOG(ERR) << "Received Error: " << result1.error().message << std::endl;
  }
  ASSERT_TRUE(result1.hasValue());

  auto result2 =
      handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
  if (!result2.hasValue()) {
    XLOG(ERR) << "Received Error: " << result2.error().message << std::endl;
  }
  ASSERT_TRUE(result2.hasValue());

  auto result3 = handle->trigger(CollTraceHandleTriggerState::KernelStarted);
  if (!result3.hasValue()) {
    XLOG(ERR) << "Received Error: " << result3.error().message << std::endl;
  }
  ASSERT_TRUE(result3.hasValue());

  auto result4 = handle->trigger(CollTraceHandleTriggerState::KernelFinished);
  if (!result4.hasValue()) {
    XLOG(ERR) << "Received Error: " << result4.error().message << std::endl;
  }
  ASSERT_TRUE(result4.hasValue());
}

// Test trigger method with invalid state sequence
TEST_F(CollTraceHandleTest, TriggerInvalidStateSequence) {
  // First trigger should be BeforeEnqueueKernel
  auto result1 = handle->trigger(CollTraceHandleTriggerState::KernelStarted);
  if (!result1.hasValue()) {
    XLOG(ERR) << "Received Error: " << result1.error().message << std::endl;
  }
  EXPECT_FALSE(result1.hasValue());
  EXPECT_EQ(result1.error().errorCode, commInvalidArgument);

  // Set up expectations for the mock for a valid first trigger
  EXPECT_CALL(
      *mockCollTrace,
      triggerEventState(
          testing::Ref(*emptyEvent),
          CollTraceHandleTriggerState::BeforeEnqueueKernel))
      .WillOnce(Return(folly::unit));

  // Trigger the first valid state
  auto result2 =
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  if (!result2.hasValue()) {
    XLOG(ERR) << "Received Error: " << result2.error().message << std::endl;
  }
  EXPECT_TRUE(result2.hasValue());
}

// Test triggering the same state multiple times
TEST_F(CollTraceHandleTest, TriggerSameStateMultipleTimes) {
  // Set up expectations for the mock
  EXPECT_CALL(
      *mockCollTrace,
      triggerEventState(
          testing::Ref(*emptyEvent),
          CollTraceHandleTriggerState::BeforeEnqueueKernel))
      .WillOnce(Return(folly::unit));

  // Trigger the first state
  auto result1 =
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  if (!result1.hasValue()) {
    XLOG(ERR) << "Received Error: " << result1.error().message << std::endl;
  }
  EXPECT_TRUE(result1.hasValue());

  // Try to trigger the same state again
  auto result2 =
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  if (!result2.hasValue()) {
    XLOG(ERR) << "Received Error: " << result2.error().message << std::endl;
  }
  EXPECT_FALSE(result2.hasValue());
  EXPECT_EQ(result2.error().errorCode, commInvalidArgument);
}

// Test invalidate method
TEST_F(CollTraceHandleTest, Invalidate) {
  // Invalidate the handle
  auto invalidateResult = handle->invalidate();
  if (!invalidateResult.hasValue()) {
    XLOG(ERR) << "Received Error: " << invalidateResult.error().message
              << std::endl;
  }
  ASSERT_TRUE(invalidateResult.hasValue());

  // Try to trigger a state after invalidation
  auto triggerResult =
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  if (!triggerResult.hasValue()) {
    XLOG(ERR) << "Received Error: " << triggerResult.error().message
              << std::endl;
  }
  EXPECT_FALSE(triggerResult.hasValue());
  EXPECT_EQ(triggerResult.error().errorCode, commInvalidArgument);
}

// Test with null CollTrace
TEST_F(CollTraceHandleTest, NullCollTrace) {
  // Create a handle with null CollTrace
  auto nullHandle =
      std::make_unique<CollTraceHandle>(nullptr, emptyEvent.get());

  // Try to trigger a state
  auto result =
      nullHandle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  if (!result.hasValue()) {
    XLOG(ERR) << "Received Error: " << result.error().message << std::endl;
  }
  EXPECT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().errorCode, commInternalError);
}

// Test with null CollTraceEvent
TEST_F(CollTraceHandleTest, NullCollTraceEvent) {
  // Create a handle with null CollTraceEvent
  auto nullHandle =
      std::make_unique<CollTraceHandle>(mockCollTrace.get(), nullptr);

  // Try to trigger a state
  auto result =
      nullHandle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  if (!result.hasValue()) {
    XLOG(ERR) << "Received Error: " << result.error().message << std::endl;
  }
  EXPECT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().errorCode, commInternalError);
}
