// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/utils/colltrace/CPUWaitEvent.h"
#include "comms/utils/colltrace/CollTrace.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/colltrace/CollTracePlugin.h"
#include "comms/utils/colltrace/tests/MockTypes.h"

using namespace meta::comms;
using namespace meta::comms::colltrace;
using ::testing::_;
using ::testing::AtLeast;
using ::testing::Exactly;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

#define EXPECT_VALUE(cmd)                               \
  {                                                     \
    const auto& res = cmd;                              \
    EXPECT_TRUE(res.hasValue()) << res.error().message; \
  }

// Test fixture for CollTrace tests
class CollTraceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create mock plugin
    auto mockPlugin = std::make_unique<NiceMock<MockCollTracePlugin>>();
    ON_CALL(*mockPlugin, getName()).WillByDefault(Return("MockPlugin"));

    // Set default actions for CommsMaybeVoid methods to return folly::unit
    ON_CALL(*mockPlugin, beforeCollKernelScheduled(_))
        .WillByDefault(Return(folly::unit));
    ON_CALL(*mockPlugin, afterCollKernelScheduled(_))
        .WillByDefault(Return(folly::unit));
    ON_CALL(*mockPlugin, afterCollKernelStart(_))
        .WillByDefault(Return(folly::unit));
    ON_CALL(*mockPlugin, collEventProgressing(_))
        .WillByDefault(Return(folly::unit));
    ON_CALL(*mockPlugin, afterCollKernelEnd(_))
        .WillByDefault(Return(folly::unit));

    // Store a raw pointer to the mock plugin before moving it
    mockPluginPtr = mockPlugin.get();

    // Create a vector of plugins
    std::vector<std::unique_ptr<ICollTracePlugin>> plugins;
    plugins.push_back(std::move(mockPlugin));

    // Create CollTrace with the mock plugin
    collTrace = std::make_unique<CollTrace>(
        CollTraceConfig{
            // Make the check interval very small to reduce time needed for
            // sleep in tests
            .maxCheckCancelInterval = std::chrono::milliseconds(1),
        },
        CommLogData{},
        []() -> CommsMaybeVoid { return folly::unit; },
        std::move(plugins));
  }

  void TearDown() override {
    // Destroy CollTrace first to ensure thread is joined
    collTrace.reset();
  }

  std::unique_ptr<CollTrace> collTrace;
  MockCollTracePlugin* mockPluginPtr;
};

// Test constructor and destructor
TEST_F(CollTraceTest, ConstructorAndDestructor) {
  // Create and destroy a CollTrace object
  auto config = CollTraceConfig{};
  auto logData = CommLogData{};
  auto threadSetupFunc = []() -> CommsMaybeVoid { return folly::unit; };
  std::vector<std::unique_ptr<ICollTracePlugin>> plugins;
  plugins.push_back(std::make_unique<NiceMock<MockCollTracePlugin>>());

  auto trace = std::make_unique<CollTrace>(
      std::move(config),
      std::move(logData),
      threadSetupFunc,
      std::move(plugins));

  // Verify that the object was created successfully
  EXPECT_NE(trace.get(), nullptr);

  // Destroy the object
  trace.reset();
}

// Test getPluginByName method
TEST_F(CollTraceTest, GetPluginByName) {
  // Get the plugin by name
  auto plugin = collTrace->getPluginByName("MockPlugin");
  EXPECT_EQ(plugin, mockPluginPtr);

  // Try to get a non-existent plugin
  auto nonExistentPlugin = collTrace->getPluginByName("NonExistentPlugin");
  EXPECT_EQ(nonExistentPlugin, nullptr);
}

// Test recordCollective method
TEST_F(CollTraceTest, RecordCollective) {
  // Create metadata and wait event
  auto metadata = std::make_unique<::testing::StrictMock<MockCollMetadata>>();
  auto waitEvent = std::make_unique<NiceMock<MockCollWaitEvent>>();

  // Set up expectations for the wait event
  ON_CALL(*waitEvent, beforeCollKernelScheduled())
      .WillByDefault(Return(folly::unit));
  ON_CALL(*waitEvent, afterCollKernelScheduled())
      .WillByDefault(Return(folly::unit));

  // Record a collective
  auto handleMaybe =
      collTrace->recordCollective(std::move(metadata), std::move(waitEvent));

  // Verify that the handle was created successfully
  EXPECT_VALUE(handleMaybe);
  EXPECT_NE(handleMaybe.value().get(), nullptr);
}

// Test triggerEventState method
TEST_F(CollTraceTest, TriggerEventState) {
  // Create metadata and wait event
  auto metadata = std::make_unique<::testing::StrictMock<MockCollMetadata>>();
  auto waitEvent = std::make_unique<NiceMock<MockCollWaitEvent>>();
  auto waitEventPtr = waitEvent.get();

  // Set up expectations for the wait event
  EXPECT_CALL(*waitEventPtr, beforeCollKernelScheduled())
      .WillOnce(Return(folly::unit));
  EXPECT_CALL(*waitEventPtr, afterCollKernelScheduled())
      .WillOnce(Return(folly::unit));

  // Record a collective
  auto handleMaybe =
      collTrace->recordCollective(std::move(metadata), std::move(waitEvent));
  ASSERT_TRUE(handleMaybe.hasValue());
  auto handle = handleMaybe.value();

  // Trigger the BeforeEnqueueKernel state
  EXPECT_VALUE(
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel));

  // Trigger the AfterEnqueueKernel state
  EXPECT_VALUE(
      handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel));
}

// Test complete workflow with multiple collectives
TEST_F(CollTraceTest, CompleteWorkflow) {
  // Create first collective
  auto metadata1 = std::make_unique<::testing::StrictMock<MockCollMetadata>>();
  auto waitEvent1 = std::make_unique<NiceMock<MockCollWaitEvent>>();

  // Set up expectations for the first wait event
  {
    ::testing::InSequence seq; // Ensure the calls happen in sequence
    EXPECT_CALL(*waitEvent1, beforeCollKernelScheduled())
        .WillOnce(Return(folly::unit));
    EXPECT_CALL(*waitEvent1, afterCollKernelScheduled())
        .WillOnce(Return(folly::unit));
    EXPECT_CALL(*waitEvent1, waitCollStart(_)).WillOnce(Return(true));
    EXPECT_CALL(*waitEvent1, waitCollEnd(_)).WillOnce(Return(true));
  }

  // Record first collective
  auto handle1Maybe =
      collTrace->recordCollective(std::move(metadata1), std::move(waitEvent1));
  ASSERT_TRUE(handle1Maybe.hasValue());
  auto handle1 = handle1Maybe.value();

  // Trigger states for first collective
  EXPECT_VALUE(
      handle1->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel));
  EXPECT_VALUE(
      handle1->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel));

  // Create second collective
  auto metadata2 = std::make_unique<::testing::StrictMock<MockCollMetadata>>();
  auto waitEvent2 = std::make_unique<NiceMock<MockCollWaitEvent>>();

  {
    ::testing::InSequence seq; // Ensure the calls happen in sequence

    // Set up expectations for the second wait event using the same config as
    // waitEvent1
    EXPECT_CALL(*waitEvent2, beforeCollKernelScheduled())
        .WillOnce(Return(folly::unit));
    EXPECT_CALL(*waitEvent2, afterCollKernelScheduled())
        .WillOnce(Return(folly::unit));
    EXPECT_CALL(*waitEvent2, waitCollStart(_)).WillOnce(Return(true));
    EXPECT_CALL(*waitEvent2, waitCollEnd(_)).WillOnce(Return(true));
  }

  // Record second collective
  auto handle2Maybe =
      collTrace->recordCollective(std::move(metadata2), std::move(waitEvent2));
  ASSERT_TRUE(handle2Maybe.hasValue());
  auto handle2 = handle2Maybe.value();

  // Trigger states for second collective
  EXPECT_VALUE(
      handle2->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel));
  EXPECT_VALUE(
      handle2->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel));

  // Sleep briefly to allow the CollTrace thread to process the events
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test complete workflow with multiple collectives
TEST_F(CollTraceTest, CompleteWorkflowWithTriggerEvent) {
  // Create first collective
  auto metadata1 = std::make_unique<::testing::StrictMock<MockCollMetadata>>();
  auto waitEvent1 = std::make_unique<NiceMock<MockCollWaitEvent>>();

  std::atomic_flag hasStartCalled;
  std::atomic_flag hasEndCalled;
  std::atomic<int> waitStartCount = 0;
  std::atomic<int> waitEndCount = 0;
  {
    ::testing::InSequence seq; // Ensure the calls happen in sequence

    // Set up expectations for the first wait event
    EXPECT_CALL(*waitEvent1, beforeCollKernelScheduled())
        .WillOnce(Return(folly::unit));
    EXPECT_CALL(*waitEvent1, afterCollKernelScheduled())
        .WillOnce(Return(folly::unit));
    EXPECT_CALL(*waitEvent1, waitCollStart(_))
        .WillRepeatedly(::testing::Invoke([&]() {
          waitStartCount++;
          return hasStartCalled.test();
        }));
    EXPECT_CALL(*waitEvent1, waitCollEnd(_))
        .WillRepeatedly(::testing::Invoke([&]() {
          waitEndCount++;
          return hasEndCalled.test();
        }));
  }

  EXPECT_CALL(*waitEvent1, signalCollStart()).WillOnce(::testing::Invoke([&]() {
    hasStartCalled.test_and_set();
    return folly::unit;
  }));
  EXPECT_CALL(*waitEvent1, signalCollEnd()).WillOnce(::testing::Invoke([&]() {
    hasEndCalled.test_and_set();
    return folly::unit;
  }));

  // Record first collective
  auto handle1Maybe =
      collTrace->recordCollective(std::move(metadata1), std::move(waitEvent1));
  ASSERT_TRUE(handle1Maybe.hasValue());
  auto handle1 = handle1Maybe.value();

  // Trigger states for first collective
  EXPECT_VALUE(
      handle1->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel));
  EXPECT_VALUE(
      handle1->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel));

  // Sleep briefly to allow the CollTrace thread to process the events
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_GT(waitStartCount, 0);

  handle1->trigger(CollTraceHandleTriggerState::KernelStarted);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_GT(waitEndCount, 0);

  handle1->trigger(CollTraceHandleTriggerState::KernelFinished);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test plugin callbacks
TEST_F(CollTraceTest, PluginCallbacks) {
  // Set up expectations for the plugin
  EXPECT_CALL(*mockPluginPtr, beforeCollKernelScheduled(_)).Times(Exactly(1));
  EXPECT_CALL(*mockPluginPtr, afterCollKernelScheduled(_)).Times(Exactly(1));
  EXPECT_CALL(*mockPluginPtr, afterCollKernelStart(_)).Times(Exactly(1));
  EXPECT_CALL(*mockPluginPtr, afterCollKernelEnd(_)).Times(Exactly(1));

  // Create metadata and wait event
  auto metadata = std::make_unique<::testing::StrictMock<MockCollMetadata>>();
  auto waitEvent = std::make_unique<CPUWaitEvent>();

  // Record a collective
  auto handleMaybe =
      collTrace->recordCollective(std::move(metadata), std::move(waitEvent));
  ASSERT_TRUE(handleMaybe.hasValue());
  auto handle = handleMaybe.value();

  // Trigger states
  EXPECT_VALUE(
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel));
  EXPECT_VALUE(
      handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel));

  EXPECT_CALL(*mockPluginPtr, collEventProgressing(_)).Times(AtLeast(1));
  // Sleep briefly to trigger collEventProgressing
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_VALUE(handle->trigger(CollTraceHandleTriggerState::KernelStarted));

  EXPECT_CALL(*mockPluginPtr, collEventProgressing(_)).Times(AtLeast(1));
  // Sleep briefly to trigger collEventProgressing
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_VALUE(handle->trigger(CollTraceHandleTriggerState::KernelFinished));

  // Sleep briefly to allow the CollTrace thread to process the events
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

// Test handle invalidation when CollTrace is destroyed
TEST_F(CollTraceTest, HandleInvalidationOnDestroy) {
  // Create metadata and wait event
  auto metadata = std::make_unique<::testing::StrictMock<MockCollMetadata>>();
  auto waitEvent = std::make_unique<NiceMock<MockCollWaitEvent>>();

  // Set up expectations for the wait event
  ON_CALL(*waitEvent, beforeCollKernelScheduled())
      .WillByDefault(Return(folly::unit));
  ON_CALL(*waitEvent, afterCollKernelScheduled())
      .WillByDefault(Return(folly::unit));

  // Record a collective
  auto handleMaybe =
      collTrace->recordCollective(std::move(metadata), std::move(waitEvent));
  ASSERT_TRUE(handleMaybe.hasValue());
  auto handle = handleMaybe.value();

  // Trigger the first state
  EXPECT_VALUE(
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel));

  // Destroy CollTrace
  collTrace.reset();

  // Try to trigger the next state, should fail because the handle is
  // invalidated
  auto result =
      handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
  EXPECT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().errorCode, commInvalidArgument);
}

// We got sigsegv when we enqueue more collectives than the pending queue could
// handle. While it is expected that some collectives will be dropped in this
// case, we should not see segfault. Add a test to ensure we don't see segfault
// in this case.
TEST_F(CollTraceTest, CheckHandleValidityWhenPendingQueueFull) {
  // Use custom colltrace config to set the max pending queue size to 1
  collTrace = std::make_unique<CollTrace>(
      CollTraceConfig{
          // Make the check interval very small to reduce time needed for
          // sleep in tests
          .maxCheckCancelInterval = std::chrono::milliseconds(1),
          .maxPendingQueueSize = 1,
      },
      CommLogData{},
      []() -> CommsMaybeVoid { return folly::unit; },
      std::vector<std::unique_ptr<ICollTracePlugin>>{});

  std::vector<std::shared_ptr<ICollTraceHandle>> handles;
  for (int i = 0; i < 10; ++i) {
    // Create metadata and wait event
    auto metadata = std::make_unique<NiceMock<MockCollMetadata>>();
    auto waitEvent = std::make_unique<NiceMock<MockCollWaitEvent>>();

    // Set up expectations for the wait event
    ON_CALL(*waitEvent, beforeCollKernelScheduled())
        .WillByDefault(Return(folly::unit));
    ON_CALL(*waitEvent, afterCollKernelScheduled())
        .WillByDefault(Return(folly::unit));
    ON_CALL(*waitEvent, waitCollStart(_))
        .WillByDefault(Return(CommsMaybe<bool>(true)));
    ON_CALL(*waitEvent, waitCollEnd(_))
        .WillByDefault(Return(CommsMaybe<bool>(true)));

    // Record a collective
    auto handleMaybe =
        collTrace->recordCollective(std::move(metadata), std::move(waitEvent));

    // Verify that the handle was created successfully
    EXPECT_VALUE(handleMaybe);
    EXPECT_NE(handleMaybe.value().get(), nullptr);

    // Trigger the enqueue
    handleMaybe.value()->trigger(
        CollTraceHandleTriggerState::BeforeEnqueueKernel);
    handleMaybe.value()->trigger(
        CollTraceHandleTriggerState::AfterEnqueueKernel);

    handles.emplace_back(std::move(handleMaybe.value()));
  }

  for (auto& handle : handles) {
    // Make sure we can get the coll record without encountering segmentation
    // fault. Getting invalid record is expected.
    auto res = handle->getCollRecord();
    if (res.hasValue()) {
      EXPECT_NE(res.value(), nullptr);
    }
  }
}
