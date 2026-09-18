// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <atomic>
#include <barrier>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/colltrace/GraphCollTraceHandle.h"
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
    cancellationGate =
        std::make_shared<EagerCancellationGate>([this](CollTraceEvent& event) {
          return mockCollTrace->cancelEvent(event);
        });
    handle = std::make_unique<CollTraceHandle>(
        mockCollTrace.get(), emptyEvent.get(), cancellationGate);
  }

  std::unique_ptr<MockCollTrace> mockCollTrace;
  std::unique_ptr<CollTraceEvent> emptyEvent;
  std::shared_ptr<EagerCancellationGate> cancellationGate;
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
  ASSERT_TRUE(result.hasValue()) << result.error().message;
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
  ASSERT_TRUE(result1.hasValue()) << result1.error().message;

  auto result2 =
      handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
  ASSERT_TRUE(result2.hasValue()) << result2.error().message;

  auto result3 = handle->trigger(CollTraceHandleTriggerState::KernelStarted);
  ASSERT_TRUE(result3.hasValue()) << result3.error().message;

  auto result4 = handle->trigger(CollTraceHandleTriggerState::KernelFinished);
  ASSERT_TRUE(result4.hasValue()) << result4.error().message;
}

// Test trigger method with invalid state sequence
TEST_F(CollTraceHandleTest, TriggerInvalidStateSequence) {
  // First trigger should be BeforeEnqueueKernel
  auto result1 = handle->trigger(CollTraceHandleTriggerState::KernelStarted);
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
  ASSERT_TRUE(result2.hasValue()) << result2.error().message;
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
  ASSERT_TRUE(result1.hasValue()) << result1.error().message;

  // Try to trigger the same state again
  auto result2 =
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  EXPECT_FALSE(result2.hasValue());
  EXPECT_EQ(result2.error().errorCode, commInvalidArgument);
}

// Test invalidate method
TEST_F(CollTraceHandleTest, Invalidate) {
  // Invalidate the handle
  auto invalidateResult = handle->invalidate();
  ASSERT_TRUE(invalidateResult.hasValue()) << invalidateResult.error().message;

  // Try to trigger a state after invalidation
  auto triggerResult =
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  EXPECT_FALSE(triggerResult.hasValue());
  EXPECT_EQ(triggerResult.error().errorCode, commInvalidArgument);
}

TEST_F(CollTraceHandleTest, CancelRemovesOwnedEventAndInvalidatesHandle) {
  EXPECT_CALL(*mockCollTrace, cancelEvent(testing::Ref(*emptyEvent)))
      .WillOnce(Return(folly::unit));

  const auto cancelResult = handle->cancel();
  ASSERT_TRUE(cancelResult.hasValue()) << cancelResult.error().message;

  auto triggerResult =
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  EXPECT_FALSE(triggerResult.hasValue());
  EXPECT_EQ(triggerResult.error().errorCode, commInvalidArgument);
}

TEST(GraphCancellationGateTest, ShutdownDrainsInFlightCancellation) {
  std::barrier callbackEntered{2};
  std::barrier releaseCallback{2};
  std::barrier shutdownStarted{2};
  std::atomic<int> callbackCount{0};
  std::atomic_bool shutdownFinished{false};
  auto gate = std::make_shared<GraphCancellationGate>([&](uint32_t collId) {
    EXPECT_EQ(collId, 17);
    callbackEntered.arrive_and_wait();
    releaseCallback.arrive_and_wait();
    callbackCount.fetch_add(1, std::memory_order_relaxed);
    return folly::unit;
  });

  std::thread cancelThread([&] { EXPECT_TRUE(gate->cancel(17).hasValue()); });
  callbackEntered.arrive_and_wait();

  std::thread shutdownThread([&] {
    shutdownStarted.arrive_and_wait();
    gate->shutdown();
    shutdownFinished.store(true, std::memory_order_release);
  });
  shutdownStarted.arrive_and_wait();
  EXPECT_FALSE(shutdownFinished.load(std::memory_order_acquire));

  releaseCallback.arrive_and_wait();
  cancelThread.join();
  shutdownThread.join();

  EXPECT_TRUE(shutdownFinished.load(std::memory_order_acquire));
  EXPECT_EQ(callbackCount.load(std::memory_order_relaxed), 1);
  EXPECT_TRUE(gate->cancel(17).hasValue());
  EXPECT_EQ(callbackCount.load(std::memory_order_relaxed), 1);
}

TEST(EagerCancellationGateTest, ShutdownDrainsInFlightCancellation) {
  CollTraceEvent event{
      .collRecord = nullptr,
      .waitEvent = nullptr,
      .replayId = std::nullopt,
      .capturedCollId = std::nullopt};
  std::barrier callbackEntered{2};
  std::barrier releaseCallback{2};
  std::atomic_bool shutdownFinished{false};
  auto gate = std::make_shared<EagerCancellationGate>(
      [&](CollTraceEvent& cancelledEvent) {
        EXPECT_EQ(&cancelledEvent, &event);
        callbackEntered.arrive_and_wait();
        releaseCallback.arrive_and_wait();
        return folly::unit;
      });

  std::thread cancelThread(
      [&] { EXPECT_TRUE(gate->cancel(&event).hasValue()); });
  callbackEntered.arrive_and_wait();

  std::thread shutdownThread([&] {
    gate->shutdown();
    shutdownFinished.store(true, std::memory_order_release);
  });
  EXPECT_FALSE(shutdownFinished.load(std::memory_order_acquire));

  releaseCallback.arrive_and_wait();
  cancelThread.join();
  shutdownThread.join();

  EXPECT_TRUE(shutdownFinished.load(std::memory_order_acquire));
  EXPECT_TRUE(gate->cancel(&event).hasValue());
}

// Teardown that completes before cancellation even reaches the gate must not
// leave the caller holding a reference to the destroyed event: the gate is
// entered with a pointer and never dereferences it once shutdown() has run.
TEST(EagerCancellationGateTest, TeardownBeforeCancelLeavesGateInert) {
  auto ownedEvent = std::make_unique<CollTraceEvent>(nullptr, nullptr);
  std::atomic<int> callbackCount{0};
  auto gate = std::make_shared<EagerCancellationGate>(
      [&](CollTraceEvent& /* cancelledEvent */) {
        callbackCount.fetch_add(1, std::memory_order_relaxed);
        return folly::unit;
      });

  auto* rawEvent = ownedEvent.get();
  gate->shutdown();
  ownedEvent.reset();

  EXPECT_TRUE(gate->cancel(rawEvent).hasValue());
  EXPECT_EQ(callbackCount.load(std::memory_order_relaxed), 0);
}

// Cancellation that is already parked on the gate mutex when teardown starts
// must still run to completion against a live event, and shutdown() must wait
// for it rather than freeing the event underneath it.
TEST(EagerCancellationGateTest, TeardownDuringCancelWaitsForCallback) {
  auto ownedEvent = std::make_unique<CollTraceEvent>(nullptr, nullptr);
  std::barrier callbackEntered{2};
  std::barrier releaseCallback{2};
  std::atomic_bool shutdownFinished{false};
  std::atomic<int> callbackCount{0};
  auto gate = std::make_shared<EagerCancellationGate>(
      [&](CollTraceEvent& cancelledEvent) {
        EXPECT_EQ(&cancelledEvent, ownedEvent.get());
        callbackEntered.arrive_and_wait();
        releaseCallback.arrive_and_wait();
        callbackCount.fetch_add(1, std::memory_order_relaxed);
        return folly::unit;
      });

  auto* rawEvent = ownedEvent.get();
  std::thread cancelThread(
      [&] { EXPECT_TRUE(gate->cancel(rawEvent).hasValue()); });
  callbackEntered.arrive_and_wait();

  std::thread teardownThread([&] {
    gate->shutdown();
    shutdownFinished.store(true, std::memory_order_release);
    ownedEvent.reset();
  });
  EXPECT_FALSE(shutdownFinished.load(std::memory_order_acquire));

  releaseCallback.arrive_and_wait();
  cancelThread.join();
  teardownThread.join();

  EXPECT_TRUE(shutdownFinished.load(std::memory_order_acquire));
  EXPECT_EQ(callbackCount.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(ownedEvent, nullptr);
}

TEST_F(CollTraceHandleTest, EnqueueGuardCancelsUnlessDisarmed) {
  EXPECT_CALL(*mockCollTrace, cancelEvent(testing::Ref(*emptyEvent)))
      .WillOnce(Return(folly::unit));

  {
    CollTraceEnqueueGuard guard(
        std::shared_ptr<ICollTraceHandle>(std::move(handle)));
  }
}

TEST_F(CollTraceHandleTest, EnqueueGuardDisarmSuppressesCancel) {
  EXPECT_CALL(*mockCollTrace, cancelEvent(_)).Times(0);

  {
    CollTraceEnqueueGuard guard(
        std::shared_ptr<ICollTraceHandle>(std::move(handle)));
    guard.disarm();
  }
}

// Test with null CollTrace
TEST_F(CollTraceHandleTest, NullCollTrace) {
  // Create a handle with null CollTrace
  auto nullHandle =
      std::make_unique<CollTraceHandle>(nullptr, emptyEvent.get());

  // Try to trigger a state
  auto result =
      nullHandle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
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
  EXPECT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().errorCode, commInternalError);
}
