// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <atomic>
#include <future>

#include <folly/Unit.h>
#include <limits>
#include <thread>

#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/colltrace/plugins/WatchdogPlugin.h"
#include "comms/utils/colltrace/tests/MockTypes.h"

using namespace meta::comms;
using namespace meta::comms::colltrace;
using ::testing::_;
using ::testing::Return;

// Helper macro for checking if a CommsMaybe has a value
#define EXPECT_VALUE(cmd)                               \
  {                                                     \
    const auto& res = cmd;                              \
    EXPECT_TRUE(res.hasValue()) << res.error().message; \
  }

// Test fixture for WatchdogPlugin tests
class WatchdogPluginTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Reset call counters
    errorCheckCallCount.store(0);
    triggerCallCount.store(0);
    lastTriggeredCollId.store(kNoCollId);
  }

  void TearDown() override {
    plugin.reset();
  }

  // Helper method to create a CollTraceEvent with a CollRecord
  CollTraceEvent createCollTraceEvent(uint64_t collId) {
    auto metadata = std::make_unique<MockCollMetadata>();
    auto collRecord = std::make_shared<CollRecord>(collId, std::move(metadata));

    CollTraceEvent event;
    event.collRecord = collRecord;
    return event;
  }

  // Mock functions for testing
  static std::atomic<int> errorCheckCallCount;
  static std::atomic<int> triggerCallCount;
  static std::atomic<uint64_t> lastTriggeredCollId;
  static constexpr uint64_t kNoCollId = std::numeric_limits<uint64_t>::max();

  static bool mockErrorCheck() {
    errorCheckCallCount.fetch_add(1);
    return false; // Default: no error
  }

  static bool mockErrorCheckTrue() {
    errorCheckCallCount.fetch_add(1);
    return true; // Simulate error condition
  }

  static void mockTriggerOnError(CollTraceEvent& event) {
    triggerCallCount.fetch_add(1);
    lastTriggeredCollId.store(event.collRecord->getCollId());
  }

  std::unique_ptr<WatchdogPlugin> plugin;
};

// Static member definitions
std::atomic<int> WatchdogPluginTest::errorCheckCallCount{0};
std::atomic<int> WatchdogPluginTest::triggerCallCount{0};
std::atomic<uint64_t> WatchdogPluginTest::lastTriggeredCollId{kNoCollId};

// Test constructor with default config
TEST_F(WatchdogPluginTest, ConstructorWithDefaultConfig) {
  WatchdogPluginConfig config;
  plugin = std::make_unique<WatchdogPlugin>(config);

  EXPECT_NE(plugin.get(), nullptr);
  EXPECT_EQ(plugin->getName(), WatchdogPlugin::kWatchdogPluginName);
  EXPECT_EQ(plugin->getName(), "WatchdogPlugin");
}

// Test constructor with custom config
TEST_F(WatchdogPluginTest, ConstructorWithCustomConfig) {
  WatchdogPluginConfig config;
  config.funcIfError = mockErrorCheck;
  config.funcTriggerOnError = mockTriggerOnError;

  plugin = std::make_unique<WatchdogPlugin>(config);

  EXPECT_NE(plugin.get(), nullptr);
  EXPECT_EQ(plugin->getName(), "WatchdogPlugin");
}

// Test getName method
TEST_F(WatchdogPluginTest, GetName) {
  WatchdogPluginConfig config;
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto name = plugin->getName();
  EXPECT_EQ(name, "WatchdogPlugin");
  EXPECT_EQ(name, WatchdogPlugin::kWatchdogPluginName);
}

// Test collEventProgressing with no error condition
TEST_F(WatchdogPluginTest, CollEventProgressingNoError) {
  WatchdogPluginConfig config;
  config.funcIfError = mockErrorCheck;
  config.funcTriggerOnError = mockTriggerOnError;
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event = createCollTraceEvent(1);
  auto result = plugin->collEventProgressing(event);

  EXPECT_VALUE(result);
  EXPECT_EQ(errorCheckCallCount.load(), 1);
  EXPECT_EQ(triggerCallCount.load(), 0);
  EXPECT_EQ(lastTriggeredCollId.load(), kNoCollId);
}

// Test collEventProgressing with error condition
TEST_F(WatchdogPluginTest, CollEventProgressingWithError) {
  WatchdogPluginConfig config;
  config.funcIfError = mockErrorCheckTrue;
  config.funcTriggerOnError = mockTriggerOnError;
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event = createCollTraceEvent(1);
  auto result = plugin->collEventProgressing(event);

  EXPECT_VALUE(result);
  EXPECT_EQ(errorCheckCallCount.load(), 1);
  EXPECT_EQ(triggerCallCount.load(), 1);
  EXPECT_EQ(lastTriggeredCollId.load(), event.collRecord->getCollId());
}

// Test collEventProgressing multiple calls with mixed error conditions
TEST_F(WatchdogPluginTest, CollEventProgressingMultipleCalls) {
  WatchdogPluginConfig config{
      .funcIfError =
          []() {
            static int callCount = 0;
            callCount++;
            // Return error on second call
            return callCount == 2;
          },
      .funcTriggerOnError = mockTriggerOnError,
  };
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event1 = createCollTraceEvent(1);
  auto event2 = createCollTraceEvent(2);

  // First call - no error
  auto result1 = plugin->collEventProgressing(event1);
  EXPECT_VALUE(result1);
  EXPECT_EQ(triggerCallCount.load(), 0);

  // Second call - error condition
  auto result2 = plugin->collEventProgressing(event2);
  EXPECT_VALUE(result2);
  EXPECT_EQ(triggerCallCount.load(), 1);
  EXPECT_EQ(lastTriggeredCollId.load(), event2.collRecord->getCollId());
}

// Test with default config functions
TEST_F(WatchdogPluginTest, DefaultConfigFunctions) {
  WatchdogPluginConfig config; // Uses default functions
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event = createCollTraceEvent(1);

  // Default funcIfError should return false, so no trigger should occur
  auto result = plugin->collEventProgressing(event);
  EXPECT_VALUE(result);

  // We can't easily test the default logFatalError function without
  // causing the test to terminate, but we can verify the plugin works
  // with default configuration
}

// Test plugin interface compliance
TEST_F(WatchdogPluginTest, PluginInterfaceCompliance) {
  WatchdogPluginConfig config;
  plugin = std::make_unique<WatchdogPlugin>(config);

  // Verify the plugin implements ICollTracePlugin interface
  ICollTracePlugin* pluginInterface = plugin.get();
  EXPECT_NE(pluginInterface, nullptr);

  auto event = createCollTraceEvent(1);

  // Test all interface methods return valid results
  EXPECT_VALUE(pluginInterface->beforeCollKernelScheduled(event));
  EXPECT_VALUE(pluginInterface->afterCollKernelScheduled(event));
  EXPECT_VALUE(pluginInterface->afterCollKernelStart(event));
  EXPECT_VALUE(pluginInterface->collEventProgressing(event));
  EXPECT_VALUE(pluginInterface->afterCollKernelEnd(event));
  EXPECT_VALUE(pluginInterface->afterCollTerminated(
      event, CollTraceTerminalReason::TraceDestroyed));

  // Test getName
  EXPECT_EQ(pluginInterface->getName(), "WatchdogPlugin");
}

// Test with lambda functions in config
TEST_F(WatchdogPluginTest, LambdaFunctionsInConfig) {
  bool errorCondition = false;
  bool triggerCalled = false;
  uint64_t triggeredCollId = kNoCollId;

  WatchdogPluginConfig config{
      .funcIfError = [&errorCondition]() { return errorCondition; },
      .funcTriggerOnError =
          [&triggerCalled, &triggeredCollId](CollTraceEvent& event) {
            triggerCalled = true;
            triggeredCollId = event.collRecord->getCollId();
          },
  };

  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event = createCollTraceEvent(1);

  // First call with no error
  errorCondition = false;
  auto result1 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result1);
  EXPECT_FALSE(triggerCalled);

  // Second call with error
  errorCondition = true;
  auto result2 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result2);
  EXPECT_TRUE(triggerCalled);
  EXPECT_EQ(triggeredCollId, event.collRecord->getCollId());
}

TEST_F(WatchdogPluginTest, AsyncErrorDelayDoesNotBlockProgress) {
  std::promise<uint64_t> triggeredCollId;
  auto triggeredCollIdFuture = triggeredCollId.get_future();

  WatchdogPluginConfig config{
      .funcIfError = mockErrorCheckTrue,
      .funcTriggerOnError =
          [&triggeredCollId](CollTraceEvent& event) {
            triggeredCollId.set_value(event.collRecord->getCollId());
          },
      .asyncErrorDelay = std::chrono::milliseconds{500},
      .deferErrorTrigger = true,
  };
  plugin = std::make_unique<WatchdogPlugin>(config);

  {
    auto event = createCollTraceEvent(1);
    const auto start = std::chrono::steady_clock::now();
    EXPECT_VALUE(plugin->collEventProgressing(event));
    const auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_LT(elapsed, std::chrono::milliseconds{250});
  }
  plugin.reset();

  ASSERT_EQ(
      triggeredCollIdFuture.wait_for(std::chrono::seconds{2}),
      std::future_status::ready);
  EXPECT_EQ(triggeredCollIdFuture.get(), 1);
}

TEST_F(WatchdogPluginTest, AsyncErrorLatchesAcrossActiveEvents) {
  WatchdogPluginConfig config{
      .funcIfError = mockErrorCheckTrue,
      .funcTriggerOnError = mockTriggerOnError,
  };
  plugin = std::make_unique<WatchdogPlugin>(config);
  auto event1 = createCollTraceEvent(1);
  auto event2 = createCollTraceEvent(2);

  EXPECT_VALUE(plugin->collEventProgressing(event1));
  EXPECT_VALUE(plugin->collEventProgressing(event2));

  EXPECT_EQ(errorCheckCallCount.load(), 1);
  EXPECT_EQ(triggerCallCount.load(), 1);
  EXPECT_EQ(lastTriggeredCollId.load(), event1.collRecord->getCollId());
}

// Test timeout functionality - disabled by default
TEST_F(WatchdogPluginTest, TimeoutDisabledByDefault) {
  WatchdogPluginConfig config{
      .checkTimeout = false, // Explicitly disabled
      .timeout = std::chrono::milliseconds{1}, // 1ms timeout
      .funcTriggerOnTimeout = mockTriggerOnError,
  };
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event = createCollTraceEvent(1);

  // Multiple calls to the same event should not trigger timeout
  auto result1 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result1);
  // Sleep longer than timeout, it shouldn't trigger as timeout is disabled
  std::this_thread::sleep_for(std::chrono::milliseconds{10});
  auto result2 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result2);

  EXPECT_EQ(triggerCallCount.load(), 0);
  EXPECT_EQ(lastTriggeredCollId.load(), kNoCollId);
}

// Test per-event timers are independent — timing out event1 doesn't
// affect event2's timer.
TEST_F(WatchdogPluginTest, PerEventTimersAreIndependent) {
  WatchdogPluginConfig config{
      .checkTimeout = true,
      .timeout = std::chrono::milliseconds{50},
      .funcTriggerOnTimeout = mockTriggerOnError,
  };
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event1 = createCollTraceEvent(1);
  auto event2 = createCollTraceEvent(2);

  // Start both timers.
  plugin->collEventProgressing(event1);
  plugin->collEventProgressing(event2);

  // Sleep past timeout.
  std::this_thread::sleep_for(std::chrono::milliseconds{60});

  // event1 times out.
  plugin->collEventProgressing(event1);
  EXPECT_EQ(triggerCallCount.load(), 1);
  EXPECT_EQ(lastTriggeredCollId.load(), event1.collRecord->getCollId());

  // event2 also times out independently.
  plugin->collEventProgressing(event2);
  EXPECT_EQ(triggerCallCount.load(), 2);
  EXPECT_EQ(lastTriggeredCollId.load(), event2.collRecord->getCollId());
}

// Test timeout functionality - timeout occurs
TEST_F(WatchdogPluginTest, TimeoutOccurs) {
  WatchdogPluginConfig config{
      .checkTimeout = true,
      .timeout = std::chrono::milliseconds{50}, // 50ms timeout
      .funcTriggerOnTimeout = mockTriggerOnError,
  };
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event = createCollTraceEvent(1);

  // First call - starts the timer
  auto result1 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result1);
  EXPECT_EQ(triggerCallCount.load(), 0);

  // Sleep longer than timeout
  std::this_thread::sleep_for(std::chrono::milliseconds{60});

  // Second call with same event - should trigger timeout
  auto result2 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result2);
  EXPECT_EQ(triggerCallCount.load(), 1);
  EXPECT_EQ(lastTriggeredCollId.load(), event.collRecord->getCollId());
}

// Test afterCollKernelEnd clears the timer for that event.
TEST_F(WatchdogPluginTest, AfterCollKernelEndClearsTimer) {
  WatchdogPluginConfig config{
      .checkTimeout = true,
      .timeout = std::chrono::milliseconds{50},
      .funcTriggerOnTimeout = mockTriggerOnError,
  };
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event = createCollTraceEvent(1);

  // Start timer.
  plugin->collEventProgressing(event);

  // Complete the collective — clears the timer.
  plugin->afterCollKernelEnd(event);

  // Sleep past timeout.
  std::this_thread::sleep_for(std::chrono::milliseconds{60});

  // Progressing again — starts a fresh timer, no timeout.
  plugin->collEventProgressing(event);
  EXPECT_EQ(triggerCallCount.load(), 0);
}

TEST_F(WatchdogPluginTest, TerminalDispositionClearsTimer) {
  WatchdogPluginConfig config{
      .checkTimeout = true,
      .timeout = std::chrono::milliseconds{50},
      .funcTriggerOnTimeout = mockTriggerOnError,
  };
  plugin = std::make_unique<WatchdogPlugin>(config);
  auto event = createCollTraceEvent(1);

  EXPECT_VALUE(plugin->collEventProgressing(event));
  EXPECT_VALUE(plugin->afterCollTerminated(
      event, CollTraceTerminalReason::TraceDestroyed));
  EXPECT_VALUE(plugin->collEventProgressing(event));

  EXPECT_EQ(triggerCallCount.load(), 0);
}

// Test startTs change resets the timer (new replay detection).
TEST_F(WatchdogPluginTest, StartTsChangeResetsTimer) {
  WatchdogPluginConfig config{
      .checkTimeout = true,
      .timeout = std::chrono::milliseconds{50},
      .funcTriggerOnTimeout = mockTriggerOnError,
  };
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event = createCollTraceEvent(1);

  // Start timer with initial startTs.
  auto t1 = std::chrono::system_clock::now();
  event.collRecord->getTimingInfo().setCollStartTs(t1);
  plugin->collEventProgressing(event);

  // Sleep past timeout.
  std::this_thread::sleep_for(std::chrono::milliseconds{60});

  // Change startTs (simulates new replay) — should reset timer.
  auto t2 = std::chrono::system_clock::now();
  event.collRecord->getTimingInfo().setCollStartTs(t2);
  plugin->collEventProgressing(event);
  EXPECT_EQ(triggerCallCount.load(), 0)
      << "Timer should reset on startTs change";

  // Sleep past timeout again — now it should fire.
  std::this_thread::sleep_for(std::chrono::milliseconds{60});
  plugin->collEventProgressing(event);
  EXPECT_EQ(triggerCallCount.load(), 1);
}

/* A timeout is reported once for one logical execution. */
TEST_F(WatchdogPluginTest, TimeoutLatchesForExecution) {
  WatchdogPluginConfig config{
      .checkTimeout = true,
      .timeout = std::chrono::milliseconds{30}, // 30ms timeout
      .funcTriggerOnTimeout = mockTriggerOnError,
  };
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event = createCollTraceEvent(1);

  // First timeout
  auto result1 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result1);
  std::this_thread::sleep_for(std::chrono::milliseconds{40});
  auto result2 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result2);
  EXPECT_EQ(triggerCallCount.load(), 1);
  EXPECT_EQ(lastTriggeredCollId.load(), event.collRecord->getCollId());

  /* Further progress for the same execution does not retrigger. */
  std::this_thread::sleep_for(std::chrono::milliseconds{40});
  auto result3 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result3);
  EXPECT_EQ(triggerCallCount.load(), 1);
  EXPECT_EQ(lastTriggeredCollId.load(), event.collRecord->getCollId());
}

TEST_F(WatchdogPluginTest, NewExecutionResetsTimeoutLatch) {
  WatchdogPluginConfig config{
      .checkTimeout = true,
      .timeout = std::chrono::milliseconds{0},
      .funcTriggerOnTimeout = mockTriggerOnError,
  };
  plugin = std::make_unique<WatchdogPlugin>(config);
  auto event = createCollTraceEvent(1);

  event.collRecord->getTimingInfo().setCollStartTs(
      std::chrono::system_clock::now());
  EXPECT_VALUE(plugin->collEventProgressing(event));
  EXPECT_VALUE(plugin->collEventProgressing(event));
  ASSERT_EQ(triggerCallCount.load(), 1);

  event.collRecord->getTimingInfo().setCollStartTs(
      std::chrono::system_clock::now());
  EXPECT_VALUE(plugin->collEventProgressing(event));
  EXPECT_VALUE(plugin->collEventProgressing(event));

  EXPECT_EQ(triggerCallCount.load(), 2);
}

// Test timeout with custom timeout callback
TEST_F(WatchdogPluginTest, TimeoutWithCustomCallback) {
  bool timeoutTriggered = false;
  CollTraceEvent* timeoutEvent = nullptr;

  WatchdogPluginConfig config{
      .checkTimeout = true,
      .timeout = std::chrono::milliseconds{50}, // 50ms timeout
      .funcTriggerOnTimeout =
          [&timeoutTriggered, &timeoutEvent](CollTraceEvent& event) {
            timeoutTriggered = true;
            timeoutEvent = &event;
          },
  };
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event = createCollTraceEvent(1);

  // First call
  auto result1 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result1);
  EXPECT_FALSE(timeoutTriggered);

  // Sleep and trigger timeout
  std::this_thread::sleep_for(std::chrono::milliseconds{60});
  auto result2 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result2);
  EXPECT_TRUE(timeoutTriggered);
  EXPECT_EQ(timeoutEvent, &event);
}

// Test both async error and timeout functionality together
TEST_F(WatchdogPluginTest, AsyncErrorAndTimeoutTogether) {
  bool errorCondition = false;
  int errorTriggerCount = 0;
  int timeoutTriggerCount = 0;

  WatchdogPluginConfig config{
      .checkAsyncError = true,
      .funcIfError = [&errorCondition]() { return errorCondition; },
      .funcTriggerOnError =
          [&errorTriggerCount](CollTraceEvent&) { errorTriggerCount++; },
      .checkTimeout = true,
      .timeout = std::chrono::milliseconds{50},
      .funcTriggerOnTimeout =
          [&timeoutTriggerCount](CollTraceEvent&) { timeoutTriggerCount++; },
  };
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event = createCollTraceEvent(1);

  // First call - no error, start timer
  errorCondition = false;
  auto result1 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result1);
  EXPECT_EQ(errorTriggerCount, 0);
  EXPECT_EQ(timeoutTriggerCount, 0);

  // Second call with error condition
  errorCondition = true;
  auto result2 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result2);
  EXPECT_EQ(errorTriggerCount, 1);
  EXPECT_EQ(timeoutTriggerCount, 0);

  // Sleep and trigger timeout as well
  std::this_thread::sleep_for(std::chrono::milliseconds{60});
  auto result3 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result3);
  EXPECT_EQ(errorTriggerCount, 1); /* Communicator error remains latched. */
  EXPECT_EQ(timeoutTriggerCount, 1); // Timeout also triggered
}

// Test timeout configuration with different timeout values
TEST_F(WatchdogPluginTest, TimeoutConfigurationValues) {
  WatchdogPluginConfig config{
      .checkTimeout = true,
      .timeout = std::chrono::minutes{10}, // Default 10 minute timeout
      .funcTriggerOnTimeout = mockTriggerOnError,
  };
  plugin = std::make_unique<WatchdogPlugin>(config);

  auto event = createCollTraceEvent(1);

  // With 10 minute timeout, short sleep should not trigger
  auto result1 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result1);
  std::this_thread::sleep_for(std::chrono::milliseconds{100});
  auto result2 = plugin->collEventProgressing(event);
  EXPECT_VALUE(result2);
  EXPECT_EQ(triggerCallCount, 0);
}
