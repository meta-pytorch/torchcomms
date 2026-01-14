// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/json/json.h>
#include <unordered_map>

#include "comms/utils/colltrace/CPUWaitEvent.h"
#include "comms/utils/colltrace/CollTrace.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"
#include "comms/utils/colltrace/tests/MockTypes.h"
#include "meta/commDump.h"

using namespace meta::comms;
using namespace meta::comms::colltrace;
using meta::comms::ncclx::dumpNewCollTrace;

std::unique_ptr<MockCollMetadata> createMockCollMetadata() {
  auto metadata = std::make_unique<MockCollMetadata>();
  EXPECT_CALL(*metadata, toDynamic()).WillRepeatedly(testing::Invoke([]() {
    return folly::dynamic::object();
  }));
  return metadata;
}

TEST(DumpNewCollTraceUT, dumpNewCollTraceEmptyState) {
  // Create a CommDumpPlugin
  auto plugin = std::make_unique<CommDumpPlugin>();
  std::vector<std::unique_ptr<ICollTracePlugin>> plugins;
  plugins.push_back(std::move(plugin));

  // Create a CollTrace instance with the plugin
  auto collTrace = std::make_unique<meta::comms::colltrace::CollTrace>(
      CollTraceConfig{},
      CommLogData{},
      []() -> CommsMaybeVoid { return folly::unit; },
      std::move(plugins));

  // Call dumpNewCollTrace on the empty CollTrace
  auto dumpMap = dumpNewCollTrace(*collTrace);

  // Verify the map has the expected keys and values for an empty state
  ASSERT_FALSE(dumpMap.empty());
  EXPECT_EQ(dumpMap.size(), 3);
  EXPECT_TRUE(dumpMap.find("CT_pastColls") != dumpMap.end());
  EXPECT_TRUE(dumpMap.find("CT_pendingColls") != dumpMap.end());
  EXPECT_TRUE(dumpMap.find("CT_currentColl") != dumpMap.end());

  // Empty state should have empty arrays and null current collective
  EXPECT_EQ(dumpMap["CT_pastColls"], "[]");
  EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");
}

TEST(DumpNewCollTraceUT, dumpNewCollTraceWithCollectives) {
  // Create a CommDumpPlugin
  auto plugin = std::make_unique<CommDumpPlugin>();
  std::vector<std::unique_ptr<ICollTracePlugin>> plugins;
  plugins.push_back(std::move(plugin));

  // Create a CollTrace instance with the plugin
  auto collTrace = std::make_unique<meta::comms::colltrace::CollTrace>(
      CollTraceConfig{},
      CommLogData{},
      []() -> CommsMaybeVoid { return folly::unit; },
      std::move(plugins));

  // Create metadata for a collective
  auto metadata = createMockCollMetadata();

  // Create a CPU wait event to manipulate the state
  auto waitEvent = std::make_unique<CPUWaitEvent>();

  // Record a collective and get the handle
  auto handleMaybe =
      collTrace->recordCollective(std::move(metadata), std::move(waitEvent));
  ASSERT_TRUE(handleMaybe.hasValue());
  auto handle = handleMaybe.value();

  // Trigger the collective lifecycle events
  ASSERT_TRUE(handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel)
                  .hasValue());
  ASSERT_TRUE(handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel)
                  .hasValue());

  // Sleep briefly to allow the CollTrace thread to process the events
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // At this point, the collective should be in the pending state. However,
  // since currently we will treat the first pending collective as the current
  // collective, we will see the collective in the current state.
  auto dumpMapPending = dumpNewCollTrace(*collTrace);
  ASSERT_FALSE(dumpMapPending.empty());

  // Verify pastColls is empty and currentColl should have one entry
  EXPECT_EQ(dumpMapPending["CT_pastColls"], "[]");
  EXPECT_NE(dumpMapPending["CT_currentColl"], "null");
  // Verify pendingColls is empty since the pending collective was moved to
  // current
  auto pendingCollsJson = folly::parseJson(dumpMapPending["CT_pendingColls"]);
  EXPECT_EQ(pendingCollsJson.size(), 0);

  // Trigger kernel start
  ASSERT_TRUE(
      handle->trigger(CollTraceHandleTriggerState::KernelStarted).hasValue());

  // Sleep briefly to allow the CollTrace thread to process the events
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Now the collective should be in the current state
  auto dumpMapCurrent = dumpNewCollTrace(*collTrace);
  ASSERT_FALSE(dumpMapCurrent.empty());

  // Verify currentColl is set and pendingColls is empty
  EXPECT_EQ(dumpMapCurrent["CT_pastColls"], "[]");
  EXPECT_EQ(dumpMapCurrent["CT_pendingColls"], "[]");
  EXPECT_NE(dumpMapCurrent["CT_currentColl"], "null");

  // Trigger kernel finish
  ASSERT_TRUE(
      handle->trigger(CollTraceHandleTriggerState::KernelFinished).hasValue());

  // Sleep briefly to allow the CollTrace thread to process the events
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Now the collective should be in the past state
  auto dumpMapPast = dumpNewCollTrace(*collTrace);
  ASSERT_FALSE(dumpMapPast.empty());

  // Verify pastColls has one entry and currentColl is null
  EXPECT_EQ(dumpMapPast["CT_currentColl"], "null");
  auto pastCollsJson = folly::parseJson(dumpMapPast["CT_pastColls"]);
  EXPECT_EQ(pastCollsJson.size(), 1);
}

TEST(DumpNewCollTraceUT, dumpNewCollTraceMultipleCollectives) {
  // Create a CommDumpPlugin
  auto plugin = std::make_unique<CommDumpPlugin>();
  std::vector<std::unique_ptr<ICollTracePlugin>> plugins;
  plugins.push_back(std::move(plugin));

  // Create a CollTrace instance with the plugin
  auto collTrace = std::make_unique<meta::comms::colltrace::CollTrace>(
      CollTraceConfig{},
      CommLogData{},
      []() -> CommsMaybeVoid { return folly::unit; },
      std::move(plugins));

  // Process first collective completely
  {
    auto metadata = createMockCollMetadata();
    auto waitEvent = std::make_unique<CPUWaitEvent>();
    auto handleMaybe =
        collTrace->recordCollective(std::move(metadata), std::move(waitEvent));
    ASSERT_TRUE(handleMaybe.hasValue());
    auto handle = handleMaybe.value();

    ASSERT_TRUE(
        handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel)
            .hasValue());
    ASSERT_TRUE(handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel)
                    .hasValue());
    ASSERT_TRUE(
        handle->trigger(CollTraceHandleTriggerState::KernelStarted).hasValue());
    ASSERT_TRUE(handle->trigger(CollTraceHandleTriggerState::KernelFinished)
                    .hasValue());

    // Sleep briefly to allow the CollTrace thread to process the events
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Start second collective but don't finish it
  {
    auto metadata = createMockCollMetadata();
    auto waitEvent = std::make_unique<CPUWaitEvent>();
    auto handleMaybe =
        collTrace->recordCollective(std::move(metadata), std::move(waitEvent));
    ASSERT_TRUE(handleMaybe.hasValue());
    auto handle = handleMaybe.value();

    ASSERT_TRUE(
        handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel)
            .hasValue());
    ASSERT_TRUE(handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel)
                    .hasValue());
    ASSERT_TRUE(
        handle->trigger(CollTraceHandleTriggerState::KernelStarted).hasValue());
    // Don't trigger KernelFinished

    // Sleep briefly to allow the CollTrace thread to process the events
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Enqueue third collective but don't start it
  {
    auto metadata = createMockCollMetadata();
    auto waitEvent = std::make_unique<CPUWaitEvent>();
    auto handleMaybe =
        collTrace->recordCollective(std::move(metadata), std::move(waitEvent));
    ASSERT_TRUE(handleMaybe.hasValue());
    auto handle = handleMaybe.value();

    ASSERT_TRUE(
        handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel)
            .hasValue());
    ASSERT_TRUE(handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel)
                    .hasValue());
    // Don't trigger KernelStarted

    // Sleep briefly to allow the CollTrace thread to process the events
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Dump the state and verify
  auto dumpMap = dumpNewCollTrace(*collTrace);
  ASSERT_FALSE(dumpMap.empty());

  // Verify pastColls has one entry
  auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
  EXPECT_EQ(pastCollsJson.size(), 1);

  // Verify currentColl is the second collective
  EXPECT_NE(dumpMap["CT_currentColl"], "null");

  // Verify pendingColls has the third collective
  auto pendingCollsJson = folly::parseJson(dumpMap["CT_pendingColls"]);
  EXPECT_EQ(pendingCollsJson.size(), 1);
}
