// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/MPMCQueue.h>
#include <folly/Synchronized.h>
#include <folly/Unit.h>

#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"
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

// Helper macro for checking if a CommsMaybe has an error
#define EXPECT_ERROR(cmd)        \
  {                              \
    const auto& res = cmd;       \
    EXPECT_TRUE(res.hasError()); \
  }

// Test fixture for CommDumpPlugin tests
// CommDumpPlugin inherits from ICollTracePlugin defined in CollTracePlugin.h
class CommDumpPluginTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create CommDumpPlugin
    plugin = std::make_unique<CommDumpPlugin>();
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

  std::unique_ptr<CommDumpPlugin> plugin;
};

// Test constructor and getName
TEST_F(CommDumpPluginTest, ConstructorAndGetName) {
  // Verify that the plugin was created successfully
  EXPECT_NE(plugin.get(), nullptr);

  // Check that getName returns the expected value
  EXPECT_EQ(plugin->getName(), "CommDumpPlugin");
}

// Test beforeCollKernelScheduled (should be a no-op)
TEST_F(CommDumpPluginTest, BeforeCollKernelScheduled) {
  auto event = createCollTraceEvent(1);

  // Call beforeCollKernelScheduled and verify it succeeds
  auto result = plugin->beforeCollKernelScheduled(event);
  EXPECT_VALUE(result);
}

// Test afterCollKernelScheduled
TEST_F(CommDumpPluginTest, AfterCollKernelScheduledWithDump) {
  auto event = createCollTraceEvent(1);

  // Call afterCollKernelScheduled and verify it succeeds
  EXPECT_VALUE(plugin->beforeCollKernelScheduled(event));
  EXPECT_VALUE(plugin->afterCollKernelScheduled(event));

  auto dump = plugin->dump();
  EXPECT_VALUE(dump);
  EXPECT_TRUE(dump.value().pastColls.empty());
  ASSERT_NE(dump.value().currentColl, nullptr);
  EXPECT_EQ(dump.value().currentColl.get(), event.collRecord.get());
}

TEST_F(CommDumpPluginTest, NullAfterCollKernel) {
  CollTraceEvent nullEvent;
  nullEvent.collRecord = nullptr;

  auto nullResult = plugin->afterCollKernelScheduled(nullEvent);
  EXPECT_FALSE(nullResult.hasValue());
  EXPECT_EQ(nullResult.error().errorCode, commInternalError);
}

// Test afterCollKernelStart
TEST_F(CommDumpPluginTest, AfterCollKernelStart) {
  auto event = createCollTraceEvent(1);

  // First, call afterCollKernelScheduled to enqueue the event
  EXPECT_VALUE(plugin->beforeCollKernelScheduled(event));
  EXPECT_VALUE(plugin->afterCollKernelScheduled(event));

  // Then call afterCollKernelStart and verify it succeeds
  EXPECT_VALUE(plugin->afterCollKernelStart(event));

  // Test the dump to be correct. We should see empty pastColls and
  // pendingColls, with currentColl be the same as what contains in the event
  auto dump = plugin->dump();
  EXPECT_VALUE(dump);
  EXPECT_TRUE(dump.value().pastColls.empty());
  EXPECT_TRUE(dump.value().pendingColls.empty());
  EXPECT_EQ(dump.value().currentColl.get(), event.collRecord.get());
}

// Test afterCollKernelEnd
TEST_F(CommDumpPluginTest, AfterCollKernelEnd) {
  auto event = createCollTraceEvent(1);

  // First, call afterCollKernelScheduled to enqueue the event
  EXPECT_VALUE(plugin->beforeCollKernelScheduled(event));
  EXPECT_VALUE(plugin->afterCollKernelScheduled(event));

  // Then call afterCollKernelStart to move it to currentColl
  EXPECT_VALUE(plugin->afterCollKernelStart(event));

  // Finally call afterCollKernelEnd and verify it succeeds
  EXPECT_VALUE(plugin->afterCollKernelEnd(event));

  // Test the dump to be correct. We should see the event in pastColls and
  // currentColl and pendingColls should be empty
  auto dump = plugin->dump();
  EXPECT_VALUE(dump);
  EXPECT_EQ(dump.value().pastColls.size(), 1);
  EXPECT_EQ(dump.value().pastColls.front().get(), event.collRecord.get());
  EXPECT_TRUE(dump.value().pendingColls.empty());
  EXPECT_EQ(dump.value().currentColl, nullptr);
}

// Test dump method
TEST_F(CommDumpPluginTest, Dump) {
  // Call dump and verify it succeeds
  auto dumpResult = plugin->dump();
  EXPECT_VALUE(dumpResult);

  // Verify that the dump is initially empty
  auto& dump = dumpResult.value();
  EXPECT_TRUE(dump.pastColls.empty());
  EXPECT_EQ(dump.currentColl, nullptr);
  EXPECT_TRUE(dump.pendingColls.empty());

  // Process a complete collective
  auto event = createCollTraceEvent(1);
  EXPECT_VALUE(plugin->afterCollKernelScheduled(event));
  EXPECT_VALUE(plugin->afterCollKernelStart(event));
  EXPECT_VALUE(plugin->afterCollKernelEnd(event));

  // Call dump again and verify the collective is in pastColls
  auto dumpResult2 = plugin->dump();
  EXPECT_VALUE(dumpResult2);

  auto& dump2 = dumpResult2.value();
  EXPECT_EQ(dump2.pastColls.size(), 1);
  EXPECT_EQ(dump2.currentColl, nullptr);
  EXPECT_TRUE(dump2.pendingColls.empty());
  EXPECT_EQ(dump2.pastColls.front()->getCollId(), 1);
}

// Test multiple collectives
TEST_F(CommDumpPluginTest, MultipleCollectives) {
  // Process first collective
  auto event1 = createCollTraceEvent(1);
  EXPECT_VALUE(plugin->afterCollKernelScheduled(event1));
  EXPECT_VALUE(plugin->afterCollKernelStart(event1));

  // Process second collective (schedule only)
  auto event2 = createCollTraceEvent(2);
  EXPECT_VALUE(plugin->afterCollKernelScheduled(event2));

  // Dump and verify state
  auto dumpResult = plugin->dump();
  EXPECT_VALUE(dumpResult);

  auto& dump = dumpResult.value();
  EXPECT_TRUE(dump.pastColls.empty());
  EXPECT_EQ(dump.currentColl.get(), event1.collRecord.get());
  EXPECT_EQ(dump.currentColl->getCollId(), 1);
  EXPECT_EQ(dump.pendingColls.size(), 1);
  EXPECT_EQ(dump.pendingColls.front()->getCollId(), 2);
  EXPECT_EQ(dump.pendingColls.front().get(), event2.collRecord.get());

  // Complete first collective
  EXPECT_VALUE(plugin->afterCollKernelEnd(event1));

  // Start second collective
  EXPECT_VALUE(plugin->afterCollKernelStart(event2));

  // Dump and verify state
  auto dumpResult2 = plugin->dump();
  EXPECT_VALUE(dumpResult2);

  auto& dump2 = dumpResult2.value();
  EXPECT_EQ(dump2.pastColls.size(), 1);
  EXPECT_EQ(dump2.pastColls.front()->getCollId(), 1);
  EXPECT_NE(dump2.currentColl, nullptr);
  EXPECT_EQ(dump2.currentColl->getCollId(), 2);
  EXPECT_TRUE(dump2.pendingColls.empty());

  // Complete second collective
  EXPECT_VALUE(plugin->afterCollKernelEnd(event2));

  // Dump and verify state
  auto dumpResult3 = plugin->dump();
  EXPECT_VALUE(dumpResult3);

  auto& dump3 = dumpResult3.value();
  EXPECT_EQ(dump3.pastColls.size(), 2);
  EXPECT_EQ(dump3.pastColls.front()->getCollId(), 1);
  EXPECT_EQ(dump3.pastColls.back()->getCollId(), 2);
  EXPECT_EQ(dump3.currentColl, nullptr);
  EXPECT_TRUE(dump3.pendingColls.empty());
}

// Test error cases
TEST_F(CommDumpPluginTest, ErrorCases) {
  // Create events with different IDs
  auto event1 = createCollTraceEvent(1);
  auto event2 = createCollTraceEvent(2);

  // Schedule event1
  EXPECT_VALUE(plugin->afterCollKernelScheduled(event1));

  // Try to start event2 (should fail due to mismatch)
  auto result = plugin->afterCollKernelStart(event2);
  EXPECT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().errorCode, commInternalError);

  // Start event1 correctly
  EXPECT_VALUE(plugin->afterCollKernelStart(event1));

  // Try to end event2 (should fail due to mismatch)
  auto result2 = plugin->afterCollKernelEnd(event2);
  EXPECT_FALSE(result2.hasValue());
  EXPECT_EQ(result2.error().errorCode, commInternalError);
}

// Test configurable pending queue size
TEST_F(CommDumpPluginTest, ConfigurablePendingQueueSize) {
  constexpr int kTestPendingQueueSize = 10;
  // Test with a small pending queue size
  auto smallQueuePlugin = std::make_unique<CommDumpPlugin>(CommDumpConfig{
      .pendingCollSize = kTestPendingQueueSize,
  });

  // The first kTestPendingQueueSize events should succeed
  for (int i = 0; i < kTestPendingQueueSize; ++i) {
    auto event = createCollTraceEvent(i);
    EXPECT_VALUE(smallQueuePlugin->beforeCollKernelScheduled(event));
    EXPECT_VALUE(smallQueuePlugin->afterCollKernelScheduled(event));
  }

  // The next event should fail since the queue is full
  auto failEvent = createCollTraceEvent(kTestPendingQueueSize);
  EXPECT_VALUE(smallQueuePlugin->beforeCollKernelScheduled(failEvent));
  auto result = smallQueuePlugin->afterCollKernelScheduled(failEvent);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().errorCode, commInternalError);

  // Verify both events are accessible through dump
  // Note: dump() moves the first pending event to currentColl if currentColl is
  // null
  auto dump = smallQueuePlugin->dump();
  EXPECT_VALUE(dump);
  EXPECT_EQ(
      dump.value().pendingColls.size(),
      kTestPendingQueueSize - 1); // One will be marked as currentColl
  EXPECT_NE(dump.value().currentColl, nullptr); // First one moved to current
  EXPECT_EQ(dump.value().currentColl->getCollId(), 0);
  if (dump.value().pendingColls.size() == kTestPendingQueueSize - 1) {
    for (int i = 0; i < kTestPendingQueueSize - 1; ++i) {
      EXPECT_EQ(dump.value().pendingColls[i]->getCollId(), i + 1); // IDs 1-3
    }
  }

  // After dumping we should be able to enqueue again
  for (int i = 0; i < kTestPendingQueueSize; ++i) {
    auto event = createCollTraceEvent(i);
    EXPECT_VALUE(smallQueuePlugin->beforeCollKernelScheduled(event));
    EXPECT_VALUE(smallQueuePlugin->afterCollKernelScheduled(event));
  }
}
