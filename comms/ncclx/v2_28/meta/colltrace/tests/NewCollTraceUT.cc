// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/json/json.h>
#include <unordered_map>

#include "nccl.h"
// We need to test internal implementation of the nccl communicator
#include "comm.h"

#include "comms/utils/colltrace/CPUWaitEvent.h"
#include "comms/utils/colltrace/CollTrace.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"
#include "comms/utils/colltrace/tests/MockTypes.h"
#include "meta/commDump.h"

using namespace meta::comms;
using namespace meta::comms::colltrace;

using meta::comms::ncclx::dumpNewCollTrace;

class NewCollTraceUT : public ::testing::Test {
 public:
  NewCollTraceUT() = default;

  std::unique_ptr<MockCollMetadata> createMockCollMetadata() {
    auto metadata = std::make_unique<MockCollMetadata>();
    EXPECT_CALL(*metadata, toDynamic()).WillRepeatedly(testing::Invoke([]() {
      return folly::dynamic::object();
    }));
    return metadata;
  }

  void SetUp() override {
    setenv("NCCL_COLLTRACE", "trace", 0);
    setenv("NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1", 0);
  }
};

TEST_F(NewCollTraceUT, createColltrace) {
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  ncclComm_t comm;
  ncclCommInitRank(&comm, 1, id, 0);

  ASSERT_TRUE(comm->newCollTrace != nullptr);

  ncclCommDestroy(comm);
}

TEST_F(NewCollTraceUT, deleteColltrace) {
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  ncclComm_t comm;
  ncclCommInitRank(&comm, 1, id, 0);

  ASSERT_TRUE(comm->newCollTrace != nullptr);
  auto colltracePtr = comm->newCollTrace;
  auto beforeDeleteCount = colltracePtr.use_count();

  ncclCommDestroy(comm);

  EXPECT_LT(colltracePtr.use_count(), beforeDeleteCount);
}

// A simple test to verify that the dumpNewCollTrace function with ncclComm_t
// More detailed tests are in the DumpNewCollTraceUT.cc
TEST_F(NewCollTraceUT, dumpNewCollTraceWithCollectives) {
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  ncclComm_t comm;
  ncclCommInitRank(&comm, 1, id, 0);

  ASSERT_TRUE(comm->newCollTrace != nullptr);
  auto* collTrace = comm->newCollTrace.get();

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
  EXPECT_EQ(dumpMapPending["CT_pendingColls"], "[]");

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

  ncclCommDestroy(comm);
}
