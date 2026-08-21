// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>

#include <folly/ScopeGuard.h>
#include <folly/String.h>

#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/colltrace/CollMetadata.h"
#include "comms/utils/colltrace/CollMetadataImpl.h"
#include "comms/utils/colltrace/DummyCollTraceHandle.h"
#include "comms/utils/colltrace/GraphCudaWaitEvent.h"
#include "comms/utils/colltrace/plugins/LifecycleEventFeedPlugin.h"
#include "comms/utils/colltrace/tests/MockTypes.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/NcclxLogger.h"
#include "meta/colltrace/CollTraceWrapper.h"

using namespace meta::comms::ncclx;
using namespace meta::comms::colltrace;
using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

TEST(CollTraceWrapperRegistrationTest, GetsOneBasedLatestCollectiveId) {
  constexpr uint64_t kCommId = 17;
  constexpr uint64_t kInternalCollId = 0;
  constexpr uint64_t kExpectedCollId = 1;
  ncclComm comm{};
  LifecycleEventFeedPlugin plugin{LifecycleEventFeedConfig{.commId = kCommId}};
  auto colltrace = std::make_shared<NiceMock<MockCollTrace>>();
  ON_CALL(*colltrace, getPluginByName(_))
      .WillByDefault(Return(static_cast<ICollTracePlugin*>(&plugin)));
  comm.newCollTrace = colltrace;

  uint64_t commId{0};
  uint64_t collId{99};
  EXPECT_EQ(ncclx::colltrace::getCollTraceCommId(&comm, commId), ncclSuccess);
  EXPECT_EQ(commId, kCommId);
  EXPECT_EQ(
      ncclx::colltrace::getLatestCollTraceCollectiveId(&comm, collId),
      ncclSuccess);
  EXPECT_EQ(collId, 0);

  auto event = CollTraceEvent{
      .collRecord = std::make_shared<CollRecord>(
          kInternalCollId, std::make_shared<NiceMock<MockCollMetadata>>()),
  };
  ASSERT_TRUE(plugin.afterCollRecorded(event).hasValue());

  EXPECT_EQ(
      ncclx::colltrace::getLatestCollTraceCollectiveId(&comm, collId),
      ncclSuccess);
  EXPECT_EQ(collId, kExpectedCollId);
  EXPECT_EQ(
      ncclx::colltrace::getLatestCollTraceCollectiveId(&comm, collId),
      ncclSuccess);
  EXPECT_EQ(collId, kExpectedCollId);
}

TEST(CollTraceWrapperLifecycleFeedTest, RejectsDisabledFeed) {
  ncclComm comm{};
  uint64_t commId{0};
  uint64_t collId{0};

  EXPECT_EQ(
      ncclx::colltrace::getCollTraceCommId(nullptr, commId),
      ncclInvalidArgument);
  EXPECT_EQ(
      ncclx::colltrace::getLatestCollTraceCollectiveId(nullptr, collId),
      ncclInvalidArgument);
  EXPECT_EQ(
      ncclx::colltrace::getCollTraceCommId(&comm, commId), ncclInvalidUsage);
  EXPECT_EQ(
      ncclx::colltrace::getLatestCollTraceCollectiveId(&comm, collId),
      ncclInvalidUsage);
}

class CollTraceWrapperUT : public ::testing::Test {
 public:
  void SetUp() override {
    // Create a mock CUDA stream
    CUDACHECK_TEST(cudaStreamCreate(&stream_));
    ncclUniqueId id;
    NCCLCHECK_TEST(ncclGetUniqueId(&id));
    NCCLCHECK_TEST(ncclCommInitRank(&comm_, 1, id, 0));
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      CUDACHECK_TEST(cudaStreamDestroy(stream_));
    }
    if (comm_ != nullptr) {
      ncclCommDestroy(comm_);
    }
  }

 protected:
  cudaStream_t stream_{nullptr};
  ncclComm_t comm_{nullptr};
  std::vector<std::unique_ptr<ncclTaskColl>> collTasks_{};
  std::vector<std::unique_ptr<ncclTaskP2p>> p2pTasks_{};

  ncclTaskColl* createNewCollTask() {
    collTasks_.emplace_back(std::make_unique<ncclTaskColl>());
    return collTasks_.back().get();
  }

  ncclTaskP2p* createNewP2pTask() {
    p2pTasks_.emplace_back(std::make_unique<ncclTaskP2p>());
    return p2pTasks_.back().get();
  }

  // Helper function to create a mock kernel plan with collective task
  ncclKernelPlan createMockKernelPlanWithColl() {
    ncclKernelPlan plan = {};
    plan.comm = comm_;

    // Create a mock collective task
    auto* collTask = createNewCollTask();
    if (collTask == nullptr) {
      NCCLX_LOG_STREAM(FATAL)
          << "Failed to create new collective task" << std::endl;
    }
    collTask->func = ncclFuncAllReduce;
    collTask->algorithm = NCCL_ALGO_RING;
    collTask->protocol = NCCL_PROTO_SIMPLE;
    collTask->opHost = ncclSum;
    collTask->root = 0;
    collTask->count = 1024;
    collTask->datatype = ncclFloat32;
    collTask->sendbuff = reinterpret_cast<void*>(0x1000);
    collTask->recvbuff = reinterpret_cast<void*>(0x2000);
    collTask->nMaxChannels = 4;

    // Initialize the collective task queue with single task
    ncclIntruQueueConstruct(&plan.collTaskQueue);
    ncclIntruQueueEnqueue(&plan.collTaskQueue, collTask);

    // Initialize empty P2P task queue
    ncclIntruQueueConstruct(&plan.p2pTaskQueue);

    return plan;
  }

  // Helper function to create a kernel plan with p2p tasks
  ncclKernelPlan createMockKernelPlanWithP2P() {
    ncclKernelPlan plan = {};
    plan.comm = comm_;

    // Create first p2p task
    auto* p2pTask1 = createNewP2pTask();
    p2pTask1->func = ncclFuncSend;
    p2pTask1->count = 512;
    p2pTask1->datatype = ncclFloat32;
    p2pTask1->root = 1;
    p2pTask1->buff = reinterpret_cast<void*>(0x3000);
    p2pTask1->bytes = 512 * 4; // 512 elements * 4 bytes per float32

    // Create second p2p task
    auto* p2pTask2 = createNewP2pTask();
    p2pTask2->func = ncclFuncSend;
    p2pTask2->count = 512;
    p2pTask2->datatype = ncclFloat32;
    p2pTask2->root = 2;
    p2pTask2->buff = reinterpret_cast<void*>(0x4000);
    p2pTask2->bytes = 512 * 4; // 512 elements * 4 bytes per float32

    // Initialize the collective task queue with multiple tasks
    ncclIntruQueueConstruct(&plan.p2pTaskQueue);
    ncclIntruQueueEnqueue(&plan.p2pTaskQueue, p2pTask1);
    ncclIntruQueueEnqueue(&plan.p2pTaskQueue, p2pTask2);

    return plan;
  }

  // Helper function to create an empty kernel plan
  ncclKernelPlan createEmptyKernelPlan() {
    ncclKernelPlan plan = {};
    plan.comm = comm_;

    // Initialize empty queues
    ncclIntruQueueConstruct(&plan.collTaskQueue);
    ncclIntruQueueConstruct(&plan.p2pTaskQueue);

    return plan;
  }

  // Helper function to create a kernel plan with multiple collective tasks
  ncclKernelPlan createMockKernelPlanWithMultipleColls() {
    ncclKernelPlan plan = {};
    plan.comm = comm_;

    // Create first collective task
    auto* collTask1 = createNewCollTask();
    collTask1->func = ncclFuncAllReduce;
    collTask1->algorithm = NCCL_ALGO_RING;
    collTask1->protocol = NCCL_PROTO_SIMPLE;

    // Create second collective task
    auto* collTask2 = createNewCollTask();
    collTask2->func = ncclFuncBroadcast;
    collTask2->algorithm = NCCL_ALGO_TREE;
    collTask2->protocol = NCCL_PROTO_LL;

    // Initialize the collective task queue with multiple tasks
    ncclIntruQueueConstruct(&plan.collTaskQueue);
    ncclIntruQueueEnqueue(&plan.collTaskQueue, collTask1);
    ncclIntruQueueEnqueue(&plan.collTaskQueue, collTask2);

    // Initialize empty P2P task queue
    ncclIntruQueueConstruct(&plan.p2pTaskQueue);

    return plan;
  }

  // Helper function to create a kernel plan with one collective and one P2P
  // task
  ncclKernelPlan createMockKernelPlanWithCollAndP2p() {
    ncclKernelPlan plan = {};
    plan.comm = comm_;

    // Create a collective task
    auto* collTask = createNewCollTask();
    collTask->func = ncclFuncAllReduce;
    collTask->algorithm = NCCL_ALGO_RING;
    collTask->protocol = NCCL_PROTO_SIMPLE;
    collTask->opHost = ncclSum;
    collTask->root = 0;
    collTask->count = 1024;
    collTask->datatype = ncclFloat32;
    collTask->sendbuff = reinterpret_cast<void*>(0x1000);
    collTask->recvbuff = reinterpret_cast<void*>(0x2000);
    collTask->nMaxChannels = 4;

    // Create a P2P task
    auto* p2pTask = createNewP2pTask();
    p2pTask->func = ncclFuncSend;
    p2pTask->count = 512;
    p2pTask->datatype = ncclFloat32;
    p2pTask->buff = reinterpret_cast<void*>(0x3000);

    // Initialize the collective task queue with single task
    ncclIntruQueueConstruct(&plan.collTaskQueue);
    ncclIntruQueueEnqueue(&plan.collTaskQueue, collTask);

    // Initialize P2P task queue with single task
    ncclIntruQueueConstruct(&plan.p2pTaskQueue);
    ncclIntruQueueEnqueue(&plan.p2pTaskQueue, p2pTask);

    return plan;
  }
};

// Test case for empty plan - should return metadata for empty kernel task
TEST_F(CollTraceWrapperUT, getMetadataFromNcclKernelPlan_EmptyPlan) {
  auto plan = createEmptyKernelPlan();

  auto metadata = getMetadataFromNcclKernelPlan(plan, stream_);

  // Should return valid metadata for empty plan (handled by
  // getEmptyKernelTaskMetadata)
  EXPECT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->getMetadataType(), "CollectiveMetadata");

  // Convert to dynamic to examine the contents
  auto dynamic = metadata->toDynamic();
  EXPECT_EQ(dynamic["opName"].asString(), "Unknown");
  EXPECT_EQ(dynamic["algoName"].asString(), "EmptyKernelTask");
}

TEST_F(CollTraceWrapperUT, PersistentEmptyPlanReturnsNoMetadata) {
  auto plan = createEmptyKernelPlan();
  plan.persistent = true;

  EXPECT_EQ(getMetadataFromNcclKernelPlan(plan, stream_), nullptr);
}

TEST_F(CollTraceWrapperUT, PersistentPlanUsesGraphWaitEvent) {
  auto plan = createMockKernelPlanWithColl();
  plan.persistent = true;
  auto colltrace = std::make_shared<NiceMock<MockCollTrace>>();
  comm_->newCollTrace = colltrace;

  EXPECT_CALL(*colltrace, recordCollective(_, _))
      .WillOnce(
          [](std::unique_ptr<ICollMetadata>,
             std::unique_ptr<ICollWaitEvent> waitEvent)
              -> meta::comms::CommsMaybe<std::shared_ptr<ICollTraceHandle>> {
            EXPECT_NE(
                dynamic_cast<GraphCudaWaitEvent*>(waitEvent.get()), nullptr);
            std::shared_ptr<ICollTraceHandle> handle =
                std::make_shared<DummyCollTraceHandle>();
            return handle;
          });

  EXPECT_NE(getHandleFromNcclKernelPlan(plan, stream_), nullptr);
}

// Test case for single collective - should be supported
TEST_F(CollTraceWrapperUT, getMetadataFromNcclKernelPlan_SingleCollective) {
  auto plan = createMockKernelPlanWithColl();

  auto metadata = getMetadataFromNcclKernelPlan(plan, stream_);

  // Should return valid metadata for single collective
  EXPECT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->getMetadataType(), "CollectiveMetadata");

  // Convert to dynamic to examine the contents
  auto dynamic = metadata->toDynamic();
  EXPECT_EQ(dynamic["opName"].asString(), "AllReduce");
  EXPECT_TRUE(
      dynamic["algoName"].asString().find("Baseline") != std::string::npos);
  EXPECT_EQ(dynamic["sendbuff"].asInt(), 0x1000);
  EXPECT_EQ(dynamic["recvbuff"].asInt(), 0x2000);
  EXPECT_EQ(dynamic["count"].asInt(), 1024);
}

// Test case for grouped P2P - should be supported
TEST_F(CollTraceWrapperUT, getMetadataFromNcclKernelPlan_P2P) {
  auto plan = createMockKernelPlanWithP2P();

  auto metadata = getMetadataFromNcclKernelPlan(plan, stream_);

  // Should return valid metadata for p2p tasks
  EXPECT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->getMetadataType(), "GroupedP2PMetaData");

  // Convert to dynamic to examine the contents
  auto dynamic = metadata->toDynamic();
  EXPECT_EQ(dynamic["opName"].asString(), "Send");
  EXPECT_TRUE(
      dynamic["algoName"].asString().find("Baseline") != std::string::npos);
  EXPECT_EQ(dynamic["dataType"].asString(), "commInt8");
  EXPECT_GT(dynamic["count"].asInt(), 0); // Should have byte count > 0
  EXPECT_TRUE(dynamic.count("ranksInGroupedP2P"));
}

// Test case for multiple collectives - should be supported (GroupedCollP2P)
TEST_F(CollTraceWrapperUT, getMetadataFromNcclKernelPlan_MultipleCollectives) {
  auto plan = createMockKernelPlanWithMultipleColls();

  auto metadata = getMetadataFromNcclKernelPlan(plan, stream_);

  // Should return valid metadata for multiple collectives (handled by
  // getGroupedCollP2PMetadataFromNcclKernelPlan)
  EXPECT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->getMetadataType(), "GroupedCollP2PMetaData");

  // Convert to dynamic to examine the contents
  // For GroupedCollP2P, toDynamic() returns the first collective metadata
  // (based on implementation)
  auto dynamic = metadata->toDynamic();
  EXPECT_EQ(dynamic["opName"].asString(), "AllReduce");
  EXPECT_TRUE(
      dynamic["algoName"].asString().find("Baseline") != std::string::npos);
}

// Test case for one collective and one P2P - should be supported
// (GroupedCollP2P)
TEST_F(CollTraceWrapperUT, getMetadataFromNcclKernelPlan_CollectiveAndP2p) {
  auto plan = createMockKernelPlanWithCollAndP2p();

  auto metadata = getMetadataFromNcclKernelPlan(plan, stream_);

  // Should return valid metadata for collective + P2P combination (handled by
  // getGroupedCollP2PMetadataFromNcclKernelPlan)
  EXPECT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->getMetadataType(), "GroupedCollP2PMetaData");

  // Convert to dynamic to examine the contents
  // For GroupedCollP2P, toDynamic() returns the first collective metadata
  // (based on implementation)
  auto dynamic = metadata->toDynamic();
  EXPECT_EQ(dynamic["opName"].asString(), "AllReduce");
  EXPECT_TRUE(
      dynamic["algoName"].asString().find("Baseline") != std::string::npos);
  EXPECT_EQ(dynamic["sendbuff"].asInt(), 0x1000);
  EXPECT_EQ(dynamic["recvbuff"].asInt(), 0x2000);
  EXPECT_EQ(dynamic["count"].asInt(), 1024);
}

// Test fixture for newCollTraceInit configuration tests
class CollTraceInitConfigTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::vector<std::string>> {
 public:
  void SetUp() override {
    ncclUniqueId id;
    NCCLCHECK_TEST(ncclGetUniqueId(&id));
    NCCLCHECK_TEST(ncclCommInitRank(&comm_, 1, id, 0));
  }

  void TearDown() override {
    if (comm_ != nullptr) {
      ncclCommDestroy(comm_);
    }
  }

 protected:
  ncclComm_t comm_{nullptr};
};

TEST_F(CollTraceInitConfigTest, PullsIdsAndEventsAcrossLifecycleFeeds) {
  constexpr uint64_t kCapturedCollId = 23;
  constexpr uint64_t kExecutionCollId = 29;
  constexpr uint64_t kOtherCollId = 31;
  constexpr uint64_t kExternalCapturedCollId = 24;
  constexpr uint64_t kExternalExecutionCollId = 30;
  constexpr uint64_t kExternalOtherCollId = 32;
  const auto firstEnqueueTs =
      std::chrono::system_clock::time_point{std::chrono::milliseconds{200}};
  const auto otherEnqueueTs =
      std::chrono::system_clock::time_point{std::chrono::milliseconds{100}};
  EnvRAII colltraceGuard(NCCL_COLLTRACE, std::vector<std::string>{"lifecycle"});
  ASSERT_EQ(newCollTraceDestroy(comm_), ncclSuccess);
  comm_->logMetaData.commId = 0;
  ASSERT_EQ(newCollTraceInit(comm_), ncclSuccess);

  ncclComm_t otherComm{nullptr};
  ncclUniqueId otherId;
  NCCLCHECK_TEST(ncclGetUniqueId(&otherId));
  NCCLCHECK_TEST(ncclCommInitRank(&otherComm, 1, otherId, 0));
  auto otherCommGuard = folly::makeGuard([&] {
    if (otherComm != nullptr) {
      ncclCommDestroy(otherComm);
    }
  });
  ASSERT_EQ(newCollTraceDestroy(otherComm), ncclSuccess);
  otherComm->logMetaData = comm_->logMetaData;
  ASSERT_EQ(newCollTraceInit(otherComm), ncclSuccess);

  auto getPlugin = [](ncclComm_t comm) {
    return dynamic_cast<LifecycleEventFeedPlugin*>(
        comm->newCollTrace->getPluginByName(
            std::string{
                LifecycleEventFeedPlugin::kLifecycleEventFeedPluginName}));
  };
  auto* plugin = getPlugin(comm_);
  auto* otherPlugin = getPlugin(otherComm);
  ASSERT_NE(plugin, nullptr);
  ASSERT_NE(otherPlugin, nullptr);

  auto executionEvent = CollTraceEvent{
      .collRecord = std::make_shared<CollRecord>(
          kExecutionCollId, std::make_shared<NiceMock<MockCollMetadata>>()),
      .replayId = 1,
      .capturedCollId = kCapturedCollId,
  };
  executionEvent.collRecord->getTimingInfo().setCollEnqueueTs(firstEnqueueTs);
  ASSERT_TRUE(plugin->afterCollRecorded(executionEvent).hasValue());
  ASSERT_TRUE(plugin->afterCollKernelScheduled(executionEvent).hasValue());

  auto otherEvent = CollTraceEvent{
      .collRecord = std::make_shared<CollRecord>(
          kOtherCollId, std::make_shared<NiceMock<MockCollMetadata>>()),
  };
  otherEvent.collRecord->getTimingInfo().setCollEnqueueTs(otherEnqueueTs);
  ASSERT_TRUE(otherPlugin->afterCollRecorded(otherEvent).hasValue());
  ASSERT_TRUE(otherPlugin->afterCollKernelScheduled(otherEvent).hasValue());

  uint64_t commId{0};
  uint64_t otherCommId{0};
  uint64_t collId{0};
  uint64_t otherCollId{0};
  EXPECT_EQ(ncclx::colltrace::getCollTraceCommId(comm_, commId), ncclSuccess);
  EXPECT_EQ(
      ncclx::colltrace::getCollTraceCommId(otherComm, otherCommId),
      ncclSuccess);
  EXPECT_NE(commId, 0);
  EXPECT_NE(otherCommId, 0);
  EXPECT_NE(commId, otherCommId);
  EXPECT_EQ(
      ncclx::colltrace::getLatestCollTraceCollectiveId(comm_, collId),
      ncclSuccess);
  EXPECT_EQ(
      ncclx::colltrace::getLatestCollTraceCollectiveId(otherComm, otherCollId),
      ncclSuccess);
  EXPECT_EQ(collId, kExternalCapturedCollId);
  EXPECT_EQ(otherCollId, kExternalOtherCollId);

  std::vector<ncclx::colltrace::LifecycleEvent> events;
  EXPECT_EQ(ncclx::colltrace::drainUnreadLifecycleEvents(events), ncclSuccess);
  ASSERT_EQ(events.size(), 2);

  EXPECT_EQ(events[0].commId, otherCommId);
  EXPECT_EQ(events[0].collId, kExternalOtherCollId);
  EXPECT_EQ(events[0].executionCollId, kExternalOtherCollId);
  EXPECT_DOUBLE_EQ(events[0].timestamp, 0.1);

  EXPECT_EQ(events[1].commId, commId);
  EXPECT_EQ(events[1].collId, kExternalCapturedCollId);
  EXPECT_EQ(events[1].executionCollId, kExternalExecutionCollId);
  EXPECT_EQ(events[1].replayId, 1);
  EXPECT_EQ(events[1].eventType, ncclx::colltrace::LifecycleEventType::Enqueue);
  EXPECT_DOUBLE_EQ(events[1].timestamp, 0.2);

  EXPECT_EQ(ncclx::colltrace::drainUnreadLifecycleEvents(events), ncclSuccess);
  EXPECT_TRUE(events.empty());
}

TEST_P(CollTraceInitConfigTest, ConfigCombinations) {
  auto config = GetParam();

  // Use EnvRAII for clean cvar override (always use new colltrace)
  EnvRAII colltraceGuard(NCCL_COLLTRACE, config);

  // Compute expected values based on config input
  const bool expectAll =
      std::any_of(config.begin(), config.end(), [](const auto& s) {
        return s == "ALL" || s == "all";
      });
  const bool expectAlgoStats = expectAll ||
      std::find(config.begin(), config.end(), "algostat") != config.end();
  const bool expectNewCollTrace =
      expectAll || std::any_of(config.begin(), config.end(), [](const auto& s) {
        return s == "lifecycle" || s == "verbose" || s == "trace";
      });

  // Reset any existing state
  comm_->algoStats.reset();
  ASSERT_EQ(newCollTraceDestroy(comm_), ncclSuccess);

  auto result = newCollTraceInit(comm_);

  EXPECT_EQ(result, ncclSuccess);
  EXPECT_EQ(comm_->algoStats != nullptr, expectAlgoStats);
  EXPECT_EQ(comm_->newCollTrace != nullptr, expectNewCollTrace);
}

TEST_F(CollTraceInitConfigTest, AllAliasesEnableProductionFeatures) {
  for (const auto* mode : {"ALL", "all"}) {
    SCOPED_TRACE(mode);
    EnvRAII colltraceGuard(NCCL_COLLTRACE, std::vector<std::string>{mode});
    comm_->algoStats.reset();
    ASSERT_EQ(newCollTraceDestroy(comm_), ncclSuccess);

    ASSERT_EQ(newCollTraceInit(comm_), ncclSuccess);
    EXPECT_NE(comm_->algoStats, nullptr);
    ASSERT_NE(comm_->newCollTrace, nullptr);
    EXPECT_NE(comm_->newCollTrace->getPluginByName("CommDumpPlugin"), nullptr);
    EXPECT_NE(
        comm_->newCollTrace->getPluginByName(
            std::string{
                LifecycleEventFeedPlugin::kLifecycleEventFeedPluginName}),
        nullptr);
  }
}

TEST_F(CollTraceInitConfigTest, RejectsUnknownModesWithoutPartialInit) {
  EnvRAII colltraceGuard(
      NCCL_COLLTRACE, std::vector<std::string>{"ALL", "bogus"});
  comm_->algoStats.reset();
  ASSERT_EQ(newCollTraceDestroy(comm_), ncclSuccess);

  EXPECT_EQ(newCollTraceInit(comm_), ncclInvalidArgument);
  EXPECT_EQ(comm_->algoStats, nullptr);
  EXPECT_EQ(comm_->newCollTrace, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    ConfigCombinations,
    CollTraceInitConfigTest,
    ::testing::Values(
        std::vector<std::string>{"algostat"},
        std::vector<std::string>{"trace"},
        std::vector<std::string>{"verbose"},
        std::vector<std::string>{"lifecycle"},
        std::vector<std::string>{"algostat", "trace"},
        std::vector<std::string>{"ALL"},
        std::vector<std::string>{"all"},
        std::vector<std::string>{}),
    [](const ::testing::TestParamInfo<std::vector<std::string>>& info) {
      if (info.param.empty()) {
        return std::string("Empty");
      }
      return folly::join("_", info.param);
    });
