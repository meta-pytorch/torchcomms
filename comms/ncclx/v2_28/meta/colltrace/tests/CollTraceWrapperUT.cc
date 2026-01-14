// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/colltrace/CollMetadata.h"
#include "comms/utils/colltrace/CollMetadataImpl.h"
#include "meta/colltrace/CollTraceWrapper.h"

using namespace meta::comms::ncclx;
using namespace meta::comms::colltrace;

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
      XLOG(FATAL) << "Failed to create new collective task" << std::endl;
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
