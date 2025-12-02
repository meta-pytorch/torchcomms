// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/tracing/CollTraceWrapper.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/utils/colltrace/CPUWaitEvent.h"
#include "comms/utils/colltrace/CollMetadataImpl.h"
#include "comms/utils/colltrace/DummyCollTraceHandle.h"
#include "comms/utils/colltrace/GenericMetadata.h"
#include "comms/utils/colltrace/tests/MockTypes.h"
#include "comms/utils/commSpecs.h"

using namespace meta::comms::colltrace;
using namespace meta::comms;
using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

// Forward declarations for the functions we're testing
namespace meta::comms::colltrace {
std::shared_ptr<ICollTraceHandle> getNewCollTraceHandle(
    CtranComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig);

std::unique_ptr<ICollMetadata> getMetadata(
    CtranComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig);

CollectiveMetadata getCollectiveMetadata(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    const uint64_t opCount);

GroupedP2PMetaData getGroupedP2PMetaData(
    const int curRank,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    const uint64_t opCount);
} // namespace meta::comms::colltrace

namespace {
// Helper functions for testing
std::unique_ptr<OpElem>
createMockOpElem(OpElem::opType type, CtranComm* comm, uint64_t opCount) {
  return std::make_unique<OpElem>(type, comm, opCount);
}
} // namespace

auto opCountStore = std::make_unique<uint64_t>(10);
// Mock classes for testing
class MockCtranComm : public CtranComm {
 public:
  MockCtranComm() {
    // Initialize required members
    logMetaData_ = CommLogData{
        .commId = 123,
        .commHash = 456,
        .commDesc = "test_comm",
        .rank = 2,
        .nRanks = 8};

    opCount_ = opCountStore.get();
    colltraceNew_ = nullptr; // Will be set in tests
  }
};

class CollTraceWrapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    comm_ = std::make_unique<MockCtranComm>();
    mockCollTrace_ = std::make_unique<MockCollTrace>();

    // Create a simple kernel config for testing
    kernelConfig_ = std::make_unique<KernelConfig>(
        KernelConfig::KernelType::ALLREDUCE,
        reinterpret_cast<cudaStream_t>(0x1234),
        "test_algo",
        5);

    // Set up kernel args for collective operations
    kernelConfig_->args.collective.allreduce.sendbuff =
        reinterpret_cast<void*>(0x5678);
    kernelConfig_->args.collective.allreduce.recvbuff =
        reinterpret_cast<void*>(0x9ABC);
    kernelConfig_->args.collective.allreduce.datatype =
        commDataType_t::commFloat32;
    kernelConfig_->args.collective.allreduce.count = 1024;
  }

  std::unique_ptr<MockCtranComm> comm_;
  std::shared_ptr<MockCollTrace> mockCollTrace_;
  std::unique_ptr<KernelConfig> kernelConfig_;
  std::vector<std::unique_ptr<struct OpElem>> opGroup_;
};

// Tests for getNewCollTraceHandle
TEST_F(
    CollTraceWrapperTest,
    GetNewCollTraceHandle_NullCollTrace_ReturnsDummyHandle) {
  // Set colltrace to null
  comm_->colltraceNew_ = nullptr;

  auto handle = getNewCollTraceHandle(comm_.get(), opGroup_, *kernelConfig_);

  ASSERT_NE(handle, nullptr);
  // Should return a DummyCollTraceHandle when colltrace is null
  auto dummyHandle = std::dynamic_pointer_cast<DummyCollTraceHandle>(handle);
  EXPECT_NE(dummyHandle, nullptr);
}

// TODO: We need to refactor getWaitEvent function to make it mockable.
// Currently it will construct CudaEvent if size(opGroup) == 0. Also we need to
// mock cudaStreamCapture
TEST_F(
    CollTraceWrapperTest,
    DISABLED_GetNewCollTraceHandle_CollTraceError_ReturnsDummyHandle) {
  // Set up mock colltrace
  comm_->colltraceNew_ = mockCollTrace_;

  // Create a simple SEND operation
  auto sendOp = createMockOpElem(OpElem::SEND, comm_.get(), 5);
  sendOp->send.sendbuff = reinterpret_cast<void*>(0x1000);
  sendOp->send.count = 100;
  sendOp->send.datatype = commDataType_t::commFloat32;
  sendOp->send.peerRank = 1;

  std::vector<std::unique_ptr<struct OpElem>> p2pOpGroup;
  p2pOpGroup.push_back(std::move(sendOp));

  // Set up expectation for recordCollective to return error
  EXPECT_CALL(*mockCollTrace_, recordCollective(_, _))
      .WillOnce(Return(
          folly::makeUnexpected<CommsError>(
              CommsError("Mocking error", commInternalError))));

  auto handle = getNewCollTraceHandle(comm_.get(), p2pOpGroup, *kernelConfig_);

  ASSERT_NE(handle, nullptr);
  // Should return a DummyCollTraceHandle when recordCollective fails
  auto dummyHandle = std::dynamic_pointer_cast<DummyCollTraceHandle>(handle);
  EXPECT_NE(dummyHandle, nullptr);
}

// Tests for getMetadata
TEST_F(CollTraceWrapperTest, GetMetadata_CollectiveOperation_Success) {
  auto metadata = getMetadata(comm_.get(), opGroup_, *kernelConfig_);

  ASSERT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->getMetadataType(), "CollectiveMetadata");
}

TEST_F(CollTraceWrapperTest, GetMetadata_P2POperation_Success) {
  // Create a P2P kernel config
  KernelConfig p2pConfig(
      KernelConfig::KernelType::SEND,
      reinterpret_cast<cudaStream_t>(0x1234),
      "test_p2p_algo",
      5);

  // Create a simple SEND operation
  auto sendOp = createMockOpElem(OpElem::SEND, comm_.get(), 5);
  sendOp->send.sendbuff = reinterpret_cast<void*>(0x1000);
  sendOp->send.count = 100;
  sendOp->send.datatype = commDataType_t::commFloat32;
  sendOp->send.peerRank = 1;

  std::vector<std::unique_ptr<struct OpElem>> p2pOpGroup;
  p2pOpGroup.push_back(std::move(sendOp));

  auto metadata = getMetadata(comm_.get(), p2pOpGroup, p2pConfig);

  ASSERT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->getMetadataType(), "GroupedP2PMetaData");
}

// Tests for getCollectiveMetadata
TEST_F(CollTraceWrapperTest, GetCollectiveMetadata_AllReduce_CorrectMetadata) {
  uint64_t opCount = 42;

  auto metadata = getCollectiveMetadata(opGroup_, *kernelConfig_, opCount);

  EXPECT_EQ(metadata.opName, "AllReduce");
  EXPECT_EQ(metadata.algoName, "test_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(
      metadata.sendbuff,
      reinterpret_cast<uintptr_t>(
          kernelConfig_->args.collective.allreduce.sendbuff));
  EXPECT_EQ(
      metadata.recvbuff,
      reinterpret_cast<uintptr_t>(
          kernelConfig_->args.collective.allreduce.recvbuff));
  EXPECT_EQ(metadata.dataType, commDataType_t::commFloat32);
  EXPECT_EQ(metadata.count, 1024);
}

TEST_F(CollTraceWrapperTest, GetCollectiveMetadata_AllGather_CorrectMetadata) {
  // Create AllGather kernel config
  KernelConfig allGatherConfig(
      KernelConfig::KernelType::ALLGATHER,
      reinterpret_cast<cudaStream_t>(0x1234),
      "allgather_algo",
      5);

  allGatherConfig.args.collective.allgather.sendbuff =
      reinterpret_cast<void*>(0x1111);
  allGatherConfig.args.collective.allgather.recvbuff =
      reinterpret_cast<void*>(0x2222);
  allGatherConfig.args.collective.allgather.datatype =
      commDataType_t::commInt32;
  allGatherConfig.args.collective.allgather.count = 512;

  uint64_t opCount = 33;

  auto metadata = getCollectiveMetadata(opGroup_, allGatherConfig, opCount);

  EXPECT_EQ(metadata.opName, "AllGather");
  EXPECT_EQ(metadata.algoName, "allgather_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(metadata.sendbuff, 0x1111);
  EXPECT_EQ(metadata.recvbuff, 0x2222);
  EXPECT_EQ(metadata.dataType, commDataType_t::commInt32);
  EXPECT_EQ(metadata.count, 512);
}

TEST_F(CollTraceWrapperTest, GetCollectiveMetadata_AllToAll_CorrectMetadata) {
  // Create AllToAll kernel config
  KernelConfig allToAllConfig(
      KernelConfig::KernelType::ALLTOALL,
      reinterpret_cast<cudaStream_t>(0x1234),
      "alltoall_algo",
      5);

  allToAllConfig.args.collective.alltoall.sendbuff =
      reinterpret_cast<void*>(0x3333);
  allToAllConfig.args.collective.alltoall.recvbuff =
      reinterpret_cast<void*>(0x4444);
  allToAllConfig.args.collective.alltoall.datatype = commDataType_t::commInt8;
  allToAllConfig.args.collective.alltoall.count = 256;

  uint64_t opCount = 77;

  auto metadata = getCollectiveMetadata(opGroup_, allToAllConfig, opCount);

  EXPECT_EQ(metadata.opName, "AllToAll");
  EXPECT_EQ(metadata.algoName, "alltoall_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(metadata.sendbuff, 0x3333);
  EXPECT_EQ(metadata.recvbuff, 0x4444);
  EXPECT_EQ(metadata.dataType, commDataType_t::commInt8);
  EXPECT_EQ(metadata.count, 256);
}

TEST_F(CollTraceWrapperTest, GetCollectiveMetadata_AllToAllv_CorrectMetadata) {
  // Create AllToAllv kernel config
  KernelConfig allToAllvConfig(
      KernelConfig::KernelType::ALLTOALLV,
      reinterpret_cast<cudaStream_t>(0x1234),
      "alltoallv_algo",
      5);

  allToAllvConfig.args.collective.alltoallv.sendbuff =
      reinterpret_cast<void*>(0x5555);
  allToAllvConfig.args.collective.alltoallv.recvbuff =
      reinterpret_cast<void*>(0x6666);
  allToAllvConfig.args.collective.alltoallv.datatype =
      commDataType_t::commFloat16;

  uint64_t opCount = 88;

  auto metadata = getCollectiveMetadata(opGroup_, allToAllvConfig, opCount);

  EXPECT_EQ(metadata.opName, "AllToAllv");
  EXPECT_EQ(metadata.algoName, "alltoallv_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(metadata.sendbuff, 0x5555);
  EXPECT_EQ(metadata.recvbuff, 0x6666);
  EXPECT_EQ(metadata.dataType, commDataType_t::commFloat16);
  EXPECT_FALSE(metadata.count.has_value()); // AllToAllv uses variable counts
}

TEST_F(CollTraceWrapperTest, GetCollectiveMetadata_Broadcast_CorrectMetadata) {
  // Create Broadcast kernel config
  KernelConfig broadcastConfig(
      KernelConfig::KernelType::BROADCAST,
      reinterpret_cast<cudaStream_t>(0x1234),
      "broadcast_algo",
      5);

  broadcastConfig.args.collective.broadcast.sendbuff =
      reinterpret_cast<void*>(0x7777);
  broadcastConfig.args.collective.broadcast.recvbuff =
      reinterpret_cast<void*>(0x8888);
  broadcastConfig.args.collective.broadcast.datatype =
      commDataType_t::commFloat64;
  broadcastConfig.args.collective.broadcast.count = 128;

  uint64_t opCount = 99;

  auto metadata = getCollectiveMetadata(opGroup_, broadcastConfig, opCount);

  EXPECT_EQ(metadata.opName, "Broadcast");
  EXPECT_EQ(metadata.algoName, "broadcast_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(metadata.sendbuff, 0x7777);
  EXPECT_EQ(metadata.recvbuff, 0x8888);
  EXPECT_EQ(metadata.dataType, commDataType_t::commFloat64);
  EXPECT_EQ(metadata.count, 128);
}

TEST_F(
    CollTraceWrapperTest,
    GetCollectiveMetadata_ReduceScatter_CorrectMetadata) {
  // Create ReduceScatter kernel config
  KernelConfig reduceScatterConfig(
      KernelConfig::KernelType::REDUCESCATTER,
      reinterpret_cast<cudaStream_t>(0x1234),
      "reducescatter_algo",
      5);

  reduceScatterConfig.args.collective.reducescatter.sendbuff =
      reinterpret_cast<void*>(0x9999);
  reduceScatterConfig.args.collective.reducescatter.recvbuff =
      reinterpret_cast<void*>(0xAAAA);
  reduceScatterConfig.args.collective.reducescatter.datatype =
      commDataType_t::commInt64;
  reduceScatterConfig.args.collective.reducescatter.recvcount = 64;

  uint64_t opCount = 111;

  auto metadata = getCollectiveMetadata(opGroup_, reduceScatterConfig, opCount);

  EXPECT_EQ(metadata.opName, "ReduceScatter");
  EXPECT_EQ(metadata.algoName, "reducescatter_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(metadata.sendbuff, 0x9999);
  EXPECT_EQ(metadata.recvbuff, 0xAAAA);
  EXPECT_EQ(metadata.dataType, commDataType_t::commInt64);
  EXPECT_EQ(metadata.count, 64);
}

TEST_F(CollTraceWrapperTest, GetCollectiveMetadata_AllGatherP_CorrectMetadata) {
  // Create AllGatherP kernel config
  KernelConfig allGatherPConfig(
      KernelConfig::KernelType::ALLGATHERP,
      reinterpret_cast<cudaStream_t>(0x1234),
      "allgatherp_algo",
      5);

  uint64_t opCount = 222;

  auto metadata = getCollectiveMetadata(opGroup_, allGatherPConfig, opCount);

  EXPECT_EQ(metadata.opName, "AllGatherP");
  EXPECT_EQ(metadata.algoName, "allgatherp_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  // AllGatherP doesn't have buffer information in the current implementation
}

TEST_F(
    CollTraceWrapperTest,
    GetCollectiveMetadata_AllGatherPInit_CorrectMetadata) {
  // Create AllGatherP_Init kernel config
  KernelConfig allGatherPInitConfig(
      KernelConfig::KernelType::ALLGATHERP_INIT,
      reinterpret_cast<cudaStream_t>(0x1234),
      "allgatherp_init_algo",
      5);

  uint64_t opCount = 333;

  auto metadata =
      getCollectiveMetadata(opGroup_, allGatherPInitConfig, opCount);

  EXPECT_EQ(metadata.opName, "AllGatherP_Init");
  EXPECT_EQ(metadata.algoName, "allgatherp_init_algo");
  EXPECT_EQ(metadata.opCount, opCount);
}

TEST_F(
    CollTraceWrapperTest,
    GetCollectiveMetadata_AllToAllvDynamic_CorrectMetadata) {
  // Create AllToAllv_Dynamic kernel config
  KernelConfig allToAllvDynamicConfig(
      KernelConfig::KernelType::ALLTOALLV_DYNAMIC,
      reinterpret_cast<cudaStream_t>(0x1234),
      "alltoallv_dynamic_algo",
      5);

  uint64_t opCount = 444;

  auto metadata =
      getCollectiveMetadata(opGroup_, allToAllvDynamicConfig, opCount);

  EXPECT_EQ(metadata.opName, "AllToAllv_Dynamic");
  EXPECT_EQ(metadata.algoName, "alltoallv_dynamic_algo");
  EXPECT_EQ(metadata.opCount, opCount);
}

TEST_F(
    CollTraceWrapperTest,
    GetCollectiveMetadata_AllToAllvDynamicSplit_CorrectMetadata) {
  // Create AllToAllv_Dynamic_Split kernel config
  KernelConfig allToAllvDynamicSplitConfig(
      KernelConfig::KernelType::ALLTOALLV_DYNAMIC_SPLIT,
      reinterpret_cast<cudaStream_t>(0x1234),
      "alltoallv_dynamic_split_algo",
      5);

  uint64_t opCount = 555;

  auto metadata =
      getCollectiveMetadata(opGroup_, allToAllvDynamicSplitConfig, opCount);

  EXPECT_EQ(metadata.opName, "AllToAllv_Dynamic_Split");
  EXPECT_EQ(metadata.algoName, "alltoallv_dynamic_split_algo");
  EXPECT_EQ(metadata.opCount, opCount);
}

TEST_F(
    CollTraceWrapperTest,
    GetCollectiveMetadata_AllToAllvDynamicSplitNonContig_CorrectMetadata) {
  // Create AllToAllv_Dynamic_Split_Non_Contig kernel config
  KernelConfig allToAllvDynamicSplitNonContigConfig(
      KernelConfig::KernelType::ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG,
      reinterpret_cast<cudaStream_t>(0x1234),
      "alltoallv_dynamic_split_non_contig_algo",
      5);

  uint64_t opCount = 666;

  auto metadata = getCollectiveMetadata(
      opGroup_, allToAllvDynamicSplitNonContigConfig, opCount);

  EXPECT_EQ(metadata.opName, "AllToAllv_Dynamic_Split_Non_Contig");
  EXPECT_EQ(metadata.algoName, "alltoallv_dynamic_split_non_contig_algo");
  EXPECT_EQ(metadata.opCount, opCount);
}

TEST_F(
    CollTraceWrapperTest,
    GetCollectiveMetadata_AllToAllDedup_CorrectMetadata) {
  // Create AllToAll_Dedup kernel config
  KernelConfig allToAllDedupConfig(
      KernelConfig::KernelType::ALLTOALL_DEDUP,
      reinterpret_cast<cudaStream_t>(0x1234),
      "alltoall_dedup_algo",
      5);

  uint64_t opCount = 777;

  auto metadata = getCollectiveMetadata(opGroup_, allToAllDedupConfig, opCount);

  EXPECT_EQ(metadata.opName, "AllToAll_Dedup");
  EXPECT_EQ(metadata.algoName, "alltoall_dedup_algo");
  EXPECT_EQ(metadata.opCount, opCount);
}

TEST_F(
    CollTraceWrapperTest,
    GetCollectiveMetadata_AllToAllvDedup_CorrectMetadata) {
  // Create AllToAllv_Dedup kernel config
  KernelConfig allToAllvDedupConfig(
      KernelConfig::KernelType::ALLTOALLV_DEDUP,
      reinterpret_cast<cudaStream_t>(0x1234),
      "alltoallv_dedup_algo",
      5);

  uint64_t opCount = 999;

  auto metadata =
      getCollectiveMetadata(opGroup_, allToAllvDedupConfig, opCount);

  EXPECT_EQ(metadata.opName, "AllToAllv_Dedup");
  EXPECT_EQ(metadata.algoName, "alltoallv_dedup_algo");
  EXPECT_EQ(metadata.opCount, opCount);
}

TEST_F(
    CollTraceWrapperTest,
    GetCollectiveMetadata_BroadcastUnpack_CorrectMetadata) {
  // Create Broadcast_Unpack kernel config
  KernelConfig broadcastUnpackConfig(
      KernelConfig::KernelType::BROADCAST_UNPACK,
      reinterpret_cast<cudaStream_t>(0x1234),
      "broadcast_unpack_algo",
      5);

  broadcastUnpackConfig.args.collective.broadcast.sendbuff =
      reinterpret_cast<void*>(0xBBBB);
  broadcastUnpackConfig.args.collective.broadcast.recvbuff =
      reinterpret_cast<void*>(0xCCCC);
  broadcastUnpackConfig.args.collective.broadcast.datatype =
      commDataType_t::commInt32;
  broadcastUnpackConfig.args.collective.broadcast.count = 32;

  uint64_t opCount = 1010;

  auto metadata =
      getCollectiveMetadata(opGroup_, broadcastUnpackConfig, opCount);

  EXPECT_EQ(metadata.opName, "Broadcast_Unpack");
  EXPECT_EQ(metadata.algoName, "broadcast_unpack_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(metadata.sendbuff, 0xBBBB);
  EXPECT_EQ(metadata.recvbuff, 0xCCCC);
  EXPECT_EQ(metadata.dataType, commDataType_t::commInt32);
  EXPECT_EQ(metadata.count, 32);
}

TEST_F(CollTraceWrapperTest, GetCollectiveMetadata_PutNotify_CorrectMetadata) {
  // Create PutNotify kernel config
  KernelConfig putNotifyConfig(
      KernelConfig::KernelType::PUTNOTIFY,
      reinterpret_cast<cudaStream_t>(0x1234),
      "putnotify_algo",
      5);

  uint64_t opCount = 1111;

  auto metadata = getCollectiveMetadata(opGroup_, putNotifyConfig, opCount);

  EXPECT_EQ(metadata.opName, "PutNotify");
  EXPECT_EQ(metadata.algoName, "putnotify_algo");
  EXPECT_EQ(metadata.opCount, opCount);
}

TEST_F(CollTraceWrapperTest, GetCollectiveMetadata_WaitNotify_CorrectMetadata) {
  // Create WaitNotify kernel config
  KernelConfig waitNotifyConfig(
      KernelConfig::KernelType::WAITNOTIFY,
      reinterpret_cast<cudaStream_t>(0x1234),
      "waitnotify_algo",
      5);

  uint64_t opCount = 1212;

  auto metadata = getCollectiveMetadata(opGroup_, waitNotifyConfig, opCount);

  EXPECT_EQ(metadata.opName, "WaitNotify");
  EXPECT_EQ(metadata.algoName, "waitnotify_algo");
  EXPECT_EQ(metadata.opCount, opCount);
}

TEST_F(CollTraceWrapperTest, GetCollectiveMetadata_PutSignal_CorrectMetadata) {
  // Create PutSignal kernel config
  KernelConfig putSignalConfig(
      KernelConfig::KernelType::PUTSIGNAL,
      reinterpret_cast<cudaStream_t>(0x1234),
      "putsignal_algo",
      5);

  uint64_t opCount = 1313;

  auto metadata = getCollectiveMetadata(opGroup_, putSignalConfig, opCount);

  EXPECT_EQ(metadata.opName, "PutSignal");
  EXPECT_EQ(metadata.algoName, "putsignal_algo");
  EXPECT_EQ(metadata.opCount, opCount);
}

TEST_F(CollTraceWrapperTest, GetCollectiveMetadata_WaitSignal_CorrectMetadata) {
  // Create WaitSignal kernel config
  KernelConfig waitSignalConfig(
      KernelConfig::KernelType::WAITSIGNAL,
      reinterpret_cast<cudaStream_t>(0x1234),
      "waitsignal_algo",
      5);

  uint64_t opCount = 1414;

  auto metadata = getCollectiveMetadata(opGroup_, waitSignalConfig, opCount);

  EXPECT_EQ(metadata.opName, "WaitSignal");
  EXPECT_EQ(metadata.algoName, "waitsignal_algo");
  EXPECT_EQ(metadata.opCount, opCount);
}

TEST_F(CollTraceWrapperTest, GetCollectiveMetadata_Signal_CorrectMetadata) {
  // Create Signal kernel config
  KernelConfig signalConfig(
      KernelConfig::KernelType::SIGNAL,
      reinterpret_cast<cudaStream_t>(0x1234),
      "signal_algo",
      5);

  uint64_t opCount = 1515;

  auto metadata = getCollectiveMetadata(opGroup_, signalConfig, opCount);

  EXPECT_EQ(metadata.opName, "Signal");
  EXPECT_EQ(metadata.algoName, "signal_algo");
  EXPECT_EQ(metadata.opCount, opCount);
}

TEST_F(
    CollTraceWrapperTest,
    GetCollectiveMetadata_P2PKernelType_ReturnsUnknown) {
  // Create a P2P kernel config (should be handled by P2P path, but test
  // fallback)
  KernelConfig p2pConfig(
      KernelConfig::KernelType::SEND,
      reinterpret_cast<cudaStream_t>(0x1234),
      "send_algo",
      5);

  uint64_t opCount = 1616;

  auto metadata = getCollectiveMetadata(opGroup_, p2pConfig, opCount);

  EXPECT_EQ(metadata.opName, "Unknown");
  EXPECT_EQ(metadata.algoName, "send_algo");
  EXPECT_EQ(metadata.opCount, opCount);
}

// Tests for getGroupedP2PMetaData
TEST_F(CollTraceWrapperTest, GetGroupedP2PMetaData_SendOnly_CorrectMetadata) {
  // Create kernel config for P2P operations
  KernelConfig p2pConfig(
      KernelConfig::KernelType::SEND,
      reinterpret_cast<cudaStream_t>(0x1234),
      "p2p_algo",
      5);

  // Create SEND operations
  std::vector<std::unique_ptr<struct OpElem>> p2pOpGroup;

  auto sendOp1 = createMockOpElem(OpElem::SEND, comm_.get(), 5);
  sendOp1->send.sendbuff = reinterpret_cast<void*>(0x1000);
  sendOp1->send.count = 100;
  sendOp1->send.datatype = commDataType_t::commFloat32;
  sendOp1->send.peerRank = 1;
  p2pOpGroup.push_back(std::move(sendOp1));

  auto sendOp2 = createMockOpElem(OpElem::SEND, comm_.get(), 5);
  sendOp2->send.sendbuff = reinterpret_cast<void*>(0x1000); // Same buffer
  sendOp2->send.count = 200;
  sendOp2->send.datatype = commDataType_t::commFloat32;
  sendOp2->send.peerRank = 2;
  p2pOpGroup.push_back(std::move(sendOp2));

  int curRank = 0;
  uint64_t opCount = 42;

  auto metadata =
      getGroupedP2PMetaData(curRank, p2pOpGroup, p2pConfig, opCount);

  EXPECT_EQ(metadata.opName, "Send");
  EXPECT_EQ(metadata.algoName, "p2p_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(metadata.dataType, commDataType_t::commFloat32);
  EXPECT_EQ(metadata.count, 300); // 100 + 200

  // Check ranks in grouped P2P (should include curRank and peer ranks)
  std::vector<int> expectedRanks = {0, 1, 2};
  EXPECT_EQ(metadata.ranksInGroupedP2P, expectedRanks);
}

TEST_F(CollTraceWrapperTest, GetGroupedP2PMetaData_RecvOnly_CorrectMetadata) {
  // Create kernel config for P2P operations
  KernelConfig p2pConfig(
      KernelConfig::KernelType::RECV,
      reinterpret_cast<cudaStream_t>(0x1234),
      "p2p_recv_algo",
      5);

  // Create RECV operations
  std::vector<std::unique_ptr<struct OpElem>> p2pOpGroup;

  auto recvOp1 = createMockOpElem(OpElem::RECV, comm_.get(), 5);
  recvOp1->recv.recvbuff = reinterpret_cast<void*>(0x2000);
  recvOp1->recv.count = 150;
  recvOp1->recv.datatype = commDataType_t::commInt32;
  recvOp1->recv.peerRank = 3;
  p2pOpGroup.push_back(std::move(recvOp1));

  auto recvOp2 = createMockOpElem(OpElem::RECV, comm_.get(), 5);
  recvOp2->recv.recvbuff = reinterpret_cast<void*>(0x2000); // Same buffer
  recvOp2->recv.count = 250;
  recvOp2->recv.datatype = commDataType_t::commInt32;
  recvOp2->recv.peerRank = 4;
  p2pOpGroup.push_back(std::move(recvOp2));

  int curRank = 1;
  uint64_t opCount = 55;

  auto metadata =
      getGroupedP2PMetaData(curRank, p2pOpGroup, p2pConfig, opCount);

  EXPECT_EQ(metadata.opName, "Recv");
  EXPECT_EQ(metadata.algoName, "p2p_recv_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(metadata.dataType, commDataType_t::commInt32);
  EXPECT_EQ(metadata.count, 400); // 150 + 250

  // Check ranks in grouped P2P (should include curRank and peer ranks)
  std::vector<int> expectedRanks = {1, 3, 4};
  EXPECT_EQ(metadata.ranksInGroupedP2P, expectedRanks);
}

TEST_F(CollTraceWrapperTest, GetGroupedP2PMetaData_SendRecv_CorrectMetadata) {
  // Create kernel config for P2P operations
  KernelConfig p2pConfig(
      KernelConfig::KernelType::SENDRECV,
      reinterpret_cast<cudaStream_t>(0x1234),
      "sendrecv_algo",
      5);

  // Create mixed SEND and RECV operations
  std::vector<std::unique_ptr<struct OpElem>> p2pOpGroup;

  auto sendOp = createMockOpElem(OpElem::SEND, comm_.get(), 5);
  sendOp->send.sendbuff = reinterpret_cast<void*>(0x1000);
  sendOp->send.count = 100;
  sendOp->send.datatype = commDataType_t::commFloat32;
  sendOp->send.peerRank = 1;
  p2pOpGroup.push_back(std::move(sendOp));

  auto recvOp = createMockOpElem(OpElem::RECV, comm_.get(), 5);
  recvOp->recv.recvbuff = reinterpret_cast<void*>(0x2000);
  recvOp->recv.count = 200;
  recvOp->recv.datatype = commDataType_t::commFloat32;
  recvOp->recv.peerRank = 2;
  p2pOpGroup.push_back(std::move(recvOp));

  int curRank = 0;
  uint64_t opCount = 77;

  auto metadata =
      getGroupedP2PMetaData(curRank, p2pOpGroup, p2pConfig, opCount);

  EXPECT_EQ(metadata.opName, "SendRecv");
  EXPECT_EQ(metadata.algoName, "sendrecv_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(metadata.dataType, commDataType_t::commFloat32);
  EXPECT_EQ(metadata.count, 300); // 100 + 200

  // Check ranks in grouped P2P (should include curRank and peer ranks)
  std::vector<int> expectedRanks = {0, 1, 2};
  EXPECT_EQ(metadata.ranksInGroupedP2P, expectedRanks);
}

TEST_F(
    CollTraceWrapperTest,
    GetGroupedP2PMetaData_MixedDataTypes_CorrectMetadata) {
  // Create kernel config for P2P operations
  KernelConfig p2pConfig(
      KernelConfig::KernelType::SEND,
      reinterpret_cast<cudaStream_t>(0x1234),
      "mixed_types_algo",
      5);

  // Create SEND operations with different data types
  std::vector<std::unique_ptr<struct OpElem>> p2pOpGroup;

  auto sendOp1 = createMockOpElem(OpElem::SEND, comm_.get(), 5);
  sendOp1->send.sendbuff = reinterpret_cast<void*>(0x1000);
  sendOp1->send.count = 100;
  sendOp1->send.datatype = commDataType_t::commFloat32;
  sendOp1->send.peerRank = 1;
  p2pOpGroup.push_back(std::move(sendOp1));

  auto sendOp2 = createMockOpElem(OpElem::SEND, comm_.get(), 5);
  sendOp2->send.sendbuff = reinterpret_cast<void*>(0x1000);
  sendOp2->send.count = 200;
  sendOp2->send.datatype = commDataType_t::commInt32; // Different data type
  sendOp2->send.peerRank = 2;
  p2pOpGroup.push_back(std::move(sendOp2));

  int curRank = 0;
  uint64_t opCount = 88;

  auto metadata =
      getGroupedP2PMetaData(curRank, p2pOpGroup, p2pConfig, opCount);

  EXPECT_EQ(metadata.opName, "Send");
  EXPECT_EQ(metadata.algoName, "mixed_types_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(
      metadata.dataType,
      commDataType_t::commNumTypes); // Mixed types should result in
                                     // commNumTypes
  EXPECT_EQ(metadata.count, 300); // 100 + 200

  // Check ranks in grouped P2P
  std::vector<int> expectedRanks = {0, 1, 2};
  EXPECT_EQ(metadata.ranksInGroupedP2P, expectedRanks);
}

TEST_F(
    CollTraceWrapperTest,
    GetGroupedP2PMetaData_DifferentBuffers_CorrectMetadata) {
  // Create kernel config for P2P operations
  KernelConfig p2pConfig(
      KernelConfig::KernelType::SEND,
      reinterpret_cast<cudaStream_t>(0x1234),
      "diff_buffers_algo",
      5);

  // Create SEND operations with different buffer addresses
  std::vector<std::unique_ptr<struct OpElem>> p2pOpGroup;

  auto sendOp1 = createMockOpElem(OpElem::SEND, comm_.get(), 5);
  sendOp1->send.sendbuff = reinterpret_cast<void*>(0x1000);
  sendOp1->send.count = 100;
  sendOp1->send.datatype = commDataType_t::commFloat32;
  sendOp1->send.peerRank = 1;
  p2pOpGroup.push_back(std::move(sendOp1));

  auto sendOp2 = createMockOpElem(OpElem::SEND, comm_.get(), 5);
  sendOp2->send.sendbuff = reinterpret_cast<void*>(0x2000); // Different buffer
  sendOp2->send.count = 200;
  sendOp2->send.datatype = commDataType_t::commFloat32;
  sendOp2->send.peerRank = 2;
  p2pOpGroup.push_back(std::move(sendOp2));

  int curRank = 0;
  uint64_t opCount = 99;

  auto metadata =
      getGroupedP2PMetaData(curRank, p2pOpGroup, p2pConfig, opCount);

  EXPECT_EQ(metadata.opName, "Send");
  EXPECT_EQ(metadata.algoName, "diff_buffers_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(metadata.dataType, commDataType_t::commFloat32);
  EXPECT_EQ(metadata.count, 300); // 100 + 200

  // Check ranks in grouped P2P
  std::vector<int> expectedRanks = {0, 1, 2};
  EXPECT_EQ(metadata.ranksInGroupedP2P, expectedRanks);
}

TEST_F(
    CollTraceWrapperTest,
    GetGroupedP2PMetaData_UnknownOpType_ReturnsUnknown) {
  // Create kernel config for P2P operations
  KernelConfig p2pConfig(
      KernelConfig::KernelType::SEND,
      reinterpret_cast<cudaStream_t>(0x1234),
      "unknown_op_algo",
      5);

  // Create operation with unknown type
  std::vector<std::unique_ptr<struct OpElem>> p2pOpGroup;

  auto unknownOp = createMockOpElem(
      OpElem::opType::ALLGATHER, comm_.get(), 5); // Non-p2p type
  p2pOpGroup.push_back(std::move(unknownOp));

  int curRank = 0;
  uint64_t opCount = 111;

  auto metadata =
      getGroupedP2PMetaData(curRank, p2pOpGroup, p2pConfig, opCount);

  EXPECT_EQ(metadata.opName, "Unknown");
  EXPECT_EQ(metadata.algoName, "unknown_op_algo");
  EXPECT_EQ(metadata.opCount, opCount);
}

TEST_F(
    CollTraceWrapperTest,
    GetGroupedP2PMetaData_EmptyOpGroup_CorrectMetadata) {
  // Create kernel config for P2P operations
  KernelConfig p2pConfig(
      KernelConfig::KernelType::SEND,
      reinterpret_cast<cudaStream_t>(0x1234),
      "empty_group_algo",
      5);

  // Empty operation group
  std::vector<std::unique_ptr<struct OpElem>> p2pOpGroup;

  int curRank = 0;
  uint64_t opCount = 123;

  auto metadata =
      getGroupedP2PMetaData(curRank, p2pOpGroup, p2pConfig, opCount);

  EXPECT_EQ(metadata.opName, "");
  EXPECT_EQ(metadata.algoName, "empty_group_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(metadata.dataType, commDataType_t::commNumTypes);
  EXPECT_EQ(metadata.count, 0);

  // Should only contain current rank
  std::vector<int> expectedRanks = {0};
  EXPECT_EQ(metadata.ranksInGroupedP2P, expectedRanks);
}

TEST_F(
    CollTraceWrapperTest,
    GetGroupedP2PMetaData_DuplicateRanks_CorrectMetadata) {
  // Create kernel config for P2P operations
  KernelConfig p2pConfig(
      KernelConfig::KernelType::SEND,
      reinterpret_cast<cudaStream_t>(0x1234),
      "duplicate_ranks_algo",
      5);

  // Create SEND operations with duplicate peer ranks
  std::vector<std::unique_ptr<struct OpElem>> p2pOpGroup;

  auto sendOp1 = createMockOpElem(OpElem::SEND, comm_.get(), 5);
  sendOp1->send.sendbuff = reinterpret_cast<void*>(0x1000);
  sendOp1->send.count = 100;
  sendOp1->send.datatype = commDataType_t::commFloat32;
  sendOp1->send.peerRank = 1;
  p2pOpGroup.push_back(std::move(sendOp1));

  auto sendOp2 = createMockOpElem(OpElem::SEND, comm_.get(), 5);
  sendOp2->send.sendbuff = reinterpret_cast<void*>(0x1000);
  sendOp2->send.count = 200;
  sendOp2->send.datatype = commDataType_t::commFloat32;
  sendOp2->send.peerRank = 1; // Same peer rank as sendOp1
  p2pOpGroup.push_back(std::move(sendOp2));

  int curRank = 0;
  uint64_t opCount = 135;

  auto metadata =
      getGroupedP2PMetaData(curRank, p2pOpGroup, p2pConfig, opCount);

  EXPECT_EQ(metadata.opName, "Send");
  EXPECT_EQ(metadata.algoName, "duplicate_ranks_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(metadata.dataType, commDataType_t::commFloat32);
  EXPECT_EQ(metadata.count, 300); // 100 + 200

  // Should not have duplicate ranks
  std::vector<int> expectedRanks = {0, 1};
  EXPECT_EQ(metadata.ranksInGroupedP2P, expectedRanks);
}

TEST_F(
    CollTraceWrapperTest,
    GetGroupedP2PMetaData_RecvDuplicatePeerRank_CorrectMetadata) {
  // Create kernel config for P2P operations
  KernelConfig p2pConfig(
      KernelConfig::KernelType::RECV,
      reinterpret_cast<cudaStream_t>(0x1234),
      "recv_duplicate_algo",
      5);

  // Create RECV operation (note: there's a duplicate insert in the original
  // code)
  std::vector<std::unique_ptr<struct OpElem>> p2pOpGroup;

  auto recvOp = createMockOpElem(OpElem::RECV, comm_.get(), 5);
  recvOp->recv.recvbuff = reinterpret_cast<void*>(0x2000);
  recvOp->recv.count = 150;
  recvOp->recv.datatype = commDataType_t::commInt32;
  recvOp->recv.peerRank = 3;
  p2pOpGroup.push_back(std::move(recvOp));

  int curRank = 1;
  uint64_t opCount = 147;

  auto metadata =
      getGroupedP2PMetaData(curRank, p2pOpGroup, p2pConfig, opCount);

  EXPECT_EQ(metadata.opName, "Recv");
  EXPECT_EQ(metadata.algoName, "recv_duplicate_algo");
  EXPECT_EQ(metadata.opCount, opCount);
  EXPECT_EQ(metadata.dataType, commDataType_t::commInt32);
  EXPECT_EQ(metadata.count, 150);

  // Should not have duplicate ranks (the original code has a bug with duplicate
  // insert)
  std::vector<int> expectedRanks = {1, 3};
  EXPECT_EQ(metadata.ranksInGroupedP2P, expectedRanks);
}
