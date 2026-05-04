// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/SendRecv/SendRecvP2pImpl.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#if not defined(__HIP_PLATFORM_AMD__) and not defined(__HIP_PLATFORM_HCC__)
#include "comms/utils/test_utils/CudaGraphTestUtils.h"
#endif

class SendRecvP2pTest : public ::testing::Test {
 public:
  int cudaDev;
  std::unique_ptr<ctran::TestCtranCommRAII> dummyCommRAII;
  CtranComm* dummyComm{nullptr};
  CtranAlgoDeviceState* dummyDevState_d{nullptr};

  SendRecvP2pTest() = default;

 protected:
  void SetUp() override {
    cudaDev = 0;
    ncclCvarInit();
    CUDACHECK_TEST(cudaMalloc(&dummyDevState_d, sizeof(CtranAlgoDeviceState)));
    dummyCommRAII = ctran::createDummyCtranComm();
    dummyComm = dummyCommRAII->ctranComm.get();
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaFree(dummyDevState_d));
  }

  // Create send/recv ops that trigger the useList path.
  struct TestOps {
    std::vector<std::unique_ptr<OpElem>> storage;
    std::vector<OpElem*> ptrs;
  };

  TestOps makeSendOps(size_t numSends, void* buf, uint64_t opCount) {
    TestOps ops;
    for (size_t i = 0; i < numSends; i++) {
      auto& op = ops.storage.emplace_back(
          std::make_unique<OpElem>(OpElem::opType::SEND, dummyComm, opCount));
      op->send.sendbuff = buf;
      op->send.count = 256;
      op->send.datatype = commInt8;
      op->send.peerRank = 0;
      ops.ptrs.push_back(op.get());
    }
    return ops;
  }

  TestOps makeSendRecvOps(
      size_t numSends,
      void* sendBuf,
      size_t numRecvs,
      void* recvBuf,
      uint64_t opCount) {
    TestOps ops;
    for (size_t i = 0; i < numSends; i++) {
      auto& op = ops.storage.emplace_back(
          std::make_unique<OpElem>(OpElem::opType::SEND, dummyComm, opCount));
      op->send.sendbuff = sendBuf;
      op->send.count = 256;
      op->send.datatype = commInt8;
      op->send.peerRank = 0;
      ops.ptrs.push_back(op.get());
    }
    for (size_t i = 0; i < numRecvs; i++) {
      auto& op = ops.storage.emplace_back(
          std::make_unique<OpElem>(OpElem::opType::RECV, dummyComm, opCount));
      op->recv.recvbuff = recvBuf;
      op->recv.count = 128;
      op->recv.datatype = commInt8;
      op->recv.peerRank = 0;
      ops.ptrs.push_back(op.get());
    }
    return ops;
  }
};

// Verify that when ops exceed kMaxSendRecvOpsPerPoolBuf, the direct
// cudaHostAlloc fallback is used and postKernelCleanup frees pinned memory.
TEST_F(SendRecvP2pTest, SetupP2pKernelConfigDirectAllocCleanup) {
  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  // Exceed pool buffer capacity to force direct cudaHostAlloc path
  constexpr size_t numOps = ctran::sendrecv::kMaxSendRecvOpsPerPoolBuf + 1;
  constexpr uint64_t dummyOpCount = 100;
  void* dummyBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&dummyBuf, 1024));

  auto ops = makeSendOps(numOps, dummyBuf, dummyOpCount);

  // Snapshot GPU free memory before allocation
  size_t freeBefore, total;
  CUDACHECK_TEST(cudaMemGetInfo(&freeBefore, &total));

  KernelConfig config(
      KernelConfig::KernelType::SENDRECV_P2P, stream, "testAlgo", dummyOpCount);
  config.args.devState_d = dummyDevState_d;
  ctran::sendrecv::KernArgs kernArgs{};

  auto res = ctran::sendrecv::setupP2pKernelConfig(
      dummyComm, ops.ptrs, config, kernArgs);
  EXPECT_EQ(res, commSuccess);
  EXPECT_TRUE(kernArgs.useList);
  EXPECT_NE(kernArgs.sendsList, nullptr);
  EXPECT_EQ(kernArgs.numSends, numOps);

  // No device memory should have been allocated
  size_t freeAfterSetup;
  CUDACHECK_TEST(cudaMemGetInfo(&freeAfterSetup, &total));
  EXPECT_EQ(freeAfterSetup, freeBefore)
      << "Device memory was allocated; expected pinned host memory only";

  // Verify postKernelCleanup was set (frees pinned host memory)
  EXPECT_TRUE(config.postKernelCleanup != nullptr);

  // Run cleanup (simulates what GPE thread or caller does)
  config.postKernelCleanup();
  config.postKernelCleanup = nullptr;

  // Verify no device memory leaked
  size_t freeAfter;
  CUDACHECK_TEST(cudaMemGetInfo(&freeAfter, &total));
  EXPECT_EQ(freeAfter, freeBefore) << "Device memory leaked by useList path";

  CUDACHECK_TEST(cudaFree(dummyBuf));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

// Verify the pinned mempool path: setupP2pKernelConfig uses
// cudaHostAllocDefault pinned memory, writes ops into the pinned host buffer,
// and the host pointer is set directly on kernArgs
TEST_F(SendRecvP2pTest, SetupP2pKernelConfigPinnedMempool) {
  constexpr size_t numSends = ctran::sendrecv::kCtranMaxNvlSendRecvOps + 1;
  constexpr size_t numRecvs = ctran::sendrecv::kCtranMaxNvlSendRecvOps + 2;
  constexpr uint64_t dummyOpCount = 100;

  void* dummySendBuf = nullptr;
  void* dummyRecvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&dummySendBuf, 1024));
  CUDACHECK_TEST(cudaMalloc(&dummyRecvBuf, 1024));

  auto ops = makeSendRecvOps(
      numSends, dummySendBuf, numRecvs, dummyRecvBuf, dummyOpCount);

  KernelConfig config(
      KernelConfig::KernelType::SENDRECV_P2P,
      nullptr,
      "testAlgo",
      dummyOpCount);
  config.args.devState_d = dummyDevState_d;
  ctran::sendrecv::KernArgs kernArgs{};

  auto res = ctran::sendrecv::setupP2pKernelConfig(
      dummyComm, ops.ptrs, config, kernArgs);
  EXPECT_EQ(res, commSuccess);
  EXPECT_TRUE(kernArgs.useList);

  // Verify sendsList and recvsList are set
  ASSERT_NE(kernArgs.sendsList, nullptr);
  ASSERT_NE(kernArgs.recvsList, nullptr);
  EXPECT_EQ(kernArgs.numSends, numSends);
  EXPECT_EQ(kernArgs.numRecvs, numRecvs);

  // Verify ops were written to pinned host memory
  for (size_t i = 0; i < numSends; i++) {
    EXPECT_EQ(kernArgs.sendsList[i].buff, dummySendBuf);
    EXPECT_EQ(kernArgs.sendsList[i].nbytes, 256);
  }
  for (size_t i = 0; i < numRecvs; i++) {
    EXPECT_EQ(kernArgs.recvsList[i].buff, dummyRecvBuf);
    EXPECT_EQ(kernArgs.recvsList[i].nbytes, 128);
  }

  // Run cleanup to release pool buffer
  ASSERT_NE(config.postKernelCleanup, nullptr);
  config.postKernelCleanup();
  config.postKernelCleanup = nullptr;
  CUDACHECK_TEST(cudaFree(dummySendBuf));
  CUDACHECK_TEST(cudaFree(dummyRecvBuf));
}

#if not defined(__HIP_PLATFORM_AMD__) and not defined(__HIP_PLATFORM_HCC__)
// Verify that during CUDA graph capture the pinned mempool is skipped and
// direct pinned host memory is used instead. Pool buffers can be reclaimed and
// reused, so they must not be referenced by a graph that replays later.
TEST_F(SendRecvP2pTest, GraphCaptureSkipsPinnedMempool) {
  constexpr size_t numOps = ctran::sendrecv::kCtranMaxNvlSendRecvOps + 1;
  constexpr uint64_t dummyOpCount = 100;
  void* dummyBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&dummyBuf, 1024));

  auto ops = makeSendOps(numOps, dummyBuf, dummyOpCount);

  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));
  CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));

  // Snapshot GPU free memory after stream creation + capture begin,
  // so the assertion isolates setupP2pKernelConfig's allocations only.
  size_t freeBefore, total;
  CUDACHECK_TEST(cudaMemGetInfo(&freeBefore, &total));

  KernelConfig config(
      KernelConfig::KernelType::SENDRECV_P2P, stream, "testAlgo", dummyOpCount);
  config.args.devState_d = dummyDevState_d;
  ctran::sendrecv::KernArgs kernArgs{};

  auto res = ctran::sendrecv::setupP2pKernelConfig(
      dummyComm, ops.ptrs, config, kernArgs);
  EXPECT_EQ(res, commSuccess);
  EXPECT_TRUE(kernArgs.useList);

  // Pool must be skipped during capture — sendsList should point to
  // directly-allocated pinned host memory.
  EXPECT_NE(kernArgs.sendsList, nullptr);

  // No device memory should have been allocated (pinned host memory only)
  size_t freeAfterSetup;
  CUDACHECK_TEST(cudaMemGetInfo(&freeAfterSetup, &total));
  EXPECT_EQ(freeAfterSetup, freeBefore)
      << "Device memory was allocated; expected pinned host memory only";

  // Cleanup callback must be set (frees pinned host memory)
  EXPECT_NE(config.postKernelCleanup, nullptr);

  cudaGraph_t graph;
  CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
  if (graph) {
    cudaGraphDestroy(graph);
  }

  // Run cleanup (simulates GPE thread freeing pinned host memory)
  config.postKernelCleanup();
  config.postKernelCleanup = nullptr;

  CUDACHECK_TEST(cudaFree(dummyBuf));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

// Verify that during graph capture, the useList path writes ops directly to
// pinned host memory
TEST_F(SendRecvP2pTest, GraphCaptureUseListDataIsValid) {
  constexpr size_t numSends = ctran::sendrecv::kCtranMaxNvlSendRecvOps + 1;
  constexpr size_t numRecvs = ctran::sendrecv::kCtranMaxNvlSendRecvOps + 1;
  constexpr uint64_t dummyOpCount = 100;

  void* dummySendBuf = nullptr;
  void* dummyRecvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&dummySendBuf, 1024));
  CUDACHECK_TEST(cudaMalloc(&dummyRecvBuf, 1024));

  auto ops = makeSendRecvOps(
      numSends, dummySendBuf, numRecvs, dummyRecvBuf, dummyOpCount);

  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));
  CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));

  KernelConfig config(
      KernelConfig::KernelType::SENDRECV_P2P, stream, "testAlgo", dummyOpCount);
  config.args.devState_d = dummyDevState_d;
  ctran::sendrecv::KernArgs kernArgs{};

  auto res = ctran::sendrecv::setupP2pKernelConfig(
      dummyComm, ops.ptrs, config, kernArgs);
  EXPECT_EQ(res, commSuccess);
  EXPECT_TRUE(kernArgs.useList);

  // End capture.
  cudaGraph_t graph;
  CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));

  // No memcpy nodes should appear in the graph — pinned host memory is
  // written directly (no H->D copy needed).
  {
    auto topo = getGraphTopology(graph);
    auto memcpyNodes = topo.nodesOfType(cudaGraphNodeTypeMemcpy);
    ASSERT_EQ(memcpyNodes.size(), 0)
        << "useList H->D memcpy was recorded into the graph; "
           "pinned host memory should be written directly";
  }

  // Pinned host memory: data is written directly, readable from host side.
  // No cudaMemcpy needed to verify — just read the pointers directly.
  for (size_t i = 0; i < numSends; i++) {
    EXPECT_EQ(kernArgs.sendsList[i].buff, dummySendBuf)
        << "sendsList[" << i << "].buff not written during capture";
    EXPECT_EQ(kernArgs.sendsList[i].nbytes, 256)
        << "sendsList[" << i << "].nbytes not written during capture";
  }

  for (size_t i = 0; i < numRecvs; i++) {
    EXPECT_EQ(kernArgs.recvsList[i].buff, dummyRecvBuf)
        << "recvsList[" << i << "].buff not written during capture";
    EXPECT_EQ(kernArgs.recvsList[i].nbytes, 128)
        << "recvsList[" << i << "].nbytes not written during capture";
  }

  if (graph) {
    cudaGraphDestroy(graph);
  }

  config.postKernelCleanup();
  config.postKernelCleanup = nullptr;

  CUDACHECK_TEST(cudaFree(dummySendBuf));
  CUDACHECK_TEST(cudaFree(dummyRecvBuf));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}
#endif
