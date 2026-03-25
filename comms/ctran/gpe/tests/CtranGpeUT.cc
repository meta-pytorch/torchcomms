// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/utils/Checks.h"
// FIXME [REBASE]: update the path once moved to fbcode/comms
#include "comms/ctran/gpe/tests/CtranGpeUTKernels.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#if not defined(__HIP_PLATFORM_AMD__) and not defined(__HIP_PLATFORM_HCC__)
#include <cupti.h>
#include "comms/utils/test_utils/CudaGraphTestUtils.h"
#endif
class CtranGpeTest : public ::testing::Test {
 public:
  CtranGpe* gpe;
  int cudaDev;
  std::unique_ptr<ctran::TestCtranCommRAII> dummyCommRAII;
  CtranComm* dummyComm{nullptr};
  CtranAlgoDeviceState* dummyDevState_d{nullptr};

  CtranGpeTest() = default;

 protected:
  void SetUp() override {
    cudaDev = 0;
    gpe = nullptr;

    // Ensure logger is initialized
    ncclCvarInit();

    CUDACHECK_TEST(cudaMalloc(&dummyDevState_d, sizeof(CtranAlgoDeviceState)));
    dummyCommRAII = ctran::createDummyCtranComm();
    dummyComm = dummyCommRAII->ctranComm.get();
  }
  void TearDown() override {
    if (gpe != nullptr) {
      delete gpe;
    }
    CUDACHECK_TEST(cudaFree(dummyDevState_d));
  }
};

class CtranGpeKernelTest : public ::testing::Test {
 public:
  volatile int* testFlag;
  CtranAlgoDeviceState* dummyDevState_d{nullptr};
  std::unique_ptr<ctran::TestCtranCommRAII> dummyCommRAII;
  CtranComm* dummyComm{nullptr};
  int cudaDev;
  CtranGpeKernelTest() = default;

 protected:
  void SetUp() override {
    cudaDev = 0;
    FB_CUDACHECKIGNORE(cudaSetDevice(cudaDev));

    // Ensure logger is initialized
    ncclCvarInit();

    dummyCommRAII = ctran::createDummyCtranComm();
    dummyComm = dummyCommRAII->ctranComm.get();

    FB_CUDACHECKIGNORE(
        cudaHostAlloc((void**)&testFlag, sizeof(int), cudaHostAllocDefault));
    *testFlag = KERNEL_UNSET;

    CUDACHECK_TEST(cudaMalloc(&dummyDevState_d, sizeof(CtranAlgoDeviceState)));
  }
  void TearDown() override {
    FB_CUDACHECKIGNORE(cudaFreeHost((void*)testFlag));
    CUDACHECK_TEST(cudaFree(dummyDevState_d));
  }
};

static const std::string kExpectedOutput{"CtranGpeTestAlgoFunc Called"};
static commResult_t CtranGpeTestAlgoFunc(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  std::cout << kExpectedOutput;
  return commSuccess;
}

TEST_F(CtranGpeTest, gpeThread) {
  gpe = new CtranGpe(cudaDev, dummyComm);
  EXPECT_THAT(gpe, testing::NotNull());
}

TEST_F(CtranGpeTest, SubmitOpBadCudaKernel) {
  commResult_t res = commSuccess;
  CtranGpe* gpe = new CtranGpe(cudaDev, dummyComm);

  uint64_t dummyOpCount = 100;
  std::vector<std::unique_ptr<struct OpElem>> ops;
  struct OpElem* op;
  op = new struct OpElem(OpElem::opType::SEND, dummyComm, dummyOpCount);
  op->send.sendbuff = nullptr;
  op->send.count = 0;
  op->send.datatype = commInt8;
  op->send.peerRank = 0;
  ops.push_back(std::unique_ptr<struct OpElem>(op));

  auto kernelConfig = KernelConfig(
      KernelConfig::KernelType::SEND, nullptr, "dummyAlgo", dummyOpCount);
  kernelConfig.args.devState_d = dummyDevState_d;

  /* NOTE: invalid CUDA kernel should return error code */
  res =
      gpe->submit(std::move(ops), &CtranGpeTestAlgoFunc, kernelConfig, nullptr);
  EXPECT_NE(res, commSuccess);

  // Expect no flag is consumed
  EXPECT_EQ(gpe->numInUseKernelFlags(), 0);
}

TEST_F(CtranGpeTest, SubmitHostAllowNullReq) {
  commResult_t res = commSuccess;
  CtranGpe* gpe = new CtranGpe(cudaDev, dummyComm);

  uint64_t dummyOpCount = 100;
  std::vector<std::unique_ptr<struct OpElem>> ops;
  struct OpElem* op;
  op = new struct OpElem(OpElem::opType::SEND, dummyComm, dummyOpCount);
  op->send.sendbuff = nullptr;
  op->send.count = 0;
  op->send.datatype = commInt8;
  op->send.peerRank = 0;
  ops.push_back(std::unique_ptr<struct OpElem>(op));

  auto kernelConfig = KernelConfig(
      KernelConfig::KernelType::SEND, nullptr, "dummyAlgo", dummyOpCount);
  kernelConfig.args.devState_d = dummyDevState_d;

  res = gpe->submitHost(
      std::move(ops),
      &CtranGpeTestAlgoFunc,
      kernelConfig,
      /* exReq */ nullptr);
  EXPECT_EQ(res, commSuccess);

  // Expect no flag is consumed
  EXPECT_EQ(gpe->numInUseKernelFlags(), 0);
}

TEST_F(CtranGpeTest, SubmitOpBadDevState) {
  commResult_t res = commSuccess;
  CtranGpe* gpe = new CtranGpe(cudaDev, dummyComm);

  uint64_t dummyOpCount = 100;
  std::vector<std::unique_ptr<struct OpElem>> ops;
  struct OpElem* op;
  op = new struct OpElem(OpElem::opType::SEND, dummyComm, dummyOpCount);
  op->send.sendbuff = nullptr;
  op->send.count = 0;
  op->send.datatype = commInt8;
  op->send.peerRank = 0;
  ops.push_back(std::unique_ptr<struct OpElem>(op));

  auto kernelConfig = KernelConfig(
      KernelConfig::KernelType::SEND, nullptr, "dummyAlgo", dummyOpCount);
  kernelConfig.args.devState_d = dummyDevState_d;

  // Invalid devState_d should be checked and return commInternalError
  kernelConfig.args.devState_d = nullptr;
  res = gpe->submit(
      std::move(ops),
      &CtranGpeTestAlgoFunc,
      kernelConfig,
      reinterpret_cast<void*>(CtranGpeTestKernel));
  EXPECT_EQ(res, commInternalError);

  // Expect no flag is consumed
  EXPECT_EQ(gpe->numInUseKernelFlags(), 0);
}

constexpr int count = 1024;
constexpr int kKernelpdatedVal = 100;

TEST_F(CtranGpeTest, SubmitOpKernel) {
  commResult_t res = commSuccess;
  CtranGpe* gpe = new CtranGpe(cudaDev, dummyComm);
  cudaStream_t stream;
  cudaEvent_t event;
  CUDACHECK_TEST(cudaStreamCreate(&stream));
  CUDACHECK_TEST(cudaEventCreate(&event));

  int* a = nullptr;
  int* expectedValPtr = nullptr;
  CUDACHECK_TEST(cudaMalloc(&a, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(a, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&expectedValPtr, sizeof(int)));
  *expectedValPtr = kKernelpdatedVal;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint64_t dummyOpCount = 100;
  std::vector<std::unique_ptr<struct OpElem>> ops;
  struct OpElem* op;
  op = new struct OpElem(OpElem::opType::RECV, dummyComm, dummyOpCount);
  op->recv.recvbuff = nullptr;
  op->recv.count = 0;
  op->recv.datatype = commInt8;
  op->recv.peerRank = 0;
  ops.push_back(std::unique_ptr<struct OpElem>(op));

  // Use ALLGATHER kernel config to pass test variables
  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", dummyOpCount);
  ctranKernelSetAllGatherArgs(
      a, expectedValPtr, commInt8, count, dummyDevState_d, &config.args);

  testing::internal::CaptureStdout();

  res = gpe->submit(
      std::move(ops),
      &CtranGpeTestAlgoFunc,
      config,
      reinterpret_cast<void*>(CtranGpeTestKernel));
  CUDACHECK_TEST(cudaEventRecord(event, stream));

  EXPECT_EQ(res, commSuccess);

  int numInuse = 0;
  while (cudaEventQuery(event) == cudaErrorNotReady) {
    // record the number of flags consumed during kernel execution
    numInuse = gpe->numInUseKernelFlags();
    if (numInuse > 0) {
      // Expect 1 flag is used during the kernel execution
      EXPECT_EQ(numInuse, 1);
    }
  }
  CUDACHECK_TEST(cudaStreamDestroy(stream));

  // check GPE hostFn has been called
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput));

  // Expect flag is returned after kernel finish
  EXPECT_EQ(gpe->numInUseKernelFlags(), 0);

  delete gpe;
  gpe = nullptr;

  // check kernel has been called
  std::vector<int> a_host(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      a_host.data(), a, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(a_host, testing::Each(kKernelpdatedVal));
  CUDACHECK_TEST(cudaEventDestroy(event));
}

TEST_F(CtranGpeTest, SubmitOnlyKernel) {
  commResult_t res = commSuccess;
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));
  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  int* a = nullptr;
  int* expectedValPtr = nullptr;
  CUDACHECK_TEST(cudaMalloc(&a, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(a, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&expectedValPtr, sizeof(int)));
  *expectedValPtr = kKernelpdatedVal;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<std::unique_ptr<struct OpElem>> emptyOps;

  // Use ALLGATHER kernel config to pass test variables
  constexpr uint64_t dummyOpCount = 0;
  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", dummyOpCount);
  ctranKernelSetAllGatherArgs(
      a, expectedValPtr, commInt8, count, dummyDevState_d, &config.args);

  // empty OpGroup would launch only kernel
  res = gpe->submit(
      std::move(emptyOps),
      nullptr,
      config,
      reinterpret_cast<void*>(CtranGpeTestKernel));
  EXPECT_EQ(res, commSuccess);
  // Kernel only submit won't consume flag
  EXPECT_EQ(gpe->numInUseKernelFlags(), 0);

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // check kernel has been called
  std::vector<int> a_host(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      a_host.data(), a, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(a_host, testing::Each(kKernelpdatedVal));

  CUDACHECK_TEST(cudaFree(a));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

#if not defined(__HIP_PLATFORM_AMD__) and not defined(__HIP_PLATFORM_HCC__)
// Verify multi-stream kernel submissions produce a valid captured graph.
// Kernels launch on user streams (streamA, streamB). Cross-stream ordering
// is enforced via execModeSyncEvent_ record/wait nodes. The graph should
// contain 2 kernel nodes with kernelA -> kernelB (event ordering edge).
//
TEST_F(CtranGpeTest, SubmitOpKernelMultiStreamGraphCapture) {
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));

  cudaStream_t primaryStream;
  CUDACHECK_TEST(cudaStreamCreate(&primaryStream));

  cudaStream_t streamA, streamB;
  CUDACHECK_TEST(cudaStreamCreate(&streamA));
  CUDACHECK_TEST(cudaStreamCreate(&streamB));

  cudaEvent_t forkEvent, joinEventA, joinEventB;
  CUDACHECK_TEST(cudaEventCreate(&forkEvent));
  CUDACHECK_TEST(cudaEventCreate(&joinEventA));
  CUDACHECK_TEST(cudaEventCreate(&joinEventB));

  constexpr int kVal = 42;
  int* bufA = nullptr;
  int* bufB = nullptr;
  int* valPtr = nullptr;
  CUDACHECK_TEST(cudaMalloc(&bufA, sizeof(int) * count));
  CUDACHECK_TEST(cudaMalloc(&bufB, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(bufA, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(bufB, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&valPtr, sizeof(int)));
  *valPtr = kVal;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(
      cudaStreamBeginCapture(primaryStream, cudaStreamCaptureModeRelaxed));

  CUDACHECK_TEST(cudaEventRecord(forkEvent, primaryStream));
  CUDACHECK_TEST(cudaStreamWaitEvent(streamA, forkEvent, 0));
  CUDACHECK_TEST(cudaStreamWaitEvent(streamB, forkEvent, 0));

  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER,
        streamA,
        "dummyAlgo",
        dummyOpCount);
    ctranKernelSetAllGatherArgs(
        bufA, valPtr, commInt8, count, dummyDevState_d, &config.args);
    auto res = gpe->submit(
        std::move(emptyOps),
        nullptr,
        config,
        reinterpret_cast<void*>(CtranGpeTestKernel));
    ASSERT_EQ(res, commSuccess);
  }

  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER,
        streamB,
        "dummyAlgo",
        dummyOpCount);
    ctranKernelSetAllGatherArgs(
        bufB, valPtr, commInt8, count, dummyDevState_d, &config.args);
    auto res = gpe->submit(
        std::move(emptyOps),
        nullptr,
        config,
        reinterpret_cast<void*>(CtranGpeTestKernel));
    ASSERT_EQ(res, commSuccess);
  }

  CUDACHECK_TEST(cudaEventRecord(joinEventA, streamA));
  CUDACHECK_TEST(cudaEventRecord(joinEventB, streamB));
  CUDACHECK_TEST(cudaStreamWaitEvent(primaryStream, joinEventA, 0));
  CUDACHECK_TEST(cudaStreamWaitEvent(primaryStream, joinEventB, 0));

  cudaGraph_t graph;
  CUDACHECK_TEST(cudaStreamEndCapture(primaryStream, &graph));
  ASSERT_NE(graph, nullptr);

  // Verify graph topology: 2 GPE kernels on user streams (streamA, streamB).
  // Cross-stream ordering via execModeSyncEvent_ record/wait nodes ensures
  // kernelA -> kernelB.
  // Each kernel should have a WAIT(E)/RECORD(E) pair from the guard.
  {
    auto topo = getGraphTopology(graph);
    auto kernelNodes = topo.nodesOfType(cudaGraphNodeTypeKernel);
    ASSERT_EQ(kernelNodes.size(), 2);

    auto kernelA = kernelNodes[0];
    auto kernelB = kernelNodes[1];
    EXPECT_TRUE(topo.hasPath(kernelA, kernelB))
        << "kernelB must transitively depend on kernelA";
    EXPECT_FALSE(topo.hasPath(kernelB, kernelA))
        << "kernelA must NOT depend on kernelB (would be a cycle)";

    // Each kernel gets a WAIT(E)/RECORD(E) pair from the guard.
    // User fork/join use regular cudaEventRecord/cudaStreamWaitEvent
    // which create graph edges, not event nodes — so they don't
    // contribute to the WAIT/RECORD node counts.
    auto waitNodes = topo.nodesOfType(cudaGraphNodeTypeWaitEvent);
    auto recordNodes = topo.nodesOfType(cudaGraphNodeTypeEventRecord);
    EXPECT_EQ(waitNodes.size(), kernelNodes.size())
        << "Each kernel should have one guard WAIT(E) node";
    EXPECT_EQ(recordNodes.size(), kernelNodes.size())
        << "Each kernel should have one guard RECORD(E) node";
  }

  cudaGraphExec_t graphExec;
  CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  CUDACHECK_TEST(cudaGraphLaunch(graphExec, primaryStream));
  CUDACHECK_TEST(cudaStreamSynchronize(primaryStream));

  std::vector<int> hostA(count, 0);
  std::vector<int> hostB(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      hostA.data(), bufA, sizeof(int) * count, cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      hostB.data(), bufB, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(hostA, testing::Each(kVal));
  EXPECT_THAT(hostB, testing::Each(kVal));

  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  CUDACHECK_TEST(cudaEventDestroy(forkEvent));
  CUDACHECK_TEST(cudaEventDestroy(joinEventA));
  CUDACHECK_TEST(cudaEventDestroy(joinEventB));
  CUDACHECK_TEST(cudaFree(bufA));
  CUDACHECK_TEST(cudaFree(bufB));
  CUDACHECK_TEST(cudaFreeHost(valPtr));
  CUDACHECK_TEST(cudaStreamDestroy(primaryStream));
  CUDACHECK_TEST(cudaStreamDestroy(streamA));
  CUDACHECK_TEST(cudaStreamDestroy(streamB));
}

// RAII helper to count cudaStreamWaitEvent calls via CUPTI callback API.
class CuptiWaitEventCounter {
 public:
  CuptiWaitEventCounter() {
    auto cb = [](void* userdata,
                 CUpti_CallbackDomain domain,
                 CUpti_CallbackId cbid,
                 const CUpti_CallbackData* cbdata) {
      if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
          cbid == CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020 &&
          cbdata->callbackSite == CUPTI_API_ENTER) {
        reinterpret_cast<std::atomic<int>*>(userdata)->fetch_add(1);
      }
    };
    CHECK_EQ(
        cuptiSubscribe(
            &subscriber_, reinterpret_cast<CUpti_CallbackFunc>(+cb), &count_),
        CUPTI_SUCCESS);
    CHECK_EQ(
        cuptiEnableCallback(
            1,
            subscriber_,
            CUPTI_CB_DOMAIN_RUNTIME_API,
            CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020),
        CUPTI_SUCCESS);
  }

  ~CuptiWaitEventCounter() {
    cuptiUnsubscribe(subscriber_);
  }

  int count() const {
    return count_.load();
  }

 private:
  std::atomic<int> count_{0};
  CUpti_SubscriberHandle subscriber_;
};

// Verify eager same-stream fast path: consecutive submits on the same
// stream skip cudaStreamWaitEvent entirely (0 acquire calls per submit).
TEST_F(CtranGpeTest, EagerSameStreamOrdering) {
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));

  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  constexpr int kVal1 = 11;
  constexpr int kVal2 = 22;
  constexpr int kVal3 = 33;
  int* buf = nullptr;
  int* valPtr1 = nullptr;
  int* valPtr2 = nullptr;
  int* valPtr3 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(buf, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&valPtr1, sizeof(int)));
  CUDACHECK_TEST(cudaMallocHost(&valPtr2, sizeof(int)));
  CUDACHECK_TEST(cudaMallocHost(&valPtr3, sizeof(int)));
  *valPtr1 = kVal1;
  *valPtr2 = kVal2;
  *valPtr3 = kVal3;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  CuptiWaitEventCounter waitCounter;

  // Three consecutive submits on the same stream
  for (auto* valPtr : {valPtr1, valPtr2, valPtr3}) {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", dummyOpCount);
    ctranKernelSetAllGatherArgs(
        buf, valPtr, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }

  // Same-stream fast path: no cudaStreamWaitEvent calls
  EXPECT_EQ(waitCounter.count(), 0)
      << "Same-stream eager submits should skip cudaStreamWaitEvent";

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Last kernel wrote kVal3
  std::vector<int> host(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      host.data(), buf, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(host, testing::Each(kVal3));

  CUDACHECK_TEST(cudaFree(buf));
  CUDACHECK_TEST(cudaFreeHost(valPtr1));
  CUDACHECK_TEST(cudaFreeHost(valPtr2));
  CUDACHECK_TEST(cudaFreeHost(valPtr3));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

// Verify eager cross-stream path: submits on alternating streams must
// call cudaStreamWaitEvent to enforce ordering.
TEST_F(CtranGpeTest, EagerCrossStreamOrdering) {
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));

  cudaStream_t streamA, streamB;
  CUDACHECK_TEST(cudaStreamCreate(&streamA));
  CUDACHECK_TEST(cudaStreamCreate(&streamB));

  constexpr int kVal1 = 11;
  constexpr int kVal2 = 22;
  int* buf = nullptr;
  int* valPtr1 = nullptr;
  int* valPtr2 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(buf, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&valPtr1, sizeof(int)));
  CUDACHECK_TEST(cudaMallocHost(&valPtr2, sizeof(int)));
  *valPtr1 = kVal1;
  *valPtr2 = kVal2;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  CuptiWaitEventCounter waitCounter;

  // Submit on streamA then streamB (cross-stream)
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER, streamA, "dummyAlgo", 0);
    ctranKernelSetAllGatherArgs(
        buf, valPtr1, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER, streamB, "dummyAlgo", 1);
    ctranKernelSetAllGatherArgs(
        buf, valPtr2, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }

  // Cross-stream: cudaStreamWaitEvent must have been called
  EXPECT_GT(waitCounter.count(), 0)
      << "Cross-stream eager submits must call cudaStreamWaitEvent";

  CUDACHECK_TEST(cudaStreamSynchronize(streamA));
  CUDACHECK_TEST(cudaStreamSynchronize(streamB));

  // Last kernel wrote kVal2
  std::vector<int> host(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      host.data(), buf, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(host, testing::Each(kVal2));

  CUDACHECK_TEST(cudaFree(buf));
  CUDACHECK_TEST(cudaFreeHost(valPtr1));
  CUDACHECK_TEST(cudaFreeHost(valPtr2));
  CUDACHECK_TEST(cudaStreamDestroy(streamA));
  CUDACHECK_TEST(cudaStreamDestroy(streamB));
}

// Verify that each kernel captured into a graph gets exactly one
// WAIT(E)/RECORD(E) pair from the guard. This test captures a single kernel
// on a single stream with no user fork/join, so the only event nodes in the
// graph are from the guard.
TEST_F(CtranGpeTest, GraphCaptureGuardWaitRecordPerKernel) {
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));

  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  int* buf = nullptr;
  int* valPtr = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(buf, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&valPtr, sizeof(int)));
  *valPtr = 42;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", 0);
    ctranKernelSetAllGatherArgs(
        buf, valPtr, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }

  cudaGraph_t graph;
  CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
  ASSERT_NE(graph, nullptr);

  {
    auto topo = getGraphTopology(graph);
    auto kernelNodes = topo.nodesOfType(cudaGraphNodeTypeKernel);
    ASSERT_EQ(kernelNodes.size(), 1);

    // No user fork/join — only guard event nodes.
    // Each kernel should have exactly 1 WAIT(E) + 1 RECORD(E).
    auto waitNodes = topo.nodesOfType(cudaGraphNodeTypeWaitEvent);
    auto recordNodes = topo.nodesOfType(cudaGraphNodeTypeEventRecord);
    EXPECT_EQ(waitNodes.size(), 1)
        << "Guard should add 1 WAIT(E) node per kernel";
    EXPECT_EQ(recordNodes.size(), 1)
        << "Guard should add 1 RECORD(E) node per kernel";
  }

  cudaGraphExec_t graphExec;
  CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream));
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  std::vector<int> host(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      host.data(), buf, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(host, testing::Each(42));

  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  CUDACHECK_TEST(cudaFree(buf));
  CUDACHECK_TEST(cudaFreeHost(valPtr));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

// Verify that consecutive same-stream submits followed by a cross-stream
// submit produce correct ordering. Same-stream kernels (kernelA1, kernelA2)
// get implicit stream ordering on streamA. Cross-stream ordering to kernelB
// on streamB is enforced via execModeSyncEvent_ record/wait nodes.
//
// Host call order:
//
//   gpe->submit(streamA, kernelA1)   // first submit on streamA
//   gpe->submit(streamA, kernelA2)   // second submit on streamA (same stream)
//   gpe->submit(streamB, kernelB)    // cross-stream submit
//
// kernelA1 -> kernelA2 (implicit, same stream)
// kernelA2 -> kernelB  (event ordering)
//
TEST_F(CtranGpeTest, GraphCaptureSameStreamThenCrossStream) {
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));

  cudaStream_t primaryStream;
  CUDACHECK_TEST(cudaStreamCreate(&primaryStream));
  cudaStream_t streamA, streamB;
  CUDACHECK_TEST(cudaStreamCreate(&streamA));
  CUDACHECK_TEST(cudaStreamCreate(&streamB));

  cudaEvent_t forkEvent, joinEventA, joinEventB;
  CUDACHECK_TEST(cudaEventCreate(&forkEvent));
  CUDACHECK_TEST(cudaEventCreate(&joinEventA));
  CUDACHECK_TEST(cudaEventCreate(&joinEventB));

  constexpr int kVal1 = 11;
  constexpr int kVal2 = 22;
  constexpr int kValB = 33;
  int* bufA = nullptr;
  int* bufB = nullptr;
  int* valPtr1 = nullptr;
  int* valPtr2 = nullptr;
  int* valPtrB = nullptr;
  CUDACHECK_TEST(cudaMalloc(&bufA, sizeof(int) * count));
  CUDACHECK_TEST(cudaMalloc(&bufB, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(bufA, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(bufB, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&valPtr1, sizeof(int)));
  CUDACHECK_TEST(cudaMallocHost(&valPtr2, sizeof(int)));
  CUDACHECK_TEST(cudaMallocHost(&valPtrB, sizeof(int)));
  *valPtr1 = kVal1;
  *valPtr2 = kVal2;
  *valPtrB = kValB;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(
      cudaStreamBeginCapture(primaryStream, cudaStreamCaptureModeRelaxed));

  CUDACHECK_TEST(cudaEventRecord(forkEvent, primaryStream));
  CUDACHECK_TEST(cudaStreamWaitEvent(streamA, forkEvent, 0));
  CUDACHECK_TEST(cudaStreamWaitEvent(streamB, forkEvent, 0));

  // First submit on streamA — writes kVal1 to bufA
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER,
        streamA,
        "dummyAlgo",
        dummyOpCount);
    ctranKernelSetAllGatherArgs(
        bufA, valPtr1, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }

  // Second submit on streamA (same stream) — overwrites bufA with kVal2
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER,
        streamA,
        "dummyAlgo",
        dummyOpCount);
    ctranKernelSetAllGatherArgs(
        bufA, valPtr2, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }

  // Cross-stream submit on streamB — writes kValB to bufB
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER,
        streamB,
        "dummyAlgo",
        dummyOpCount);
    ctranKernelSetAllGatherArgs(
        bufB, valPtrB, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }

  CUDACHECK_TEST(cudaEventRecord(joinEventA, streamA));
  CUDACHECK_TEST(cudaEventRecord(joinEventB, streamB));
  CUDACHECK_TEST(cudaStreamWaitEvent(primaryStream, joinEventA, 0));
  CUDACHECK_TEST(cudaStreamWaitEvent(primaryStream, joinEventB, 0));

  cudaGraph_t graph;
  CUDACHECK_TEST(cudaStreamEndCapture(primaryStream, &graph));
  ASSERT_NE(graph, nullptr);

  // Verify graph topology: 3 GPE kernels on user streams.
  // kernelA1 -> kernelA2 (implicit, same stream)
  // kernelA2 -> kernelB  (event ordering, cross-stream)
  // Each kernel should have a WAIT(E)/RECORD(E) pair from the guard.
  {
    auto topo = getGraphTopology(graph);
    auto kernelNodes = topo.nodesOfType(cudaGraphNodeTypeKernel);
    ASSERT_EQ(kernelNodes.size(), 3);

    auto kernelA1 = kernelNodes[0];
    auto kernelA2 = kernelNodes[1];
    auto kernelB = kernelNodes[2];

    // kernelA1 -> kernelA2 (implicit, same user stream)
    EXPECT_TRUE(topo.hasPath(kernelA1, kernelA2))
        << "kernelA2 must depend on kernelA1";

    // kernelA2 -> kernelB (event ordering, cross-stream)
    EXPECT_TRUE(topo.hasPath(kernelA2, kernelB))
        << "kernelB must depend on kernelA2";

    // Transitively: kernelA1 -> kernelA2 -> kernelB
    EXPECT_TRUE(topo.hasPath(kernelA1, kernelB))
        << "kernelB must transitively depend on kernelA1";

    // Each kernel gets a WAIT(E)/RECORD(E) pair from the guard.
    // User fork/join use regular cudaEventRecord/cudaStreamWaitEvent
    // which create graph edges, not event nodes.
    auto waitNodes = topo.nodesOfType(cudaGraphNodeTypeWaitEvent);
    auto recordNodes = topo.nodesOfType(cudaGraphNodeTypeEventRecord);
    EXPECT_EQ(waitNodes.size(), kernelNodes.size())
        << "Each kernel should have one guard WAIT(E) node";
    EXPECT_EQ(recordNodes.size(), kernelNodes.size())
        << "Each kernel should have one guard RECORD(E) node";
  }

  cudaGraphExec_t graphExec;
  CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  CUDACHECK_TEST(cudaGraphLaunch(graphExec, primaryStream));
  CUDACHECK_TEST(cudaStreamSynchronize(primaryStream));

  // bufA should have kVal2 (second kernel overwrites first)
  std::vector<int> hostA(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      hostA.data(), bufA, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(hostA, testing::Each(kVal2));

  // bufB should have kValB
  std::vector<int> hostB(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      hostB.data(), bufB, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(hostB, testing::Each(kValB));

  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  CUDACHECK_TEST(cudaEventDestroy(forkEvent));
  CUDACHECK_TEST(cudaEventDestroy(joinEventA));
  CUDACHECK_TEST(cudaEventDestroy(joinEventB));
  CUDACHECK_TEST(cudaFree(bufA));
  CUDACHECK_TEST(cudaFree(bufB));
  CUDACHECK_TEST(cudaFreeHost(valPtr1));
  CUDACHECK_TEST(cudaFreeHost(valPtr2));
  CUDACHECK_TEST(cudaFreeHost(valPtrB));
  CUDACHECK_TEST(cudaStreamDestroy(primaryStream));
  CUDACHECK_TEST(cudaStreamDestroy(streamA));
  CUDACHECK_TEST(cudaStreamDestroy(streamB));
}

// Verify that non-GPE work enqueued on a user stream between two submit()
// calls does not create false dependencies. Kernels launch on their
// respective user streams. The memset on streamA is ordered after kernelA
// (implicit stream ordering) but does not create a false dependency on
// kernelB (which is on streamB, ordered via event after kernelA).
//
// Host call order:
//
//   gpe->submit(streamA, kernelA)
//   cudaMemsetAsync(nonGpeBuf, ..., streamA)   <-- non-GPE work on user stream
//   gpe->submit(streamB, kernelB)
//
// kernelA -> kernelB (event ordering, cross-stream).
// memset on streamA, after kernelA (implicit), independent of kernelB.
//
TEST_F(CtranGpeTest, GraphCaptureNonGpeWorkBetweenSubmits) {
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));

  cudaStream_t primaryStream;
  CUDACHECK_TEST(cudaStreamCreate(&primaryStream));
  cudaStream_t streamA, streamB;
  CUDACHECK_TEST(cudaStreamCreate(&streamA));
  CUDACHECK_TEST(cudaStreamCreate(&streamB));

  cudaEvent_t forkEvent, joinEventA, joinEventB;
  CUDACHECK_TEST(cudaEventCreate(&forkEvent));
  CUDACHECK_TEST(cudaEventCreate(&joinEventA));
  CUDACHECK_TEST(cudaEventCreate(&joinEventB));

  constexpr int kVal = 55;
  int* bufA = nullptr;
  int* bufB = nullptr;
  int* nonGpeBuf = nullptr;
  int* valPtr = nullptr;
  CUDACHECK_TEST(cudaMalloc(&bufA, sizeof(int) * count));
  CUDACHECK_TEST(cudaMalloc(&bufB, sizeof(int) * count));
  CUDACHECK_TEST(cudaMalloc(&nonGpeBuf, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(bufA, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(bufB, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(nonGpeBuf, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&valPtr, sizeof(int)));
  *valPtr = kVal;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(
      cudaStreamBeginCapture(primaryStream, cudaStreamCaptureModeRelaxed));

  CUDACHECK_TEST(cudaEventRecord(forkEvent, primaryStream));
  CUDACHECK_TEST(cudaStreamWaitEvent(streamA, forkEvent, 0));
  CUDACHECK_TEST(cudaStreamWaitEvent(streamB, forkEvent, 0));

  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER,
        streamA,
        "dummyAlgo",
        dummyOpCount);
    ctranKernelSetAllGatherArgs(
        bufA, valPtr, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }

  CUDACHECK_TEST(
      cudaMemsetAsync(nonGpeBuf, 0xFF, sizeof(int) * count, streamA));

  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER,
        streamB,
        "dummyAlgo",
        dummyOpCount);
    ctranKernelSetAllGatherArgs(
        bufB, valPtr, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }

  CUDACHECK_TEST(cudaEventRecord(joinEventA, streamA));
  CUDACHECK_TEST(cudaEventRecord(joinEventB, streamB));
  CUDACHECK_TEST(cudaStreamWaitEvent(primaryStream, joinEventA, 0));
  CUDACHECK_TEST(cudaStreamWaitEvent(primaryStream, joinEventB, 0));

  cudaGraph_t graph;
  CUDACHECK_TEST(cudaStreamEndCapture(primaryStream, &graph));
  ASSERT_NE(graph, nullptr);

  // Verify graph topology: 2 GPE kernels on user streams + memset on
  // streamA. Cross-stream ordering via event ensures kernelA -> kernelB.
  // The memset is on streamA after kernelA but independent of kernelB.
  {
    auto topo = getGraphTopology(graph);
    auto kernelNodes = topo.nodesOfType(cudaGraphNodeTypeKernel);
    ASSERT_EQ(kernelNodes.size(), 2);

    auto kernelA = kernelNodes[0];
    auto kernelB = kernelNodes[1];

    // kernelB must depend on kernelA (event ordering, cross-stream)
    EXPECT_TRUE(topo.hasPath(kernelA, kernelB))
        << "kernelB must transitively depend on kernelA";
  }

  cudaGraphExec_t graphExec;
  CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  CUDACHECK_TEST(cudaGraphLaunch(graphExec, primaryStream));
  CUDACHECK_TEST(cudaStreamSynchronize(primaryStream));

  std::vector<int> hostA(count, 0);
  std::vector<int> hostB(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      hostA.data(), bufA, sizeof(int) * count, cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      hostB.data(), bufB, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(hostA, testing::Each(kVal));
  EXPECT_THAT(hostB, testing::Each(kVal));

  std::vector<int> hostNonGpe(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      hostNonGpe.data(),
      nonGpeBuf,
      sizeof(int) * count,
      cudaMemcpyDeviceToHost));

  EXPECT_THAT(hostNonGpe, testing::Each(static_cast<int>(0xFFFFFFFF)));

  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  CUDACHECK_TEST(cudaEventDestroy(forkEvent));
  CUDACHECK_TEST(cudaEventDestroy(joinEventA));
  CUDACHECK_TEST(cudaEventDestroy(joinEventB));
  CUDACHECK_TEST(cudaFree(bufA));
  CUDACHECK_TEST(cudaFree(bufB));
  CUDACHECK_TEST(cudaFree(nonGpeBuf));
  CUDACHECK_TEST(cudaFreeHost(valPtr));
  CUDACHECK_TEST(cudaStreamDestroy(primaryStream));
  CUDACHECK_TEST(cudaStreamDestroy(streamA));
  CUDACHECK_TEST(cudaStreamDestroy(streamB));
}

// Verify ordering between graph replay and subsequent eager submissions.
// The captured kernel launches on the user stream and its doRelease adds
// an execModeSyncEvent_ record node to the graph. At replay time, this
// records execModeSyncEvent_. A subsequent non-capture doAcquire waits on
// execModeSyncEvent_ before launching on the user stream.
//
// Host call order:
//   1. Eager kernel A (writes kVal1 to buf) on streamA
//   2. Capture kernel B (writes kVal2 to buf) on streamA
//   3. Replay graph on streamA
//   4. Eager kernel C (writes kVal3 to buf) on streamB
//   5. Sync and verify buf == kVal3 (C ran after graph replay)
//
TEST_F(CtranGpeTest, GraphReplayThenEagerOrdering) {
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));

  cudaStream_t streamA, streamB;
  CUDACHECK_TEST(cudaStreamCreate(&streamA));
  CUDACHECK_TEST(cudaStreamCreate(&streamB));

  constexpr int kVal1 = 11;
  constexpr int kVal2 = 22;
  constexpr int kVal3 = 33;
  int* buf = nullptr;
  int* valPtr1 = nullptr;
  int* valPtr2 = nullptr;
  int* valPtr3 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(buf, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&valPtr1, sizeof(int)));
  CUDACHECK_TEST(cudaMallocHost(&valPtr2, sizeof(int)));
  CUDACHECK_TEST(cudaMallocHost(&valPtr3, sizeof(int)));
  *valPtr1 = kVal1;
  *valPtr2 = kVal2;
  *valPtr3 = kVal3;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Step 1: Eager kernel A on streamA — writes kVal1 to buf
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER,
        streamA,
        "dummyAlgo",
        dummyOpCount);
    ctranKernelSetAllGatherArgs(
        buf, valPtr1, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }
  CUDACHECK_TEST(cudaStreamSynchronize(streamA));

  // Step 2: Capture kernel B on streamA — writes kVal2 to buf
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  CUDACHECK_TEST(cudaStreamBeginCapture(streamA, cudaStreamCaptureModeRelaxed));
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER,
        streamA,
        "dummyAlgo",
        dummyOpCount);
    ctranKernelSetAllGatherArgs(
        buf, valPtr2, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }
  CUDACHECK_TEST(cudaStreamEndCapture(streamA, &graph));
  ASSERT_NE(graph, nullptr);
  CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Step 3: Replay graph on streamA
  CUDACHECK_TEST(cudaGraphLaunch(graphExec, streamA));

  // Step 4: Eager kernel C on streamB — writes kVal3 to buf.
  // The unified event ensures C waits on the graph replay of B.
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER,
        streamB,
        "dummyAlgo",
        dummyOpCount);
    ctranKernelSetAllGatherArgs(
        buf, valPtr3, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }

  // Step 5: Sync and verify
  CUDACHECK_TEST(cudaStreamSynchronize(streamB));

  std::vector<int> host(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      host.data(), buf, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(host, testing::Each(kVal3));

  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  CUDACHECK_TEST(cudaFree(buf));
  CUDACHECK_TEST(cudaFreeHost(valPtr1));
  CUDACHECK_TEST(cudaFreeHost(valPtr2));
  CUDACHECK_TEST(cudaFreeHost(valPtr3));
  CUDACHECK_TEST(cudaStreamDestroy(streamA));
  CUDACHECK_TEST(cudaStreamDestroy(streamB));
}

// Verify ordering between eager submissions and subsequent graph replay.
// The first capture sets everCaptured_. The eager kernel's doRelease records
// execModeSyncEvent_ on the user stream. When the graph is replayed, its
// WAIT node (with cudaEventWaitExternal) waits on execModeSyncEvent_'s
// live state.
//
// Host call order:
//   1. Capture kernel A (writes kVal1 to buf) on streamA
//   2. Eager kernel B (writes kVal2 to buf) on streamA
//   3. Replay graph on streamA (should wait on B via cudaEventWaitExternal)
//   4. Sync and verify buf == kVal1 (graph replay ran after B)
//
TEST_F(CtranGpeTest, EagerThenGraphReplayOrdering) {
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));

  cudaStream_t streamA;
  CUDACHECK_TEST(cudaStreamCreate(&streamA));

  constexpr int kVal1 = 11;
  constexpr int kVal2 = 22;
  int* buf = nullptr;
  int* valPtr1 = nullptr;
  int* valPtr2 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(buf, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&valPtr1, sizeof(int)));
  CUDACHECK_TEST(cudaMallocHost(&valPtr2, sizeof(int)));
  *valPtr1 = kVal1;
  *valPtr2 = kVal2;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Step 1: Capture kernel A on streamA — writes kVal1 to buf
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  CUDACHECK_TEST(cudaStreamBeginCapture(streamA, cudaStreamCaptureModeRelaxed));
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER,
        streamA,
        "dummyAlgo",
        dummyOpCount);
    ctranKernelSetAllGatherArgs(
        buf, valPtr1, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }
  CUDACHECK_TEST(cudaStreamEndCapture(streamA, &graph));
  ASSERT_NE(graph, nullptr);
  CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Step 2: Eager kernel B on streamA — writes kVal2 to buf
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER,
        streamA,
        "dummyAlgo",
        dummyOpCount);
    ctranKernelSetAllGatherArgs(
        buf, valPtr2, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }

  // Step 3: Replay graph on streamA. The cudaEventWaitExternal flag in the
  // graph's WAIT node ensures it waits on B's event recording at replay time.
  CUDACHECK_TEST(cudaGraphLaunch(graphExec, streamA));
  CUDACHECK_TEST(cudaStreamSynchronize(streamA));

  // Step 4: Verify buf == kVal1 (graph replay of A ran last, after B)
  std::vector<int> host(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      host.data(), buf, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(host, testing::Each(kVal1));

  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  CUDACHECK_TEST(cudaFree(buf));
  CUDACHECK_TEST(cudaFreeHost(valPtr1));
  CUDACHECK_TEST(cudaFreeHost(valPtr2));
  CUDACHECK_TEST(cudaStreamDestroy(streamA));
}

// Verify ordering: Replay -> Eager -> Replay.
// The eager submit must wait on the first replay, and the second replay
// must wait on the eager submit. Each transition exercises the cross-mode
// bridge in alternating directions.
//
// Host call order:
//   1. Capture kernel A (writes kVal1) into a graph
//   2. Replay graph (buf = kVal1)
//   3. Eager kernel B (writes kVal2, should overwrite kVal1)
//   4. Replay graph again (buf = kVal1, should overwrite kVal2)
//   5. Verify buf == kVal1
//
TEST_F(CtranGpeTest, ReplayEagerReplayOrdering) {
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));

  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  constexpr int kVal1 = 11;
  constexpr int kVal2 = 22;
  int* buf = nullptr;
  int* valPtr1 = nullptr;
  int* valPtr2 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(buf, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&valPtr1, sizeof(int)));
  CUDACHECK_TEST(cudaMallocHost(&valPtr2, sizeof(int)));
  *valPtr1 = kVal1;
  *valPtr2 = kVal2;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Step 1: Capture kernel A — writes kVal1
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", dummyOpCount);
    ctranKernelSetAllGatherArgs(
        buf, valPtr1, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }
  CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
  ASSERT_NE(graph, nullptr);
  CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Step 2: First replay — buf = kVal1
  CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream));

  // Step 3: Eager kernel B — writes kVal2 (must wait on first replay)
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", dummyOpCount);
    ctranKernelSetAllGatherArgs(
        buf, valPtr2, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }

  // Step 4: Second replay — buf = kVal1 (must wait on eager B)
  CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream));
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Step 5: Verify buf == kVal1 (second replay ran last)
  std::vector<int> host(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      host.data(), buf, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(host, testing::Each(kVal1));

  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  CUDACHECK_TEST(cudaFree(buf));
  CUDACHECK_TEST(cudaFreeHost(valPtr1));
  CUDACHECK_TEST(cudaFreeHost(valPtr2));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

// Verify ordering: Eager -> Replay -> Eager.
// The replay must wait on the first eager submit, and the second eager
// submit must wait on the replay. Each transition exercises the cross-mode
// bridge in alternating directions.
//
// Host call order:
//   1. Capture kernel A (writes kVal1) into a graph
//   2. Eager kernel B (writes kVal2)
//   3. Replay graph (buf = kVal1, should overwrite kVal2)
//   4. Eager kernel C (writes kVal3, should overwrite kVal1)
//   5. Verify buf == kVal3
//
TEST_F(CtranGpeTest, EagerReplayEagerOrdering) {
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));

  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  constexpr int kVal1 = 11;
  constexpr int kVal2 = 22;
  constexpr int kVal3 = 33;
  int* buf = nullptr;
  int* valPtr1 = nullptr;
  int* valPtr2 = nullptr;
  int* valPtr3 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(buf, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&valPtr1, sizeof(int)));
  CUDACHECK_TEST(cudaMallocHost(&valPtr2, sizeof(int)));
  CUDACHECK_TEST(cudaMallocHost(&valPtr3, sizeof(int)));
  *valPtr1 = kVal1;
  *valPtr2 = kVal2;
  *valPtr3 = kVal3;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Step 1: Capture kernel A — writes kVal1
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", dummyOpCount);
    ctranKernelSetAllGatherArgs(
        buf, valPtr1, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }
  CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
  ASSERT_NE(graph, nullptr);
  CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Step 2: Eager kernel B — writes kVal2
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", dummyOpCount);
    ctranKernelSetAllGatherArgs(
        buf, valPtr2, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }

  // Step 3: Replay graph — buf = kVal1 (must wait on eager B)
  CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream));

  // Step 4: Eager kernel C — writes kVal3 (must wait on replay)
  {
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    constexpr uint64_t dummyOpCount = 0;
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", dummyOpCount);
    ctranKernelSetAllGatherArgs(
        buf, valPtr3, commInt8, count, dummyDevState_d, &config.args);
    ASSERT_EQ(
        gpe->submit(
            std::move(emptyOps),
            nullptr,
            config,
            reinterpret_cast<void*>(CtranGpeTestKernel)),
        commSuccess);
  }

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Step 5: Verify buf == kVal3 (second eager ran last)
  std::vector<int> host(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      host.data(), buf, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(host, testing::Each(kVal3));

  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  CUDACHECK_TEST(cudaFree(buf));
  CUDACHECK_TEST(cudaFreeHost(valPtr1));
  CUDACHECK_TEST(cudaFreeHost(valPtr2));
  CUDACHECK_TEST(cudaFreeHost(valPtr3));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

// Verify that graph capture with a non-empty opGroup matches eager-mode
// ordering: the host node (GPE cmdEnqueue) executes before the kernel.
//
// In eager mode, submit() calls cmdEnqueue() then cudaLaunchKernel() — the
// cmd is in the GPE queue before the kernel can start. During capture,
// addHostNode() uses cudaLaunchHostFunc on the captured stream, which
// places the host node before the kernel in stream order, matching this.
//
// On replay:
//   1. Host node fires cmdCb — enqueues cmd to GPE thread (fast: lock+push)
//   2. Kernel starts, sets KERNEL_STARTED
//   3. GPE thread dequeues cmd, sees KERNEL_STARTED, runs algo func
//   4. GPE thread signals KERNEL_TERMINATE
//   5. Kernel completes
//
TEST_F(CtranGpeTest, GraphCaptureWithHostNode) {
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));
  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  constexpr int kVal = 77;
  int* buf = nullptr;
  int* valPtr = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(buf, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&valPtr, sizeof(int)));
  *valPtr = kVal;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));

  {
    uint64_t dummyOpCount = 100;
    std::vector<std::unique_ptr<struct OpElem>> ops;
    auto& op = ops.emplace_back(
        std::make_unique<OpElem>(
            OpElem::opType::RECV, dummyComm, dummyOpCount));
    op->recv.recvbuff = nullptr;
    op->recv.count = 0;
    op->recv.datatype = commInt8;
    op->recv.peerRank = 0;

    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", dummyOpCount);
    ctranKernelSetAllGatherArgs(
        buf, valPtr, commInt8, count, dummyDevState_d, &config.args);

    auto res = gpe->submit(
        std::move(ops),
        &CtranGpeTestAlgoFunc,
        config,
        reinterpret_cast<void*>(CtranGpeTestKernel));
    ASSERT_EQ(res, commSuccess);
  }

  cudaGraph_t graph;
  CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
  ASSERT_NE(graph, nullptr);

  // Matches eager mode: host node (cmdEnqueue) before kernel node.
  // cudaLaunchHostFunc on the captured stream produces this ordering.
  {
    auto topo = getGraphTopology(graph);
    auto hostNodes = topo.nodesOfType(cudaGraphNodeTypeHost);
    auto kernelNodes = topo.nodesOfType(cudaGraphNodeTypeKernel);

    ASSERT_EQ(hostNodes.size(), 1) << "Expected 1 host node from addHostNode";
    ASSERT_EQ(kernelNodes.size(), 1) << "Expected 1 kernel node from submit";

    auto hostNode = hostNodes[0];
    auto kernelNode = kernelNodes[0];

    // Host node must precede kernel node, matching eager-mode ordering
    // where cmdEnqueue completes before cudaLaunchKernel is called.
    EXPECT_TRUE(topo.hasPath(hostNode, kernelNode))
        << "kernel must depend on host node (eager-mode ordering)";
  }

  // Instantiate and replay to verify functional correctness:
  // the kernel writes expected values AND the GPE algo func runs via the
  // host node callback.
  cudaGraphExec_t graphExec;
  CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  testing::internal::CaptureStdout();

  CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream));
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Verify kernel wrote the expected values
  std::vector<int> hostBuf(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      hostBuf.data(), buf, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(hostBuf, testing::Each(kVal));

  // Verify GPE algo func was called via the host node callback
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput));

  // For persistent (graph) cmds, the flag stays in-use between replays to
  // prevent the pool from reclaiming it. It is released when the graph
  // (and thus the cmd) is destroyed.
  EXPECT_EQ(gpe->numInUseKernelFlags(), 1);

  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));

  // cudaUserObjectNoDestructorSync: cmdDestroy fires asynchronously after
  // graph destruction. Wait for it to release the flag back to the pool.
  while (gpe->numInUseKernelFlags() > 0) {
    std::this_thread::yield();
  }
  EXPECT_EQ(gpe->numInUseKernelFlags(), 0);
  CUDACHECK_TEST(cudaFree(buf));
  CUDACHECK_TEST(cudaFreeHost(valPtr));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}
#endif

// Verify that destroying a captured graph triggers cmdDestroy, which resets
// KernelElem statuses and frees the persistent cmd. After graph destruction,
// all pool resources held by the graph should be reclaimable.
//
TEST_F(CtranGpeTest, GraphCaptureDestroyFreesResources) {
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));
  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  constexpr int kVal = 99;
  int* buf = nullptr;
  int* valPtr = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(buf, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaMallocHost(&valPtr, sizeof(int)));
  *valPtr = kVal;
  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));

  {
    uint64_t dummyOpCount = 100;
    std::vector<std::unique_ptr<struct OpElem>> ops;
    auto& op = ops.emplace_back(
        std::make_unique<OpElem>(
            OpElem::opType::RECV, dummyComm, dummyOpCount));
    op->recv.recvbuff = nullptr;
    op->recv.count = 0;
    op->recv.datatype = commInt8;
    op->recv.peerRank = 0;

    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", dummyOpCount);
    ctranKernelSetAllGatherArgs(
        buf, valPtr, commInt8, count, dummyDevState_d, &config.args);

    auto res = gpe->submit(
        std::move(ops),
        &CtranGpeTestAlgoFunc,
        config,
        reinterpret_cast<void*>(CtranGpeTestKernel));
    ASSERT_EQ(res, commSuccess);
  }

  cudaGraph_t graph;
  CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
  ASSERT_NE(graph, nullptr);

  cudaGraphExec_t graphExec;
  CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream));
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  std::vector<int> hostBuf(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      hostBuf.data(), buf, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(hostBuf, testing::Each(kVal));

  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));

  EXPECT_EQ(gpe->numInUseKernelFlags(), 0);

  CUDACHECK_TEST(cudaFree(buf));
  CUDACHECK_TEST(cudaFreeHost(valPtr));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}
TEST_F(CtranGpeTest, SubmitCustomKernArgs) {
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));
  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  std::vector<std::unique_ptr<struct OpElem>> emptyOps;

  const int numElems = 1024;
  const int scaleFactor = 5;
  CtranKernelCustomArgs customArgs = {
      .scaleFactor = scaleFactor, .data = nullptr, .numElems = numElems};
  CUDACHECK_TEST(cudaHostAlloc(
      &customArgs.data, sizeof(int) * numElems, cudaHostAllocDefault));

  for (int i = 0; i < numElems; i++) {
    customArgs.data[i] = i;
  }

  // Use ALLGATHER kernel config to pass test variables
  constexpr uint64_t dummyOpCount = 0;
  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHER,
      stream,
      "dummyAlgoWithCustomArgs",
      &customArgs,
      dummyOpCount);
  config.numBlocks = 2;
  config.numThreads = 256;
  config.args.devState_d = dummyDevState_d;

  // empty OpGroup would launch only kernel
  ASSERT_EQ(
      gpe->submit(
          std::move(emptyOps),
          nullptr,
          config,
          reinterpret_cast<void*>(CtranGpeTestCustomArgsKernel)),
      commSuccess);

  // Kernel only submit won't consume flag
  EXPECT_EQ(gpe->numInUseKernelFlags(), 0);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // check kernel has been called
  for (int i = 0; i < numElems; i++) {
    ASSERT_EQ(customArgs.data[i], i * scaleFactor)
        << fmt::format(" with data[{}] scaleFactor {}", i, scaleFactor)
        << std::endl;
  }

  CUDACHECK_TEST(cudaFreeHost(customArgs.data));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

TEST_F(CtranGpeKernelTest, launchTerminateStallKernel) {
  dim3 grid = {1, 1, 1};
  dim3 blocks = {1, 1, 1};
  void* args[] = {&testFlag};
  ASSERT_EQ(
      cudaFuncSetAttribute(
          reinterpret_cast<void*>(CtranGpeTestTerminateKernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(CtranAlgoDeviceState)),
      cudaSuccess);
  auto res = cudaLaunchKernel(
      reinterpret_cast<void*>(CtranGpeTestTerminateKernel),
      grid,
      blocks,
      args,
      sizeof(CtranAlgoDeviceState),
      0);

  EXPECT_EQ(res, cudaSuccess);

  while (*testFlag != KERNEL_STARTED) {
    EXPECT_THAT(*testFlag, testing::Not(KERNEL_TERMINATE));
  }

  *testFlag = KERNEL_TERMINATE;
  res = cudaStreamSynchronize(0);

  EXPECT_EQ(res, cudaSuccess);
}

TEST_F(CtranGpeTest, SubmitKernelWithStartAndExit) {
  commResult_t res = commSuccess;
  auto gpe = std::make_unique<CtranGpe>(cudaDev, dummyComm);
  cudaStream_t stream;
  cudaEvent_t event;
  CUDACHECK_TEST(cudaStreamCreate(&stream));
  CUDACHECK_TEST(cudaEventCreate(&event));

  constexpr int nIter = 100;
  for (auto i = 0; i < nIter; i++) {
    uint64_t dummyOpCount = 100;
    std::vector<std::unique_ptr<struct OpElem>> ops;
    auto& op = ops.emplace_back(
        std::make_unique<OpElem>(
            OpElem::opType::RECV, dummyComm, dummyOpCount));
    op->recv.recvbuff = nullptr;
    op->recv.count = 0;
    op->recv.datatype = commInt8;
    op->recv.peerRank = 0;

    // Use ALLGATHER kernel config to pass test variables
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", dummyOpCount);
    ctranKernelSetAllGatherArgs(
        nullptr, nullptr, commInt8, count, dummyDevState_d, &config.args);

    res = gpe->submit(
        std::move(ops),
        &CtranGpeTestAlgoFunc,
        config,
        reinterpret_cast<void*>(CtranGpeTestStartAndExitKernel));
    EXPECT_EQ(res, commSuccess);
  }

  // Expect all flags used by the submitted ops can be returned
  // NOTE: we have no good way to drain the GPE thread activities in
  // startAndExit mode. Thus, we simply busy wait till all flags have been
  // returned. If leak happens, the test will timeout.
  while (gpe->numInUseKernelFlags() > 0) {
  }
}

TEST_F(CtranGpeKernelTest, SubmitKernelWithKElems) {
  // Ensure NCCL_CTRAN_NUM_KERNEL_ELEMS has been set
  ncclCvarInit();
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));
  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  // Allocate p2pElems
  KernelElem* elemList = nullptr;
  constexpr int ngroups = 5;
  COMMCHECK_TEST(gpe->allocKernelElems(numKElems, ngroups, &elemList));

  // Check allocated number of p2pElems is as expected
  int nAllocated = 0;
  KernelElem* elem = elemList;
  while (elem) {
    EXPECT_EQ(elem->isFree(), false);
    elem = elem->next;
    nAllocated++;
  }
  EXPECT_EQ(nAllocated, numKElems);

  // Use ALLGATHER kernel config to pass test variables and launch with ngroups
  // gridSize to consume the elems
  std::vector<std::unique_ptr<struct OpElem>> emptyOps;
  constexpr uint64_t dummyOpCount = 0;
  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", dummyOpCount);
  ctranKernelSetAllGatherArgs(
      elemList, nullptr, commInt8, 0, dummyDevState_d, &config.args);
  config.numBlocks = ngroups;

  // Empty OpGroup would launch only kernel
  COMMCHECK_TEST(gpe->submit(
      std::move(emptyOps),
      nullptr,
      config,
      reinterpret_cast<void*>(CtranGpeTestKElemsKernel)));
  // Empty opGroup won't consume flag
  EXPECT_EQ(gpe->numInUseKernelFlags(), 0);

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Check each element has been consumed by kernel
  elem = elemList;
  while (elem) {
    EXPECT_EQ(elem->isFree(), true);
    elem = elem->next;
  }

  // Skip check for reclaim which is an internal operation and triggered in GPE
  // destructor. Coverd by separate UT

  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

TEST_F(CtranGpeTest, kernelConfigToString) {
  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));
  constexpr uint64_t dummyOpCount = 0;
  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHER, stream, "dummyAlgo", dummyOpCount);

  auto str = config.toString();

  std::stringstream streamSs;
  streamSs << "stream=" << std::hex << stream;
  auto streamStr = streamSs.str();

  EXPECT_THAT(str, testing::HasSubstr("ALLGATHER"));
  EXPECT_THAT(str, testing::HasSubstr(streamStr));

  // Cannot test potential unknown type because compiler already catches the
  // type mismatch

  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

TEST_F(CtranGpeKernelTest, InsufficientKElem) {
  // Ensure NCCL_CTRAN_NUM_KERNEL_ELEMS has been set
  ncclCvarInit();
  constexpr int totalNumKElems = 102;
  constexpr int numValidAllocs = totalNumKElems / numKElems;

  // Overwrite NCCL_CTRAN_NUM_KERNEL_ELEMS value
  EnvRAII env(NCCL_CTRAN_NUM_KERNEL_ELEMS, totalNumKElems);

  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev, dummyComm));
  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  std::vector<KernelElem*> elemLists;

  for (int i = 0; i < numValidAllocs + 1; i++) {
    KernelElem* elemList = nullptr;
    constexpr int ngroups = 5;
    auto res = gpe->allocKernelElems(numKElems, ngroups, &elemList);

    // Expect we use up elements and the last allocation should fail
    if (i == numValidAllocs) {
      ASSERT_EQ(res, commInternalError);
    } else {
      ASSERT_EQ(res, commSuccess);
      elemLists.push_back(elemList);
    }
  }
  ASSERT_EQ(elemLists.size(), numValidAllocs);

  // Check we see inuse elements as allocated
  ASSERT_EQ(gpe->numInUseKernelElems(), numKElems * numValidAllocs);

  // Free all allocated elements
  for (auto kList : elemLists) {
    auto elem = kList;
    while (elem) {
      EXPECT_EQ(elem->isFree(), false);
      elem->unuse();
      elem->free();
      elem = elem->next;
    }
  }

  // Check no more inuse elements
  ASSERT_EQ(gpe->numInUseKernelElems(), 0);
}

static commResult_t CtranGpeAsyncExceptionTestAlgoFunc(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  // return error and expect the main gpeThreadFn to convert it to asyncErr
  return commSystemError;
}

TEST_F(CtranGpeTest, ThrowAsyncException) {
  auto gpe = std::make_unique<CtranGpe>(cudaDev, dummyComm);

  uint64_t dummyOpCount = 100;
  std::vector<std::unique_ptr<struct OpElem>> ops;
  struct OpElem* op;
  op = new struct OpElem(OpElem::opType::SEND, dummyComm, dummyOpCount);
  op->send.sendbuff = nullptr;
  op->send.count = 0;
  op->send.datatype = commInt8;
  op->send.peerRank = 0;
  ops.push_back(std::unique_ptr<struct OpElem>(op));

  auto kernelConfig = KernelConfig(
      KernelConfig::KernelType::SEND, nullptr, "dummyAlgo", dummyOpCount);
  kernelConfig.args.devState_d = dummyDevState_d;

  // Submit only GPE function, expect asyncErr
  ASSERT_EQ(
      gpe->submitHost(
          std::move(ops),
          &CtranGpeAsyncExceptionTestAlgoFunc,
          kernelConfig,
          nullptr),
      commSuccess);

  // Expect no flag is consumed
  EXPECT_EQ(gpe->numInUseKernelFlags(), 0);

  // Wait till asyncErr is properly set.
  // If the asyncErr is not set, the test will hang and fail.
  while (dummyComm->getAsyncResult() == commSuccess)
    ;

  // Expect asyncErr is set with proper info
  EXPECT_EQ(dummyComm->getAsyncResult(), commSystemError);
  const auto e = dummyComm->getAsyncException();
  EXPECT_THAT(e.what(), testing::HasSubstr("commSystemError"));
  EXPECT_EQ(e.result(), commSystemError);

  const auto statex = dummyComm->statex_.get();
  EXPECT_EQ(e.commHash(), statex->commHash());
  EXPECT_EQ(e.rank(), statex->rank());
}
