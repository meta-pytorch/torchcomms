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
#include "comms/ctran/tests/CtranXPlatUtUtils.h"

class CtranGpeTest : public ::testing::Test {
 public:
  CtranGpe* gpe;
  int cudaDev;
  std::unique_ptr<TestCtranCommRAII> dummyCommRAII;
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
    dummyCommRAII = createDummyCtranComm();
    dummyComm = dummyCommRAII->ctranComm;
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
  std::unique_ptr<TestCtranCommRAII> dummyCommRAII;
  CtranComm* dummyComm{nullptr};
  int cudaDev;
  CtranGpeKernelTest() = default;

 protected:
  void SetUp() override {
    cudaDev = 0;
    FB_CUDACHECKIGNORE(cudaSetDevice(cudaDev));

    // Ensure logger is initialized
    ncclCvarInit();

    dummyCommRAII = createDummyCtranComm();
    dummyComm = dummyCommRAII->ctranComm;

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
  while (gpe->numInUseKernelFlags() > 0)
    ;
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
