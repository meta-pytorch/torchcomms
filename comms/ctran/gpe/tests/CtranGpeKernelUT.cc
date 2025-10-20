// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/gpe/tests/CtranGpeUTKernels.h"
#include "comms/ctran/tests/CtranStandaloneUTUtils.h"
#include "comms/ctran/utils/Abort.h"

namespace ctran::testing {

using ::ctran::utils::Abort;

#define ASSERT_CUDASUCCESS(cmd)                                     \
  do {                                                              \
    cudaError_t ret;                                                \
    ASSERT_EQ(cudaSuccess, ret = (cmd)) << cudaGetErrorString(ret); \
  } while (0)

// let GPE and Kernel sync on algo lifecycle without errors
commResult_t CtranGpeNoopAlgoFn(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  return commSuccess;
}

using GpeKernelTestParams =
    std::tuple<std::string, OpElem::opType, KernelConfig::KernelType>;

class CtranGpeKernelTestBase
    : public ::ctran::testing::CtranStandaloneBaseTest,
      public ::testing::WithParamInterface<GpeKernelTestParams> {
 protected:
  static constexpr int kNumBlocks = CTRAN_ALGO_MAX_THREAD_BLOCKS;
  volatile int* testFlag_;
  CtranAlgoDeviceState* devState_d_{nullptr};

  // ctranComm, rank, cudaDev from parent

  cudaStream_t stream_{nullptr};

  void SetUp() override {
    setupBase();

    initCtranComm(ctran::utils::createAbort(/*enabled=*/true));

    ASSERT_CUDASUCCESS(cudaStreamCreate(&stream_));
    ASSERT_CUDASUCCESS(cudaHostAlloc(
        (void**)&testFlag_, kNumBlocks * sizeof(int), cudaHostAllocDefault));
    for (int i = 0; i < kNumBlocks; i++) {
      testFlag_[i] = KERNEL_UNSET;
    }
    ASSERT_CUDASUCCESS(cudaMalloc(&devState_d_, sizeof(CtranAlgoDeviceState)));
  }

  void TearDown() override {
    ASSERT_CUDASUCCESS(cudaFree(devState_d_));
    ASSERT_CUDASUCCESS(cudaFreeHost((void*)testFlag_));
    ASSERT_CUDASUCCESS(cudaStreamDestroy(stream_));
  }

  void runTest(void* kernelFn);
};

// This test is expected to run one of the two `UT kernelFn`s to verify GPE
// KernelFlag usage. `KernelType`s can be allowlisted to make GPE use per-block
// KernelFlags.
void CtranGpeKernelTestBase::runTest(void* kernelFn) {
  auto [name, opType, kernelType] = GetParam();

  commResult_t res = commSuccess;
  auto gpe = std::make_unique<CtranGpe>(cudaDev, ctranComm.get());

  uint64_t dummyOpCount = 100;
  std::vector<std::unique_ptr<struct OpElem>> ops;
  auto op = std::make_unique<struct OpElem>(
      opType, stream_, ctranComm.get(), dummyOpCount);
  // This is only the ensure kernelElems does not block freeing
  switch (opType) {
    case OpElem::opType::ALLGATHER:
      op->allgather.bcastElem = nullptr;
      break;
    case OpElem::opType::SEND:
      op->send.kElem = nullptr;
      break;
    case OpElem::opType::RECV:
      op->recv.kElem = nullptr;
      break;
    case OpElem::opType::REDUCESCATTER:
      op->reducescatter.interReduce = nullptr;
      break;
    case OpElem::opType::ALLTOALLV_DYNAMIC:
    case OpElem::opType::ALLTOALLV_DYNAMIC_SPLIT:
    case OpElem::opType::ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG:
      op->alltoallv_dynamic.kElem = nullptr;
      break;
    default:
      break;
  }
  ops.push_back(std::move(op));

  auto kernelConfig =
      KernelConfig(kernelType, stream_, "dummyAlgo", dummyOpCount);
  kernelConfig.numBlocks = kNumBlocks;
  kernelConfig.args.devState_d = devState_d_;
  CtranKernelAllGatherArgs dummyArgs;
  kernelConfig.algoArgs = &dummyArgs;

  res =
      gpe->submit(std::move(ops), &CtranGpeNoopAlgoFn, kernelConfig, kernelFn);
  ASSERT_EQ(res, commSuccess);

  ASSERT_CUDASUCCESS(cudaStreamSynchronize(stream_));
  EXPECT_EQ(ctranComm->getAsyncResult(), commSuccess);
}

#define TESTCASE(OP_TYPE, KERNEL_TYPE) \
  std::make_tuple(                     \
      #KERNEL_TYPE,                    \
      OpElem::opType::OP_TYPE,         \
      KernelConfig::KernelType::KERNEL_TYPE)

// test case to ensure usage of kernel flag as one flag across all blocks does
// not break
class CtranGpeKernelNonAllowlistedKernelTypeTest
    : public CtranGpeKernelTestBase {};

TEST_P(CtranGpeKernelNonAllowlistedKernelTypeTest, OneFlagAllBlocks) {
  // 1. for non-allowlisted `KernelTypes`, they work with the UT kernel
  // `CtranGpeTestOneFlagKernel` that uses 1 flag for all blocks.
  runTest((void*)CtranGpeTestOneFlagKernel);
}

// All combinations should be able to pass here
//
// For * test cases, op type doesn't matter, as we are only using the KernelType
// to control flag usage
INSTANTIATE_TEST_SUITE_P(
    OneFlagAllBlocks,
    CtranGpeKernelNonAllowlistedKernelTypeTest,
    ::testing::Values(
        TESTCASE(ALLGATHERP, ALLGATHERP),
        TESTCASE(ALLGATHERP_INIT, ALLGATHERP_INIT),
        TESTCASE(ALLGATHER, ALLGATHER),
        TESTCASE(SEND, SEND_NOTIFY),
        TESTCASE(RECV, RECV_NOTIFY),
        TESTCASE(SEND, SENDRECV_NOTIFY), // *
        TESTCASE(ALLTOALL, ALLTOALL),
        TESTCASE(ALLTOALLV, ALLTOALLV),
        TESTCASE(ALLTOALLV_DYNAMIC, ALLTOALLV_DYNAMIC),
        TESTCASE(ALLTOALLV_DYNAMIC_SPLIT, ALLTOALLV_DYNAMIC_SPLIT),
        TESTCASE(
            ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG,
            ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG),
        TESTCASE(ALLTOALL_DEDUP, ALLTOALL_DEDUP),
        TESTCASE(ALLTOALLV_DEDUP, ALLTOALLV_DEDUP_PREPARE),
        TESTCASE(ALLTOALLV_DEDUP, ALLTOALLV_DEDUP),
        TESTCASE(BROADCAST, BROADCAST),
        TESTCASE(BROADCAST, BROADCAST_UNPACK),
        TESTCASE(REDUCESCATTER, REDUCESCATTER),
        TESTCASE(PUTNOTIFY, PUTNOTIFY),
        TESTCASE(WAITNOTIFY, WAITNOTIFY),
        TESTCASE(PUTSIGNAL, PUTSIGNAL),
        TESTCASE(WAITSIGNAL, WAITSIGNAL),
        TESTCASE(SIGNAL, SIGNAL),
        TESTCASE(GET, GET)),
    [](const ::testing::TestParamInfo<GpeKernelTestParams>& info) {
      return std::get<0>(info.param);
    });

// test case to test for upgraded kernel flag usage
class CtranGpeKernelAllowlistedKernelTypeTest : public CtranGpeKernelTestBase {
};

TEST_P(CtranGpeKernelAllowlistedKernelTypeTest, PerBlockFlag) {
  // 2. for allowlisted `KernelTypes`, they work with the UT kernel
  // `CtranGpeTestPerBlockFlagKernel` that uses 1 flag per block.
  runTest((void*)CtranGpeTestPerBlockFlagKernel);
}

// Only Enabled coll kernels should be able to pass here
//
// For * test cases, op type doesn't matter, as we are only using the KernelType
// to control flag usage
INSTANTIATE_TEST_SUITE_P(
    PerBlockFlag,
    CtranGpeKernelAllowlistedKernelTypeTest,
    ::testing::Values(
        TESTCASE(ALLREDUCE, ALLREDUCE),
        TESTCASE(SEND, SEND),
        TESTCASE(RECV, RECV),
        TESTCASE(SEND, SENDRECV), // *
        TESTCASE(RECV, RECV_UNPACK),
        TESTCASE(SEND, SENDRECV_UNPACK) // *
        ),
    [](const ::testing::TestParamInfo<GpeKernelTestParams>& info) {
      return std::get<0>(info.param);
    });

} // namespace ctran::testing
