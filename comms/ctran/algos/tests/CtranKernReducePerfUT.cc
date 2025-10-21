// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>

#include "comms/ctran/algos/tests/CtranAlgoDevTestUtils.h"
#include "comms/ctran/algos/tests/CtranDistAlgoDevPerfUTBase.h"
#include "comms/testinfra/TestsDistUtils.h"

class CtranKernReducePerf : public CtranDistAlgoDevPerfTestBase {
 public:
  void SetUp() override {
    CtranDistAlgoDevPerfTestBase::SetUp();
  }

  void TearDown() override {
    CtranDistAlgoDevPerfTestBase::TearDown();
  }
};

class CtranKernReducePerfTestParam
    : public CtranKernReducePerf,
      public ::testing::WithParamInterface<CtranDistAlgoDevPerfTestParams> {};

TEST_P(CtranKernReducePerfTestParam, ctranKernMultiReducePerf) {
  const auto& [testType, numElems, inplace, barrierLastElem, nGroups, beginCount, endCount, op, warmup, iters, localRankNSrcs, nDsts] =
      GetParam();

  if (testType == kTestElemRepost && inplace) {
    GTEST_SKIP() << "Do not support kTestElemRepost with inplace, skip test";
  }

  void* fn = getReduceKernelFn<int>(op);
  dim3 grid = {nGroups, 1, 1};
  dim3 blocks = {1024, 1, 1};

  startBenchmark<int>(
      "ctranKernReduce",
      [&](KernelElem* reduceELemList, cudaStream_t stream) {
        int stepId = 0;
        auto devState_d = comm_->ctranComm_->ctran_->algo->getDevState();
        void* args[] = {&reduceELemList, &devState_d, &stepId};
        CUDA_LAUNCH_KERNEL_WITH_DYNAMIC_SHM_TEST(
            fn, grid, blocks, args, sizeof(CtranAlgoDeviceState), stream);
      },
      [](KernelElem* elemList) {},
      [](KernelElem* elemList) {},
      GetParam());
}

static inline std::string getTestName(
    const testing::TestParamInfo<CtranKernReducePerfTestParam::ParamType>&
        info) {
  return std::to_string(std::get<1>(info.param)) + "nElems_" +
      (std::get<2>(info.param) ? "InPlace_" : "OutOfPlace_") +
      (std::get<3>(info.param) ? "barrierLastElem_" : "nobarrierLastElem_") +
      std::to_string(std::get<4>(info.param)) + "groups_" +
      std::to_string(std::get<5>(info.param)) + "beginCount_" +
      std::to_string(std::get<6>(info.param)) + "endCount_" +
      commOpToString(std::get<7>(info.param)) + "_" +
      std::to_string(std::get<8>(info.param)) + "wramup_" +
      std::to_string(std::get<9>(info.param)) + "iters_" +
      std::to_string(std::get<10>(info.param)) + "localRankNSrcs_" +
      std::to_string(std::get<11>(info.param)) + "nDsts";
}

INSTANTIATE_TEST_SUITE_P(
    CtranKernReducePerf,
    CtranKernReducePerfTestParam,
    ::testing::Values(
        // CompleteOrFree, numKernElems, inplace, barrierLastElem, nGroups,
        // beginCount, endCount, redOp, warmup iters, iters, localRankNSrcs,
        // nDsts
        std::make_tuple(
            kTestElemComplete,
            1,
            false,
            false,
            8,
            1 << 20, // 1MB
            1 << 26, // 32MB
            commSum,
            10,
            100,
            1, // localRankNSrcs
            1), // nDsts
        std::make_tuple(
            kTestElemComplete,
            8,
            false,
            false,
            8,
            1 << 20, // 1MB
            1 << 26, // 32MB
            commMax,
            10,
            100,
            1, // localRankNSrcs
            1), // nDsts
        std::make_tuple(
            kTestElemComplete,
            1,
            false,
            false,
            8,
            1 << 20, // 1MB
            1 << 26, // 32MB
            commSum,
            10,
            100,
            2, // localRankNSrcs > 1, so we are reducing from local rank only
            2), // nDsts
        std::make_tuple(
            kTestElemComplete,
            8,
            false,
            false,
            8,
            1 << 20, // 1MB
            1 << 26, // 32MB
            commMax,
            10,
            100,
            2, // localRankNSrcs > 1, so we are reducing from local rank only
            2), // nDsts
        std::make_tuple(
            kTestElemComplete,
            1,
            false,
            false,
            8,
            1 << 20, // 1MB
            1 << 26, // 32MB
            commSum,
            10,
            100,
            1, // localRankNSrcs
            2), // nDsts
        std::make_tuple(
            kTestElemComplete,
            8,
            false,
            false,
            8,
            1 << 20, // 1MB
            1 << 26, // 32MB
            commMax,
            10,
            100,
            1, // localRankNSrcs
            2), // nDsts
        std::make_tuple(
            kTestElemComplete,
            1,
            false,
            false,
            8,
            1 << 20, // 1MB
            1 << 26, // 32MB
            commSum,
            10,
            100,
            2, // localRankNSrcs > 1, so we are reducing from local rank only
            1), // nDsts
        std::make_tuple(
            kTestElemComplete,
            8,
            false,
            false,
            8,
            1 << 20, // 1MB
            1 << 26, // 32MB
            commMax,
            10,
            100,
            2, // localRankNSrcs > 1, so we are reducing from local rank only
            1)), // nDsts
    getTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
