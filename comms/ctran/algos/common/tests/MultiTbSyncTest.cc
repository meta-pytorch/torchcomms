// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include "comms/ctran/algos/common/tests/MultiTbSyncTest.cuh"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ctran/tests/CtranXPlatUtUtils.h"

extern __global__ void MultiTbSyncTestResetKernel(
    int* shmCnts,
    int numCounters);

template <TestSyncType syncType>
extern __global__ void MultiTbSyncTestKernel(
    const int numWorkers,
    const int numIter,
    const int count,
    int* shmData,
    int* shmCnts,
    int* outData);

extern __global__ void MultiTbBcastTestKernel(
    const int numWorkers,
    const int numIter,
    const int count,
    int* shmData,
    int* shmCnts,
    int* outData);

template <PerfSyncType syncType>
extern __global__ void MultiTbSyncTestPerfKernel(
    const int numWorkers,
    const int numIter,
    const int runId,
    int* shmCnt);

class CtranMultiTbSyncTest : public ::testing::Test {
 public:
  CtranMultiTbSyncTest() = default;

 protected:
  static void SetUpTestCase() {}

  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(cudaDev_));
    CUDACHECK_TEST(cudaEventCreate(&start_));
    CUDACHECK_TEST(cudaEventCreate(&stop_));
  }
  void TearDown() override {
    CUDACHECK_TEST(cudaEventDestroy(start_));
    CUDACHECK_TEST(cudaEventDestroy(stop_));
  }

 protected:
  int cudaDev_{0};
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

std::vector<std::string> kPerfSyncTypeStrs = {
    "Barrier",
    "Fence",
    "Dispatch",
    "Join",
    "Signal",
    "SignalWithSync",
    "Bcast",
    "Cluster"};

std::unordered_map<PerfSyncType, void*> kSyncTypeToPerfKernel = {
    {PerfSyncType::kBarrier,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kBarrier>},
    {PerfSyncType::kFence,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kFence>},
    {PerfSyncType::kDispatch,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kDispatch>},
    {PerfSyncType::kJoin,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kJoin>},
    {PerfSyncType::kSignal,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kSignal>},
    {PerfSyncType::kSignalWithSync,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kSignalWithSync>},
    {PerfSyncType::kBcast,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kBcast>},
    {PerfSyncType::kClusterSync,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kClusterSync>}};

std::vector<std::string> kTestSyncTypeStrs = {
    "FullBarrier",
    "DispatchJoin",
    "OneSideSignal",
    "BcastVal"};
std::unordered_map<TestSyncType, void*> kSyncTypeToTestKernel = {
    {TestSyncType::kFullBarrier,
     (void*)MultiTbSyncTestKernel<TestSyncType::kFullBarrier>},
    {TestSyncType::kDispatchJoin,
     (void*)MultiTbSyncTestKernel<TestSyncType::kDispatchJoin>},
    {TestSyncType::kOneSideSignal,
     (void*)MultiTbSyncTestKernel<TestSyncType::kOneSideSignal>},
    {TestSyncType::kBcastVal, (void*)MultiTbBcastTestKernel}};

void launchPerfReset(
    const int numWorkers,
    void* resetArgs[2],
    const PerfSyncType syncType) {
  dim3 resetGrid = {(unsigned int)1, 1, 1};
  dim3 resetBlock = {1, 1, 1};
  void* resetFn = (void*)MultiTbSyncTestResetKernel;
  // no reset for cluster
  if (syncType != PerfSyncType::kClusterSync) {
    ASSERT_EQ(
        cudaLaunchKernel(resetFn, resetGrid, resetBlock, resetArgs),
        cudaSuccess);
  }
}

// TODO: add AMD support
#ifndef __HIP_PLATFORM_AMD__
void setupClusterLaunchConfig(
    const int numWorkers,
    cudaLaunchAttribute* attribute,
    cudaLaunchConfig_t& launchConfig) {
  if (attribute) {
    launchConfig.attrs = attribute;
    launchConfig.numAttrs = 1;
  }
  launchConfig.gridDim = numWorkers;
  launchConfig.blockDim = 256;
}
#endif

void launchPerfTest(
    const int numWorkers,
    void* execArgs[4],
    const PerfSyncType syncType) {
  void* fn = kSyncTypeToPerfKernel.at(syncType);
  if (syncType == PerfSyncType::kClusterSync) {
// TODO: add AMD support
#ifndef __HIP_PLATFORM_AMD__
    cudaLaunchConfig_t launchConfig = {0};
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = numWorkers; // Cluster size in X-dimension
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    setupClusterLaunchConfig(numWorkers, attribute, launchConfig);
    ASSERT_EQ(cudaLaunchKernelExC(&launchConfig, fn, execArgs), cudaSuccess);
#endif
  } else {
    dim3 grid = {(unsigned int)numWorkers, 1, 1};
    dim3 block = {256, 1, 1};
    ASSERT_EQ(cudaLaunchKernel(fn, grid, block, execArgs), cudaSuccess);
  }
}

class MultiTbSyncPerfParamFixture
    : public CtranMultiTbSyncTest,
      // number of thread blocks for each sync group
      public ::testing::WithParamInterface<std::tuple<int, PerfSyncType>> {};

TEST_P(MultiTbSyncPerfParamFixture, Perf) {
  auto [numWorkers, syncType] = GetParam();

  const int numIter = 1000;
  const int warmRuns = 10;
  const int numRuns = 100;

  if (syncType == PerfSyncType::kClusterSync) {
// TODO: add AMD support
#ifndef __HIP_PLATFORM_AMD__
    int maxClusterSize = 0;
    cudaLaunchConfig_t launchConfig = {0};
    setupClusterLaunchConfig(numWorkers, nullptr, launchConfig);
    ASSERT_EQ(
        cudaOccupancyMaxPotentialClusterSize(
            &maxClusterSize,
            (const void*)kSyncTypeToPerfKernel.at(syncType),
            &launchConfig),
        cudaSuccess);

    if (maxClusterSize < numWorkers) {
      GTEST_SKIP() << "Unsupported cluster size " << numWorkers << ", max "
                   << maxClusterSize << std::endl;
    }
#else
    GTEST_SKIP() << "Cluster sync test not supported on AMD";
#endif
  }

  int* shmCnts = nullptr;
  int numCnts = 3; // bcast require 3
  ASSERT_EQ(cudaMalloc((void**)&shmCnts, sizeof(int)), cudaSuccess);
  void* resetArgs[2] = {(void*)&shmCnts, (void*)&numCnts};
  void* execArgs[4] = {
      (void*)&numWorkers,
      (void*)&numIter,
      nullptr /* update in iteration */,
      (void*)&shmCnts};

  // warm up
  for (int i = 0; i < warmRuns; i++) {
    launchPerfReset(numWorkers, resetArgs, syncType);

    execArgs[2] = &i; // runId
    launchPerfTest(numWorkers, execArgs, syncType);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(start_, 0), cudaSuccess);
  for (int i = 0; i < numRuns; i++) {
    launchPerfReset(numWorkers, resetArgs, syncType);

    execArgs[2] = &i;
    launchPerfTest(numWorkers, execArgs, syncType);
  }
  ASSERT_EQ(cudaEventRecord(stop_, 0), cudaSuccess);
  ASSERT_EQ(cudaEventSynchronize(stop_), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float timeMs;
  ASSERT_EQ(cudaEventElapsedTime(&timeMs, start_, stop_), cudaSuccess);

  std::cout
      << fmt::format(
             "{} Perf: {} numWorkers, {:2f} us per iteration, averaged from {} runs, each with {} iterations",
             kPerfSyncTypeStrs[(int)syncType],
             numWorkers,
             timeMs * 1e3 / numIter / numRuns,
             numRuns,
             numIter)
      << std::endl;

  ASSERT_EQ(cudaFree(shmCnts), cudaSuccess);
}

class MultiTbSyncTestParamFixture
    : public CtranMultiTbSyncTest,
      // number of thread blocks for each sync group
      public ::testing::WithParamInterface<std::tuple<int, TestSyncType>> {};

TEST_P(MultiTbSyncTestParamFixture, Test) {
  auto [numWorkers, syncType] = GetParam();

  const int numIter = 1000;
  // bcast uses shmData to exchange single integer, others use shmData for
  // multi-element data
  const int count = syncType == TestSyncType::kBcastVal ? 1 : 16805;

  int* shmCnts = nullptr;
  // at most numWorkers counters (only for signal) or 1-2 for others
  int numCnts = std::max(numWorkers, 2);
  ASSERT_EQ(cudaMalloc((void**)&shmCnts, sizeof(int) * numCnts), cudaSuccess);

  int *shmData = nullptr, *outputData = nullptr;
  size_t size = count * numWorkers * sizeof(int);

  ASSERT_EQ(cudaMalloc((void**)&shmData, size), cudaSuccess);
  ASSERT_EQ(cudaMalloc((void**)&outputData, size * numIter), cudaSuccess);
  cudaMemset(shmData, 0, size);
  cudaMemset(outputData, 0, size * numIter);

  void* resetArgs[2] = {(void*)&shmCnts, (void*)&numCnts};
  dim3 resetGrid = {(unsigned int)1, 1, 1};
  dim3 resetBlock = {1, 1, 1};
  void* resetFn = (void*)MultiTbSyncTestResetKernel;
  ASSERT_EQ(
      cudaLaunchKernel(resetFn, resetGrid, resetBlock, resetArgs), cudaSuccess);

  dim3 grid = {(unsigned int)numWorkers, 1, 1};
  dim3 block = {256, 1, 1};
  void* execArgs[6] = {
      (void*)&numWorkers,
      (void*)&numIter,
      (void*)&count,
      (void*)&shmData,
      (void*)&shmCnts,
      (void*)&outputData};
  ASSERT_EQ(
      cudaLaunchKernel(
          kSyncTypeToTestKernel.at(syncType), grid, block, execArgs),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Check output for each iteration
  std::vector<int> checkData(size * numIter, -1);
  ASSERT_EQ(
      cudaMemcpy(
          checkData.data(), outputData, size * numIter, cudaMemcpyDefault),
      cudaSuccess);

  for (int x = 0; x < numIter; x++) {
    const auto iterOffset = x * count * numWorkers;
    for (auto workerId = 0; workerId < numWorkers; workerId++) {
      const auto checkOffset = iterOffset + workerId * count;
      auto it = checkData.begin() + checkOffset;
      std::vector<int> dataP = std::vector<int>(it, it + count);
      std::vector<int> expData(count);

      const auto nextWorkerId = (workerId + 1) % numWorkers;
      for (auto i = 0; i < count; i++) {
        expData[i] = syncType == TestSyncType::kBcastVal
            ? WORKER_ID_TO_VAL(0, count, i, x)
            : WORKER_ID_TO_VAL(nextWorkerId, count, i, x);
      }
      ASSERT_EQ(dataP, expData) << fmt::format(
          "workerId {}/{} at iteration {}", workerId, numWorkers, x);
    }
  }

  ASSERT_EQ(cudaFree(shmCnts), cudaSuccess);
  ASSERT_EQ(cudaFree(shmData), cudaSuccess);
  ASSERT_EQ(cudaFree(outputData), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    MultiTbSyncTestParamFixture,
    ::testing::Combine(
        ::testing::Values(2, 4, 8, 16, 32),
        ::testing::Values(
            TestSyncType::kFullBarrier,
            TestSyncType::kDispatchJoin,
            TestSyncType::kOneSideSignal,
            TestSyncType::kBcastVal)),
    [&](const testing::TestParamInfo<MultiTbSyncTestParamFixture::ParamType>&
            info) {
      const auto syncType = std::get<1>(info.param);
      return std::to_string(std::get<0>(info.param)) + "numWorkers_" +
          kTestSyncTypeStrs[(int)syncType];
    });

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    MultiTbSyncPerfParamFixture,
    ::testing::Combine(
        // 16 and more may hang, likely because not all blocks can be scheduled
        // at the same time
        ::testing::Values(2, 4, 8, 16, 32, 64),
        ::testing::Values(
            PerfSyncType::kBarrier,
            PerfSyncType::kFence,
            PerfSyncType::kDispatch,
            PerfSyncType::kJoin,
            PerfSyncType::kSignal,
            PerfSyncType::kSignalWithSync,
            PerfSyncType::kBcast,
            PerfSyncType::kClusterSync)),
    [&](const testing::TestParamInfo<MultiTbSyncPerfParamFixture::ParamType>&
            info) {
      const auto syncType = std::get<1>(info.param);
      return std::to_string(std::get<0>(info.param)) + "numWorkers_" +
          kPerfSyncTypeStrs[(int)syncType];
    });
