// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include <cuda_runtime.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ctran/algos/common/tests/MultiTbSyncTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

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
