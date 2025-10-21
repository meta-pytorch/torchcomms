// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/AllToAllvDedup/FwdGroupSync.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/testinfra/TestUtils.h"

using ctran::alltoallvdedup::FwdGroupSync;

extern __global__ void fwdGroupSyncTestInitKernel(
    int numGroups,
    int numWorkers,
    FwdGroupSync* fwdGroupSync);

extern __global__ void fwdGroupSyncTestKernel(
    int numGroups,
    int numWorkers,
    int numIter,
    FwdGroupSync* fwdGroupSync,
    // output of numGroups * numWorkers * iter steps
    int* steps);
__global__ void fwdGroupSyncPerfBenchKernel(
    int numGroups,
    int numWorkers,
    int numIter,
    FwdGroupSync* fwdGroupSync);

class CtranFwdGroupSyncTest : public ::testing::Test {
 public:
  CtranFwdGroupSyncTest() = default;

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

class CtranFwdGroupSyncTestParamFixture
    : public CtranFwdGroupSyncTest,
      // numGroups, numWorkers
      public ::testing::WithParamInterface<std::tuple<int, int>> {};

TEST_P(CtranFwdGroupSyncTestParamFixture, Check) {
  auto [numGroups, numWorkers] = GetParam();

  const int numIter = 50;

  FwdGroupSync* fwdGroupSync = nullptr;
  ASSERT_EQ(
      cudaMalloc((void**)&fwdGroupSync, sizeof(FwdGroupSync)), cudaSuccess);

  int* outputSteps = nullptr;
  ASSERT_EQ(
      cudaHostAlloc(
          (void**)&outputSteps,
          numGroups * numWorkers * numIter * sizeof(int),
          cudaHostAllocDefault),
      cudaSuccess);

  dim3 grid = {1, 1, 1};
  dim3 block = {kWarpSize, 1, 1};
  void* initArgs[3] = {
      (void*)&numGroups,
      (void*)&numWorkers,
      (void*)&fwdGroupSync,
  };
  ASSERT_EQ(
      cudaLaunchKernel(
          (void*)fwdGroupSyncTestInitKernel, grid, block, initArgs),
      cudaSuccess);

  grid = {(unsigned int)numGroups * numWorkers, 1, 1};
  block = {256, 1, 1};
  void* execArgs[5] = {
      (void*)&numGroups,
      (void*)&numWorkers,
      (void*)&numIter,
      (void*)&fwdGroupSync,
      (void*)&outputSteps};
  ASSERT_EQ(
      cudaLaunchKernel((void*)fwdGroupSyncTestKernel, grid, block, execArgs),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Check output, expect at each iteration, each worker in a group will have
  // the same step, but different from other groups
  for (int x = 0; x < numIter; x++) {
    std::unordered_set<int> stepsAcrossGroupsSet;
    std::vector<int> stepsAcrossGroupsVec; // for sorting based on gId
    for (int g = 0; g < numGroups; g++) {
      auto outOffset = (g * numWorkers + 0) * numIter + x;
      const auto stepInGroup = outputSteps[outOffset];
      ASSERT_FALSE(stepsAcrossGroupsSet.contains(stepInGroup))
          << "at iter " << x << " group " << g << " step " << stepInGroup
          << " already exists";

      stepsAcrossGroupsSet.insert(stepInGroup);
      stepsAcrossGroupsVec.push_back(stepInGroup);

      for (int w = 0; w < numWorkers; w++) {
        const auto outOffsetW = (g * numWorkers + w) * numIter + x;
        const auto myStep = outputSteps[outOffsetW];
        ASSERT_EQ(myStep, stepInGroup)
            << "at iter " << x << " group " << g << " worker " << w;
      }
    }
#if 0
    std::cout << "iter " << x << " steps across groups: "
              << folly::join(", ", stepsAcrossGroupsVec) << std::endl;
#endif
    stepsAcrossGroupsSet.clear();
    stepsAcrossGroupsVec.clear();
  }

  ASSERT_EQ(cudaFree(fwdGroupSync), cudaSuccess);
  ASSERT_EQ(cudaFreeHost(outputSteps), cudaSuccess);
}

TEST_P(CtranFwdGroupSyncTestParamFixture, Perf) {
  auto [numGroups, numWorkers] = GetParam();

  const int numIter = 1000;

  FwdGroupSync* fwdGroupSync = nullptr;
  ASSERT_EQ(
      cudaMalloc((void**)&fwdGroupSync, sizeof(FwdGroupSync)), cudaSuccess);

  dim3 grid = {1, 1, 1};
  dim3 block = {kWarpSize, 1, 1};
  void* initArgs[3] = {
      (void*)&numGroups,
      (void*)&numWorkers,
      (void*)&fwdGroupSync,
  };
  ASSERT_EQ(
      cudaLaunchKernel(
          (void*)fwdGroupSyncTestInitKernel, grid, block, initArgs),
      cudaSuccess);

  grid = {(unsigned int)numGroups * numWorkers, 1, 1};
  block = {256, 1, 1};
  void* execArgs[4] = {
      (void*)&numGroups,
      (void*)&numWorkers,
      (void*)&numIter,
      (void*)&fwdGroupSync};

  // Report performance
  constexpr int warmupIter = 100;
  for (int x = 0; x < warmupIter; x++) {
    ASSERT_EQ(
        cudaLaunchKernel(
            (void*)fwdGroupSyncPerfBenchKernel, grid, block, execArgs),
        cudaSuccess);
  }

  constexpr int execIter = 1000;
  ASSERT_EQ(cudaEventRecord(start_), cudaSuccess);
  for (int x = 0; x < execIter; x++) {
    ASSERT_EQ(
        cudaLaunchKernel(
            (void*)fwdGroupSyncPerfBenchKernel, grid, block, execArgs),
        cudaSuccess);
  }
  ASSERT_EQ(cudaEventRecord(stop_), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float gpuTimeMs_ = 0;
  ASSERT_EQ(cudaEventElapsedTime(&gpuTimeMs_, start_, stop_), cudaSuccess);
  printf(
      "Executed fwdGroupSyncTestKernel with %d numGroups %d numWorkers, total latency of %d updates %.2f us\n",
      numGroups,
      numWorkers,
      numIter,
      gpuTimeMs_ * 1000 / execIter);

  ASSERT_EQ(cudaFree(fwdGroupSync), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranFwdGroupSyncTestParamFixture,
    ::testing::Values(
        // numGroups, numWorkers
        std::make_tuple(1, 1),
        std::make_tuple(2, 1),
        std::make_tuple(2, 2),
        std::make_tuple(4, 2),
        std::make_tuple(2, 4),
        std::make_tuple(4, 4)),
    [&](const testing::TestParamInfo<
        CtranFwdGroupSyncTestParamFixture::ParamType>& info) {
      return std::to_string(std::get<0>(info.param)) + "numGroups_" +
          std::to_string(std::get<1>(info.param)) + "numWorkers";
    });
