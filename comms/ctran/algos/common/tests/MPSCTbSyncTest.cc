// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/common/MPSCTbSync.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include "comms/ctran/utils/Utils.h"

using ctran::algos::MPSCTbSync;

__global__ void MPSCTbSyncTestKernel(
    int numProducers,
    int numIter,
    int count,
    int* shmData,
    MPSCTbSync<>* sync,
    int* outData);

__global__ void SPSCTbSyncTestKernel(
    int numIter,
    int count, // number of elements in data
    int* shmData, // count elements
    MPSCTbSync<1>* postSync,
    MPSCTbSync<1>* completeSync,
    // received data from producer by consumer, total count * numIter elements
    // returned to host for test correctness check
    int* outData);

class CtranMPSCTbSyncTest : public ::testing::Test {
 public:
  CtranMPSCTbSyncTest() = default;

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

class CtranMPSCTbSyncTestParamFixture
    : public CtranMPSCTbSyncTest,
      // numProducers, count
      public ::testing::WithParamInterface<std::tuple<int, int>> {};

TEST_P(CtranMPSCTbSyncTestParamFixture, Check) {
  auto [numProducers, count] = GetParam();

  const int numIter = 50;

  MPSCTbSync<>* sync = nullptr;
  MPSCTbSync<> syncH = MPSCTbSync<>(numProducers);

  ASSERT_EQ(cudaMalloc((void**)&sync, sizeof(MPSCTbSync<>)), cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(sync, &syncH, sizeof(MPSCTbSync<>), cudaMemcpyDefault),
      cudaSuccess);

  int *shmData = nullptr, *outputData = nullptr;
  size_t size =
      ctran::utils::align(count * numProducers * sizeof(int), (size_t)16);

  ASSERT_EQ(cudaMalloc((void**)&shmData, size), cudaSuccess);
  ASSERT_EQ(cudaMalloc((void**)&outputData, size * numIter), cudaSuccess);
  cudaMemset(shmData, 0, size);
  cudaMemset(outputData, 0, size * numIter);

  dim3 grid = {(unsigned int)numProducers + 1, 1, 1};
  dim3 block = {256, 1, 1};
  void* execArgs[6] = {
      (void*)&numProducers,
      (void*)&numIter,
      (void*)&count,
      (void*)&shmData,
      (void*)&sync,
      (void*)&outputData};
  ASSERT_EQ(
      cudaLaunchKernel((void*)MPSCTbSyncTestKernel, grid, block, execArgs),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Check output for each iteration
  std::vector<int> checkData(size * numIter, -1);
  ASSERT_EQ(
      cudaMemcpy(
          checkData.data(), outputData, size * numIter, cudaMemcpyDefault),
      cudaSuccess);
  for (int x = 0; x < numIter; x++) {
    for (auto p = 0; p < numProducers; p++) {
      const auto offset = x * numProducers * count + p * count;
      auto it = checkData.begin() + offset;
      std::vector<int> dataP = std::vector<int>(it, it + count);
      std::vector<int> expData(count);
      for (auto c = 0; c < count; c++) {
        expData[c] = x * count * numProducers + p * count + c;
      }
      ASSERT_EQ(dataP, expData) << "iter " << x << " producer " << p
                                << " at offset " << offset << std::endl;
#if 0
      std::cout << "iter " << x << " received from producer " << p << ": "
                << folly::join(", ", dataP) << std::endl;
#endif
    }
  }

  ASSERT_EQ(cudaFree(sync), cudaSuccess);
  ASSERT_EQ(cudaFree(shmData), cudaSuccess);
  ASSERT_EQ(cudaFree(outputData), cudaSuccess);
}

TEST_F(CtranMPSCTbSyncTestParamFixture, Perf) {
  // do not actually copy anything, just testing sync latency
  int count = 0;

  int startNumProducers = 1;
  int endNumProducers = 64;

  for (int numProducers = startNumProducers; numProducers <= endNumProducers;
       numProducers *= 2) {
    const int numIter = 1000;
    MPSCTbSync<>* sync = nullptr;
    MPSCTbSync<> syncH = MPSCTbSync<>(numProducers);
    ASSERT_EQ(cudaMalloc((void**)&sync, sizeof(MPSCTbSync<>)), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(sync, &syncH, sizeof(MPSCTbSync<>), cudaMemcpyDefault),
        cudaSuccess);
    int *shmData = nullptr, *outputData = nullptr;
    size_t size =
        ctran::utils::align(count * numProducers * sizeof(int), (size_t)16);
    ASSERT_EQ(cudaMalloc((void**)&shmData, size), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void**)&outputData, size * numIter), cudaSuccess);
    cudaMemset(shmData, 0, size);
    cudaMemset(outputData, 0, size * numIter);
    dim3 grid = {(unsigned int)numProducers + 1, 1, 1};
    dim3 block = {256, 1, 1};
    void* execArgs[6] = {
        (void*)&numProducers,
        (void*)&numIter,
        (void*)&count,
        (void*)&shmData,
        (void*)&sync,
        (void*)&outputData};
    // Warmup
    ASSERT_EQ(
        cudaLaunchKernel((void*)MPSCTbSyncTestKernel, grid, block, execArgs),
        cudaSuccess);
    // Performance measurement
    ASSERT_EQ(cudaEventRecord(start_), cudaSuccess);
    ASSERT_EQ(
        cudaLaunchKernel((void*)MPSCTbSyncTestKernel, grid, block, execArgs),
        cudaSuccess);
    ASSERT_EQ(cudaEventRecord(stop_), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    float gpuTimeMs = 0;
    ASSERT_EQ(cudaEventElapsedTime(&gpuTimeMs, start_, stop_), cudaSuccess);
    printf(
        "MPSCTbSyncTestKernel with numProducers %d and %d iters: %f us\n",
        numProducers,
        numIter,
        gpuTimeMs * 1000);
    ASSERT_EQ(cudaFree(sync), cudaSuccess);
    ASSERT_EQ(cudaFree(shmData), cudaSuccess);
    ASSERT_EQ(cudaFree(outputData), cudaSuccess);
  }
}

TEST_P(CtranMPSCTbSyncTestParamFixture, CheckSPSCTestCase) {
  auto [numProducers, count] = GetParam();

  int numIter = 10;

  MPSCTbSync<1>* postSync = nullptr;
  MPSCTbSync<1> syncH(1);

  ASSERT_EQ(cudaMalloc((void**)&postSync, sizeof(MPSCTbSync<1>)), cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(postSync, &syncH, sizeof(MPSCTbSync<1>), cudaMemcpyDefault),
      cudaSuccess);

  MPSCTbSync<1>* completeSync = nullptr;
  MPSCTbSync<1> completeSyncH(1);
  ASSERT_EQ(
      cudaMalloc((void**)&completeSync, sizeof(MPSCTbSync<1>)), cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(
          completeSync,
          &completeSyncH,
          sizeof(MPSCTbSync<1>),
          cudaMemcpyDefault),
      cudaSuccess);

  int *shmData = nullptr, *outputData = nullptr;
  size_t size = ctran::utils::align(count * sizeof(int), (size_t)16);

  ASSERT_EQ(cudaMalloc((void**)&shmData, size), cudaSuccess);
  ASSERT_EQ(cudaMalloc((void**)&outputData, size * numIter), cudaSuccess);
  cudaMemset(shmData, 0, size);
  cudaMemset(outputData, 0, size * numIter);

  // Launch 2 blocks: one producer (blockIdx.x=0) and one consumer
  // (blockIdx.x=1)
  dim3 grid = {2, 1, 1};
  dim3 block = {256, 1, 1};
  void* execArgs[6] = {
      (void*)&numIter,
      (void*)&count,
      (void*)&shmData,
      (void*)&postSync,
      (void*)&completeSync,
      (void*)&outputData};
  ASSERT_EQ(
      cudaLaunchKernel((void*)SPSCTbSyncTestKernel, grid, block, execArgs),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Check output for each iteration
  std::vector<int> checkData(size * numIter, -1);
  ASSERT_EQ(
      cudaMemcpy(
          checkData.data(), outputData, size * numIter, cudaMemcpyDefault),
      cudaSuccess);
  for (int x = 0; x < numIter; x++) {
    const auto offset = x * count;
    auto it = checkData.begin() + offset;
    std::vector<int> dataIter = std::vector<int>(it, it + count);
    std::vector<int> expData(count);
    for (auto c = 0; c < count; c++) {
      expData[c] = x * count + c;
    }
    ASSERT_EQ(dataIter, expData)
        << "iter " << x << " at offset " << offset << std::endl;
#if 0
    std::cout << "iter " << x << " received data: "
              << folly::join(", ", dataIter) << std::endl;
#endif
  }

  ASSERT_EQ(cudaFree(postSync), cudaSuccess);
  ASSERT_EQ(cudaFree(completeSync), cudaSuccess);
  ASSERT_EQ(cudaFree(shmData), cudaSuccess);
  ASSERT_EQ(cudaFree(outputData), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranMPSCTbSyncTestParamFixture,
    ::testing::Values(
        // numProducers, count
        std::make_tuple(2, 1024),
        std::make_tuple(4, 1024),
        std::make_tuple(8, 15),
        std::make_tuple(8, 1048571)),
    [&](const testing::TestParamInfo<
        CtranMPSCTbSyncTestParamFixture::ParamType>& info) {
      return std::to_string(std::get<0>(info.param)) + "numProducers_" +
          std::to_string(std::get<1>(info.param)) + "count";
    });
