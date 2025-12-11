// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/testinfra/TestXPlatUtils.h"

using ctran::algos::GpeKernelSync;

extern __global__ void
GpeKernelSyncKernel(GpeKernelSync* sync, int* data, int numElem, int nSteps);
extern __global__ void GpeKernelSyncResetKernel(
    GpeKernelSync* sync,
    const int nworkers);

class CtranGpeKernelSyncTest : public ::testing::Test {
 public:
  CtranGpeKernelSyncTest() = default;

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

enum class ResetMode {
  kFromHost,
  kFromDevice,
};

class CtranGpeKernelSyncTestParamFixture
    : public CtranGpeKernelSyncTest,
      // number of thread blocks for each sync group
      public ::testing::WithParamInterface<std::tuple<int, ResetMode>> {};

TEST_P(CtranGpeKernelSyncTestParamFixture, kernelSync) {
  auto [nWorkers, resetType] = GetParam();

  int numElem = 8192;
  int niter = 10;
  void* ptr = nullptr;
  ASSERT_EQ(
      cudaHostAlloc(
          &ptr,
          sizeof(GpeKernelSync) + sizeof(int) * numElem * 3,
          cudaHostAllocDefault),
      cudaSuccess);

  // Assign sync and data pointers from the allocated memory
  GpeKernelSync* sync = reinterpret_cast<GpeKernelSync*>(ptr);

  int* data = reinterpret_cast<int*>(
      reinterpret_cast<char*>(ptr) + sizeof(GpeKernelSync));
  for (int e = 0; e < numElem; ++e) {
    data[e] = e;
  }

  if (resetType == ResetMode::kFromDevice) {
    std::array<void*, 2> resetArgs;
    resetArgs.at(0) = &sync;
    resetArgs.at(1) = &nWorkers;
    CUDACHECK_ASSERT(cudaLaunchKernel(
        (const void*)GpeKernelSyncResetKernel,
        {1, 1, 1},
        {32, 1, 1},
        resetArgs.data(),
        0,
        0));
    CUDACHECK_ASSERT(cudaDeviceSynchronize());
  } else {
    new (sync) GpeKernelSync(nWorkers);
  }

  std::array<void*, 4> kernArgs;
  kernArgs.at(0) = &sync;
  kernArgs.at(1) = &data;
  kernArgs.at(2) = &numElem;
  kernArgs.at(3) = &niter;
  dim3 grid = {(unsigned int)nWorkers, 1, 1};
  dim3 blocks = {128, 1, 1};
  ASSERT_EQ(
      cudaFuncSetAttribute(
          (const void*)GpeKernelSyncKernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(CtranAlgoDeviceState)),
      cudaSuccess);
  ASSERT_EQ(
      cudaLaunchKernel(
          (const void*)GpeKernelSyncKernel,
          grid,
          blocks,
          kernArgs.data(),
          sizeof(CtranAlgoDeviceState),
          0),
      cudaSuccess);

  for (int i = 0; i < niter; ++i) {
    for (int e = 0; e < numElem; ++e) {
      data[e] += i;
    }

    sync->post(i);

    while (!sync->isComplete(i)) {
      std::this_thread::yield();
    }

    for (int e = 0; e < numElem; ++e) {
      ASSERT_EQ(data[e], e + i * (i + 1)) << " at " << e << " iteration " << i;
    }
  }

  ASSERT_EQ(cudaFreeHost(ptr), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranGpeKernelSyncTestParamFixture,
    ::testing::Combine(
        ::testing::Values(1, 2, 4, 8, 16, 32, 64),
        ::testing::Values(ResetMode::kFromHost, ResetMode::kFromDevice)),
    [&](const testing::TestParamInfo<
        CtranGpeKernelSyncTestParamFixture::ParamType>& info) {
      const auto resetTypeStr = std::get<1>(info.param) == ResetMode::kFromHost
          ? "ResetFromHost"
          : "ResetFromDevice";
      return std::to_string(std::get<0>(info.param)) + "numWorkers_" +
          resetTypeStr;
    });
