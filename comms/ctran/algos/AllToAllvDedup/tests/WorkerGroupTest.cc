// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/algos/AllToAllvDedup/WorkerGroup.h"
#include "comms/ctran/algos/AllToAllvDedup/tests/WorkerGroupTest.cuh"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/testinfra/TestUtils.h"

using ctran::alltoallvdedup::WorkerGroupSync;

__global__ void testAssignMultiWorkerGroupKernel(
    const int numRoles,
    const int* numGroups,
    const int* numWorkers,
    WorkerGroupInfo* outputs);

__global__ void testWorkerGroupSyncKernel(
    const int numGroups,
    const int numWorkers,
    const int numIter,
    WorkerGroupSync* syncs,
    int* outputs);

class WorkerGroupTest : public ::testing::Test {
 public:
  WorkerGroupTest() = default;

 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(cudaDev_));
  }

  void TearDown() override {}

  template <typename T>
  void allocDeviceArg(std::vector<T>& argH, T*& argD) {
    const size_t nBytes = sizeof(T) * argH.size();
    ASSERT_EQ(cudaMalloc((void**)&argD, nBytes), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(argD, argH.data(), nBytes, cudaMemcpyDefault), cudaSuccess);
  }

 protected:
  int cudaDev_{0};
};

TEST_F(WorkerGroupTest, AssignGroups) {
  const int numRoles = 5;
  std::vector<int> numGroupsH = {1, 1, 2, 4, 1};
  std::vector<int> numWorkersH = {5, 5, 8, 8, 1};

  int numThreadBlocks = 0;
  for (auto i = 0; i < numRoles; i++) {
    numThreadBlocks += numGroupsH[i] * numWorkersH[i];
  }

  int* numGroups = nullptr;
  int* numWorkers = nullptr;
  WorkerGroupInfo* outputs = nullptr;

  allocDeviceArg(numGroupsH, numGroups);
  allocDeviceArg(numWorkersH, numWorkers);
  CUDACHECK_ASSERT(cudaMalloc(
      (void**)&outputs, numThreadBlocks * numRoles * sizeof(WorkerGroupInfo)));
  CUDACHECK_ASSERT(cudaMemset(
      outputs, 0, numThreadBlocks * numRoles * sizeof(WorkerGroupInfo)));

  dim3 grid = {static_cast<unsigned int>(numThreadBlocks), 1, 1};
  // 1 thread per block since all should behave the same for group assignment
  dim3 block = {1, 1, 1};
  void* args[4] = {
      (void*)&numRoles, (void*)&numGroups, (void*)&numWorkers, (void*)&outputs};

  CUDACHECK_ASSERT(cudaLaunchKernel(
      (void*)testAssignMultiWorkerGroupKernel, grid, block, args));
  CUDACHECK_ASSERT(cudaDeviceSynchronize());

  // Check results
  std::vector<WorkerGroupInfo> outputsH(numThreadBlocks * numRoles);
  CUDACHECK_ASSERT(cudaMemcpy(
      outputsH.data(),
      outputs,
      numThreadBlocks * numRoles * sizeof(WorkerGroupInfo),
      cudaMemcpyDefault));

  int bid = 0;
  int startBid = 0;
  for (auto i = 0; i < numRoles; i++) {
    const int numGroups = numGroupsH[i];
    const int numWorkers = numWorkersH[i];

    const auto expStart = startBid;
    const auto expEnd = startBid + numGroups * numWorkers - 1;
    for (auto j = 0; j < numGroups; j++) {
      for (auto k = 0; k < numWorkers; k++) {
        const auto& groupH = outputsH[bid * numRoles + i];
        const auto curWg = fmt::format("role {} group {} worker {}", i, j, k);
        EXPECT_EQ(groupH.numGroups, numGroups);
        EXPECT_EQ(groupH.numWorkers, numWorkers);
        EXPECT_EQ(groupH.start, expStart) << " at bid " << bid << " " << curWg;
        EXPECT_EQ(groupH.end, expEnd) << " at bid " << bid << " " << curWg;
        if (bid >= expStart && bid <= expEnd) {
          EXPECT_EQ(groupH.groupId, j) << " at bid " << bid << " " << curWg;
          EXPECT_EQ(groupH.workerId, k) << " at bid " << bid << " " << curWg;
        } else {
          EXPECT_EQ(groupH.groupId, -1) << " at bid " << bid << " " << curWg;
          EXPECT_EQ(groupH.workerId, -1) << " at bid " << bid << " " << curWg;
        }
        bid++;
      }
    }
    startBid += numGroups * numWorkers;
  }

  CUDACHECK_ASSERT(cudaFree(outputs));
  CUDACHECK_ASSERT(cudaFree(numGroups));
  CUDACHECK_ASSERT(cudaFree(numWorkers));
}

// Lightweight sync check within each group; more comprehensive tests has been
// covered in algos/comms/tests/MultiTbSyncTest.cc
TEST_F(WorkerGroupTest, Sync) {
  const int numGroups = 5;
  const int numWorkers = 8;
  const int numIter = 100;

  const int numThreadBlocks = numGroups * numWorkers;

  int* outputs = nullptr;
  WorkerGroupSync* syncs = nullptr;

  // every group uses a different sync; this is a simple reset for test.
  // Alternatively using MultiTbSyncDev::reset to reset each counter from
  // kernel, before any thread block starts. I.e., reset in a separate kernel.
  std::vector<WorkerGroupSync> syncsH(numGroups, {0, 0, 0, 0});
  allocDeviceArg(syncsH, syncs);
  std::vector<int> outputsH(numThreadBlocks * numIter);
  allocDeviceArg(outputsH, outputs);

  dim3 grid = {static_cast<unsigned int>(numThreadBlocks), 1, 1};
  dim3 block = {128, 1, 1};
  void* args[5] = {
      (void*)&numGroups,
      (void*)&numWorkers,
      (void*)&numIter,
      (void*)&syncs,
      (void*)&outputs};

  CUDACHECK_ASSERT(
      cudaLaunchKernel((void*)testWorkerGroupSyncKernel, grid, block, args));
  CUDACHECK_ASSERT(cudaDeviceSynchronize());

  // Check results
  CUDACHECK_ASSERT(cudaMemcpy(
      outputsH.data(),
      outputs,
      numThreadBlocks * numIter * sizeof(int),
      cudaMemcpyDefault));

  for (auto i = 0; i < numIter; i++) {
    for (auto b = 0; b < numThreadBlocks; b++) {
      const auto lead = b / numWorkers * numWorkers;
      const auto exp = i * numThreadBlocks + lead;
      EXPECT_EQ(outputsH[i * numThreadBlocks + b], exp)
          << " at iter " << i << " blockIdx.x " << b << " with lead " << lead;
    }
  }

  CUDACHECK_ASSERT(cudaFree(syncs));
  CUDACHECK_ASSERT(cudaFree(outputs));
}
