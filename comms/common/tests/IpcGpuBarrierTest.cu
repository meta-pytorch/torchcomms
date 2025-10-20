// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/Random.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <folly/stop_watch.h>
#include <gtest/gtest.h>
#include "comms/common/IpcGpuBarrier.cuh"
#include "comms/common/IpcMemHandler.h"
#include "comms/common/tests/TestBaselineBootstrap.h"
#include "comms/rcclx/develop/meta/lib/tests/RcclxTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::rcclx;
using namespace meta::comms;

__global__ void
setDeviceMboxFlag(DeviceMailbox mbox, int rank, bool needMemFence) {
  if (threadIdx.x == 0) {
    if (needMemFence) {
      mbox.setFlagWithMemFence(rank, blockIdx.x);
    } else {
      mbox.setFlagNoMemFence(rank, blockIdx.x);
    }
  }
}

__global__ void
waitDeviceMboxFlag(DeviceMailbox mbox, int rank, bool needMemFence) {
  if (threadIdx.x == 0) {
    if (needMemFence) {
      mbox.waitFlagWithMemFence(rank, blockIdx.x);
    } else {
      mbox.waitFlagNoMemFence(rank, blockIdx.x);
    }
  }
}

class DeviceMailboxTest : public RcclxBaseTestFixture,
                          public ::testing::WithParamInterface<bool> {
 public:
  void SetUp() override {
    RcclxBaseTestFixture::SetUp();
    needMemFence = GetParam();
    LOG(INFO) << "DeviceMailboxTest with needMemFence = " << needMemFence;
    ASSERT_EQ(numRanks, 8);
    CUDA_CHECK(cudaSetDevice(localRank));

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    NCCL_CHECK(
        ncclCommInitRankConfig(&comm, numRanks, commId, localRank, &config));
    XLOGF(INFO, "rank {} init done; total ranks: {}", localRank, numRanks);
  }

  bool needMemFence;
  ncclComm_t comm{nullptr};
  const int nBlocks = 8;
  const int nThreads = 128;
};

INSTANTIATE_TEST_SUITE_P(
    DeviceMailboxTests,
    DeviceMailboxTest,
    // needMemFence = true, false
    ::testing::Values(true, false));

TEST_P(DeviceMailboxTest, testFlagSetAndRead) {
  auto [deviceBuf, selfMbox] = DeviceMailbox::mallocAndInit(numRanks, nBlocks);
  auto bootstrap = std::make_shared<TestBaselineBootstrap>(comm);
  IpcMemHandler handler(bootstrap, localRank, numRanks);
  handler.addSelfDeviceMemPtr(deviceBuf->get());
  handler.exchangeMemPtrs();

  // Rank 0 will set flag in the mailbox of all other peer ranks.
  // Non-0 ranks will wait to see the flag set by rank 0, then sleep for N
  // ms, then set the flag in the mailbox of rank 0. Thus, it should take rank 0
  // at least N ms to see the flag set by other ranks.
  std::chrono::milliseconds sleepMs{2000};
  if (localRank == 0) {
    folly::stop_watch<std::chrono::milliseconds> timer;
    for (int peer = 1; peer < numRanks; ++peer) {
      auto peerMbox =
          DeviceMailbox(numRanks, nBlocks, handler.getPeerDeviceMemPtr(peer));
      setDeviceMboxFlag<<<nBlocks, nThreads>>>(
          peerMbox, localRank, needMemFence);
    }
    for (int peer = 1; peer < numRanks; ++peer) {
      waitDeviceMboxFlag<<<nBlocks, nThreads>>>(selfMbox, peer, needMemFence);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_GE(timer.elapsed().count(), sleepMs.count());
  } else {
    // wait for flag set by rank 0
    waitDeviceMboxFlag<<<nBlocks, nThreads>>>(selfMbox, 0, needMemFence);
    std::this_thread::sleep_for(sleepMs);
    // set the flag of rank 0's mailbox
    auto mbox0 =
        DeviceMailbox(numRanks, nBlocks, handler.getPeerDeviceMemPtr(0));
    setDeviceMboxFlag<<<nBlocks, nThreads>>>(mbox0, localRank, needMemFence);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

TEST_P(DeviceMailboxTest, stressTest) {
  auto [deviceBuf, selfMbox] = DeviceMailbox::mallocAndInit(numRanks, nBlocks);
  auto bootstrap = std::make_shared<TestBaselineBootstrap>(comm);
  IpcMemHandler handler(bootstrap, localRank, numRanks);
  handler.addSelfDeviceMemPtr(deviceBuf->get());
  handler.exchangeMemPtrs();
  std::vector<DeviceMailbox> peerMailboxes;
  for (int i = 0; i < numRanks; ++i) {
    if (i == localRank) {
      continue;
    }
    peerMailboxes.emplace_back(
        DeviceMailbox(numRanks, nBlocks, handler.getPeerDeviceMemPtr(i)));
  }

  for (int i = 1; i <= 1000; ++i) {
    for (auto& peerMbox : peerMailboxes) {
      setDeviceMboxFlag<<<nBlocks, nThreads>>>(
          peerMbox, localRank, needMemFence);
    }
    for (int peer = 0; peer < numRanks; ++peer) {
      if (peer == localRank) {
        continue;
      }
      waitDeviceMboxFlag<<<nBlocks, nThreads>>>(selfMbox, peer, needMemFence);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

class IpcGpuBarrierTest : public RcclxBaseTestFixture {
 public:
  void SetUp() override {
    RcclxBaseTestFixture::SetUp();

    ASSERT_EQ(numRanks, 8);
    CUDA_CHECK(cudaSetDevice(localRank));

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    NCCL_CHECK(
        ncclCommInitRankConfig(&comm, numRanks, commId, localRank, &config));
    XLOGF(INFO, "rank {} init done; total ranks: {}", localRank, numRanks);
  }

  bool needMemFence;
  ncclComm_t comm{nullptr};
  const int nBlocks = 128;
  const int nThreads = 256;
};

__global__ void testBarrierKernel(IpcGpuBarrier barrier) {
  barrier.syncOnSameBlockIdx<
      false /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();
  // usually in prod kernel we have some mem access here
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();
  // usually in prod kernel we have some mem access here
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

TEST_F(IpcGpuBarrierTest, syncOnSameBlockIdx) {
  auto bootstrap = std::make_shared<TestBaselineBootstrap>(comm);
  auto [barrierResources, barrier] =
      IpcGpuBarrier::mallocAndInit(numRanks, nBlocks, localRank, bootstrap);

  int slowRank = 0;
  int sleepMs = 5000;
  // the slow rank will launch the kernel late
  if (localRank == slowRank) {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
  }

  folly::stop_watch<std::chrono::milliseconds> timer;
  testBarrierKernel<<<nBlocks, nThreads>>>(barrier);
  cudaDeviceSynchronize();
  auto kernelMs = timer.elapsed().count();

  // The non-slow rank's kernel will wait for the slow rank at the barrier,
  // so it should spend more time in the kernel.
  // We use 0.1 and 0.9 to soften the threadhold because the processes on
  // different ranks may start/execute at different time
  if (localRank == slowRank) {
    EXPECT_LE(kernelMs, sleepMs * 0.1);
  } else {
    EXPECT_GT(kernelMs, sleepMs * 0.9);
  }
}

TEST_F(IpcGpuBarrierTest, stressTest) {
  auto bootstrap = std::make_shared<TestBaselineBootstrap>(comm);
  auto [barrierResources, barrier] =
      IpcGpuBarrier::mallocAndInit(numRanks, nBlocks, localRank, bootstrap);

  for (int i = 0; i < 1000; ++i) {
    // pick a random rank to simulate up to 10ms slowess
    auto slowRank = folly::Random::rand32(numRanks);
    auto slownessMs = folly::Random::rand32(10);
    if (localRank == slowRank) {
      std::this_thread::sleep_for(std::chrono::milliseconds(slownessMs));
    }
    testBarrierKernel<<<nBlocks, nThreads>>>(barrier);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
