// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "comms/ctran/algos/common/SpscP2pSync.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/testinfra/TestXPlatUtils.h"

using namespace ctran;
using ctran::algos::SpscP2pSync;
using ctran::utils::CtranIpcDesc;
using ctran::utils::CtranIpcMem;
using ctran::utils::CtranIpcRemMem;

//------------------------------------------------------------------------------
// External Kernel Declaration
//------------------------------------------------------------------------------

__global__ void
SpscP2pSyncBenchKernel(int myLocalRank, int numIter, SpscP2pSync* shmSync);

//------------------------------------------------------------------------------
// Common Helper Functions
//------------------------------------------------------------------------------

namespace {

class SpscP2pSyncBenchSetup : public CudaBenchBase {
 public:
  CtranIpcDesc ipcDesc;
  SpscP2pSyncBenchSetup(int cudaDev, size_t nBytes = 0) : cudaDev_(cudaDev) {
    if (nBytes == 0) {
      return;
    }
    CUDACHECK_TEST(cudaSetDevice(cudaDev_));
    ipcMem_ = std::make_unique<CtranIpcMem>(
        nBytes, cudaDev_, &dummyLogMetaData_, "Benchmark");
    CHECK_EQ(ipcMem_->ipcExport(ipcDesc), commSuccess);
  }

  void* localPtr() const {
    return ipcMem_->getBase();
  }

  void* importRemotePtr(const CtranIpcDesc& remoteIpcDesc) {
    CUDACHECK_TEST(cudaSetDevice(cudaDev_));
    ipcRemMem_ = std::make_unique<CtranIpcRemMem>(
        remoteIpcDesc, cudaDev_, &dummyLogMetaData_, "Benchmark");
    return ipcRemMem_->getBase();
  }

  ~SpscP2pSyncBenchSetup() {
    CUDACHECK_TEST(cudaSetDevice(cudaDev_));
    if (ipcRemMem_) {
      CHECK_EQ(ipcRemMem_->release(), commSuccess);
    }
    if (ipcMem_) {
      CHECK_EQ(ipcMem_->free(), commSuccess);
    }
  }

  void initializeSync() {
    CUDACHECK_TEST(cudaSetDevice(cudaDev_));
    SpscP2pSync initSync = SpscP2pSync();
    CUDACHECK_TEST(cudaMemcpyAsync(
        localPtr(),
        &initSync,
        sizeof(SpscP2pSync),
        cudaMemcpyHostToDevice,
        stream));
    CUDACHECK_TEST(cudaStreamSynchronize(stream));
  }

 private:
  int cudaDev_;
  std::unique_ptr<CtranIpcMem> ipcMem_;
  std::unique_ptr<CtranIpcRemMem> ipcRemMem_;
  const struct CommLogData dummyLogMetaData_ = {
      0,
      0xfaceb00c12345678 /*Dummy placeholder value for commHash*/,
      "BenchComm",
      0,
      0};
};

} // anonymous namespace

//------------------------------------------------------------------------------
// Benchmark Functions
//------------------------------------------------------------------------------

/**
 * Benchmark SpscP2pSync producer-consumer synchronization with varying number
 * of iterations per run.
 *
 * This benchmark measures the performance of single-producer single-consumer
 * synchronization primitives across two GPUs using IPC shared memory.
 * - GPU 0 (producer): posts sync notifications
 * - GPU 1 (consumer): waits for and completes sync notifications
 */
static void SpscP2pSyncPerf(
    uint32_t iters,
    int numIterPerRun,
    folly::UserCounters& counters) {
  const int consumerCudaDev = 1;
  CUDACHECK_TEST(cudaSetDevice(consumerCudaDev));
  SpscP2pSyncBenchSetup consumerBench(consumerCudaDev, sizeof(SpscP2pSync));

  const int producerCudaDev = 0;
  CUDACHECK_TEST(cudaSetDevice(producerCudaDev));
  SpscP2pSyncBenchSetup producerBench(producerCudaDev);
  auto syncPtr = producerBench.importRemotePtr(consumerBench.ipcDesc);

  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Initialize sync structure on consumer
    consumerBench.initializeSync();

    // Start timing from consumer perspective
    CUDACHECK_TEST(cudaSetDevice(consumerCudaDev));
    consumerBench.startTiming();

    // Launch consumer kernel (rank 1)
    {
      int myLocalRank = 1;
      auto localSyncPtr = consumerBench.localPtr();
      void* kernArgs[3] = {
          (void*)&myLocalRank, (void*)&numIterPerRun, (void*)&localSyncPtr};
      dim3 grid = {1, 1, 1};
      dim3 blocks = {256, 1, 1};
      CUDACHECK_TEST(cudaLaunchKernel(
          (const void*)SpscP2pSyncBenchKernel,
          grid,
          blocks,
          kernArgs,
          0,
          consumerBench.stream));
    }

    // Launch producer kernel (rank 0)
    CUDACHECK_TEST(cudaSetDevice(producerCudaDev));
    {
      int myLocalRank = 0;
      void* kernArgs[3] = {
          (void*)&myLocalRank, (void*)&numIterPerRun, (void*)&syncPtr};
      dim3 grid = {1, 1, 1};
      dim3 blocks = {256, 1, 1};
      CUDACHECK_TEST(cudaLaunchKernel(
          (const void*)SpscP2pSyncBenchKernel,
          grid,
          blocks,
          kernArgs,
          0,
          producerBench.stream));
    }

    // Stop timing and measure
    CUDACHECK_TEST(cudaSetDevice(consumerCudaDev));
    consumerBench.stopTiming();
    totalTimeMs += consumerBench.measureTime();
  }

  float avgTimeUs =
      (totalTimeMs / iters / numIterPerRun) * 1000.0f; // Convert ms to us
  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["numIterPerRun"] =
      folly::UserMetric(numIterPerRun, folly::UserMetric::Type::METRIC);
}

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

BENCHMARK_SINGLE_PARAM_COUNTERS(SpscP2pSyncPerf, 1000);

int main(int argc, char** argv) {
  CHECK_GE(bench_utils::getNumCudaDevices(), 2);

  // Initialize CUDA driver library to load cuMem* functions for IPC support
  CHECK_EQ(ctran::utils::commCudaLibraryInit(), commSuccess);
  CHECK_EQ(ctran::utils::CtranIpcSupport(), true);

  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  // Cleanup
  cudaSetDevice(0);
  cudaDeviceReset();
  cudaSetDevice(1);
  cudaDeviceReset();

  return 0;
}
