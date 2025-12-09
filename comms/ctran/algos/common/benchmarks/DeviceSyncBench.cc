// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/testinfra/TestXPlatUtils.h"

using namespace ctran;
using ctran::utils::CtranIpcDesc;
using ctran::utils::CtranIpcMem;
using ctran::utils::CtranIpcRemMem;

//------------------------------------------------------------------------------
// External Kernel Declaration
//------------------------------------------------------------------------------

extern __global__ void devSyncWaitNotifyKernel(
    CtranAlgoDeviceSync* localSync,
    int nGroups);

extern __global__ void
devSyncOnStepsKernel(CtranAlgoDeviceSync* sync, bool isProducer, int nSteps);
//------------------------------------------------------------------------------
// Common Helper Functions
//------------------------------------------------------------------------------

namespace {

class DeviceSyncBenchSetup : public CudaBenchBase {
 public:
  CtranIpcDesc ipcDesc;
  DeviceSyncBenchSetup(int cudaDev) : cudaDev_(cudaDev) {
    CHECK_EQ(cudaSetDevice(cudaDev_), cudaSuccess);
    ipcMem_ = std::make_unique<CtranIpcMem>(
        sizeof(CtranAlgoDeviceSync), cudaDev_, &dummyLogMetaData_, "Benchmark");
    CHECK_EQ(ipcMem_->ipcExport(ipcDesc), commSuccess);
  }

  void* localDeviceSyncPtr() const {
    return ipcMem_->getBase();
  }

  void* importRemoteDeviceSyncPtr(const CtranIpcDesc& remoteIpcDesc) {
    CHECK_EQ(cudaSetDevice(cudaDev_), cudaSuccess);
    ipcRemMem_ = std::make_unique<CtranIpcRemMem>(
        remoteIpcDesc, cudaDev_, &dummyLogMetaData_, "Benchmark");
    return ipcRemMem_->getBase();
  }

  ~DeviceSyncBenchSetup() {
    CHECK_EQ(cudaSetDevice(cudaDev_), cudaSuccess);
    if (ipcRemMem_) {
      CHECK_EQ(ipcRemMem_->release(), commSuccess);
    }
    CHECK_EQ(ipcMem_->free(), commSuccess);
  }

  void initializeSyncs(int step) {
    CHECK_EQ(cudaSetDevice(cudaDev_), cudaSuccess);
    CtranAlgoDeviceSync initSync;
    for (int j = 0; j < CTRAN_ALGO_MAX_THREAD_BLOCKS; j++) {
      initSync.syncs[j].stepOnSameBlockIdx = step;
    }
    CHECK_EQ(
        cudaMemcpyAsync(
            localDeviceSyncPtr(),
            &initSync,
            sizeof(CtranAlgoDeviceSync),
            cudaMemcpyHostToDevice,
            stream),
        cudaSuccess);
    CHECK_EQ(cudaStreamSynchronize(stream), cudaSuccess);
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
 * Benchmark devSyncWaitNotify with varying number of groups
 */
static void
DevSyncWaitNotify(uint32_t iters, int nGroups, folly::UserCounters& counters) {
  const int cudaDev = 0;
  CHECK_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  DeviceSyncBenchSetup bench(cudaDev);
  auto deviceSync = bench.localDeviceSyncPtr();
  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Initialize sync structure with NOTIFY_SET for all groups
    bench.initializeSyncs(CTRAN_ALGO_NOTIFY_SET);
    // Start timing the receiver kernel
    bench.startTiming();

    // Launch receiver kernel (this will wait for notification)
    {
      std::array<void*, 2> kernArgs;
      kernArgs[0] = &deviceSync;
      kernArgs[1] = (void*)&nGroups;
      dim3 grid = {1, 1, 1};
      dim3 blocks = {256, 1, 1};
      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)devSyncWaitNotifyKernel,
              grid,
              blocks,
              kernArgs.data(),
              sizeof(CtranAlgoDeviceState), // Dynamic shared memory size
              bench.stream),
          cudaSuccess);
    }

    // Stop timing and measure
    bench.stopTiming();
    totalTimeMs += bench.measureTime();
  }

  float avgTimeUs = (totalTimeMs / iters) * 1000.0f; // Convert ms to us
  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["nGroups"] =
      folly::UserMetric(nGroups, folly::UserMetric::Type::METRIC);
}

/**
 * Benchmark devSyncSetStepKernel/devSyncWaitStepKernel with varying number of
 * groups
 */
static void MultiBlockDevSyncOnSteps(
    uint32_t iters,
    int nGroups,
    folly::UserCounters& counters) {
  const int nSteps = 100;

  const int consumerCudaDev = 1;
  CHECK_EQ(cudaSetDevice(consumerCudaDev), cudaSuccess);
  DeviceSyncBenchSetup consumerBench(consumerCudaDev);
  auto deviceSync = consumerBench.localDeviceSyncPtr();
  // Initialize sync structure with reset for all groups
  consumerBench.initializeSyncs(CTRAN_ALGO_STEP_RESET);

  const int producerCudaDev = 0;
  CHECK_EQ(cudaSetDevice(producerCudaDev), cudaSuccess);
  DeviceSyncBenchSetup producerBench(producerCudaDev);
  auto remDeviceSync =
      producerBench.importRemoteDeviceSyncPtr(consumerBench.ipcDesc);

  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Start timing the kernel
    CHECK_EQ(cudaSetDevice(consumerCudaDev), cudaSuccess);
    consumerBench.startTiming();

    {
      bool isProducer = false;
      void* kernArgs[3] = {
          (void*)&deviceSync, (void*)&isProducer, (void*)&nSteps};
      dim3 grid = {(unsigned int)nGroups, 1, 1};
      dim3 blocks = {256, 1, 1};
      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)devSyncOnStepsKernel,
              grid,
              blocks,
              kernArgs,
              sizeof(CtranAlgoDeviceState), // Dynamic shared memory size
              consumerBench.stream),
          cudaSuccess);

      CHECK_EQ(cudaSetDevice(producerCudaDev), cudaSuccess);
      isProducer = true;
      kernArgs[0] = (void*)&remDeviceSync;
      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)devSyncOnStepsKernel,
              grid,
              blocks,
              kernArgs,
              sizeof(CtranAlgoDeviceState), // Dynamic shared memory size
              producerBench.stream),
          cudaSuccess);
    }

    // Stop timing and measure
    CHECK_EQ(cudaSetDevice(consumerCudaDev), cudaSuccess);
    consumerBench.stopTiming();
    totalTimeMs += consumerBench.measureTime();
  }

  float avgTimeUs =
      (totalTimeMs / iters / nSteps) * 1000.0f; // Convert ms to us
  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["nGroups"] =
      folly::UserMetric(nGroups, folly::UserMetric::Type::METRIC);
}

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// Test with different numbers of thread block groups
BENCHMARK_SINGLE_PARAM_COUNTERS(DevSyncWaitNotify, 1);
BENCHMARK_SINGLE_PARAM_COUNTERS(DevSyncWaitNotify, 2);
BENCHMARK_SINGLE_PARAM_COUNTERS(DevSyncWaitNotify, 4);
BENCHMARK_SINGLE_PARAM_COUNTERS(DevSyncWaitNotify, 8);
BENCHMARK_SINGLE_PARAM_COUNTERS(DevSyncWaitNotify, 16);
BENCHMARK_SINGLE_PARAM_COUNTERS(DevSyncWaitNotify, 32);
BENCHMARK_SINGLE_PARAM_COUNTERS(DevSyncWaitNotify, 64);

BENCHMARK_SINGLE_PARAM_COUNTERS(MultiBlockDevSyncOnSteps, 1);
BENCHMARK_SINGLE_PARAM_COUNTERS(MultiBlockDevSyncOnSteps, 2);
BENCHMARK_SINGLE_PARAM_COUNTERS(MultiBlockDevSyncOnSteps, 4);
BENCHMARK_SINGLE_PARAM_COUNTERS(MultiBlockDevSyncOnSteps, 8);
BENCHMARK_SINGLE_PARAM_COUNTERS(MultiBlockDevSyncOnSteps, 16);
BENCHMARK_SINGLE_PARAM_COUNTERS(MultiBlockDevSyncOnSteps, 32);
BENCHMARK_SINGLE_PARAM_COUNTERS(MultiBlockDevSyncOnSteps, 64);

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
