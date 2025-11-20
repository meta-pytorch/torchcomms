// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/testinfra/TestXPlatUtils.h"

using namespace ctran;

//------------------------------------------------------------------------------
// External Kernel Declaration
//------------------------------------------------------------------------------

extern __global__ void devSyncWaitNotifyKernel(
    CtranAlgoDeviceSync* localSync,
    int nGroups);

//------------------------------------------------------------------------------
// Common Helper Functions
//------------------------------------------------------------------------------

namespace {

class DeviceSyncBenchSetup : public CudaBenchBase {
 public:
  CtranAlgoDeviceSync* deviceSync;

  DeviceSyncBenchSetup() {
    CUDACHECK_TEST(cudaMalloc(&deviceSync, sizeof(CtranAlgoDeviceSync)));
  }

  ~DeviceSyncBenchSetup() {
    CHECK_EQ(cudaFree(deviceSync), cudaSuccess);
  }

  void initializeSyncs() {
    CtranAlgoDeviceSync initSync;
    for (int j = 0; j < CTRAN_ALGO_MAX_THREAD_BLOCKS; j++) {
      initSync.syncs[j].stepOnSameBlockIdx = CTRAN_ALGO_NOTIFY_SET;
    }
    CHECK_EQ(
        cudaMemcpyAsync(
            deviceSync,
            &initSync,
            sizeof(CtranAlgoDeviceSync),
            cudaMemcpyHostToDevice,
            stream),
        cudaSuccess);
    CHECK_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }
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

  DeviceSyncBenchSetup bench;
  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Initialize sync structure with NOTIFY_SET for all groups
    bench.initializeSyncs();
    // Start timing the receiver kernel
    bench.startTiming();

    // Launch receiver kernel (this will wait for notification)
    {
      std::array<void*, 2> kernArgs;
      kernArgs[0] = &bench.deviceSync;
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

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// Test with different numbers of thread block groups
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 1);
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 2);
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 4);
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 8);
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 16);
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 32);
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 64);

int main(int argc, char** argv) {
  CHECK_GE(bench_utils::getNumCudaDevices(), 1);

  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  // Cleanup
  cudaSetDevice(0);
  cudaDeviceReset();

  return 0;
}
