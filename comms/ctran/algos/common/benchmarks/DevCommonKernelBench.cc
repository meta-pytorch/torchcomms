// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <memory>
#include <thread>
#include <vector>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/logger/Logger.h"

using namespace ctran;

//------------------------------------------------------------------------------
// External Kernel Declaration
//------------------------------------------------------------------------------

extern __global__ void devSyncWaitNotifyKernel(
    CtranAlgoDeviceSync* localSync,
    int nGroups);

extern __global__ void
KernelElemPutNotifyKernel(KernelElem* elem, int nGroups, int iters);

extern __global__ void KernelElemWaitNotifyKernel(KernelElem* elem, int iters);

//------------------------------------------------------------------------------
// Common Helper Functions
//------------------------------------------------------------------------------

namespace {

struct CudaBenchmarkResources {
  cudaStream_t stream;
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;

  CudaBenchmarkResources() {
    CHECK_EQ(cudaStreamCreate(&stream), cudaSuccess);
    CHECK_EQ(cudaEventCreate(&startEvent), cudaSuccess);
    CHECK_EQ(cudaEventCreate(&stopEvent), cudaSuccess);
  }

  ~CudaBenchmarkResources() {
    CHECK_EQ(cudaStreamDestroy(stream), cudaSuccess);
    CHECK_EQ(cudaEventDestroy(startEvent), cudaSuccess);
    CHECK_EQ(cudaEventDestroy(stopEvent), cudaSuccess);
  }

  float measureTime() {
    CHECK_EQ(cudaEventSynchronize(stopEvent), cudaSuccess);
    float deltaMs;
    CHECK_EQ(
        cudaEventElapsedTime(&deltaMs, startEvent, stopEvent), cudaSuccess);
    return deltaMs;
  }

  void startTiming() {
    CHECK_EQ(cudaEventRecord(startEvent, stream), cudaSuccess);
  }

  void stopTiming() {
    CHECK_EQ(cudaEventRecord(stopEvent, stream), cudaSuccess);
  }
};

struct KernelElemBenchmarkSetup {
  KernelElem* elem;
  int nGroups;

  KernelElemBenchmarkSetup(int nGroups) : nGroups(nGroups) {
    CUDACHECK_TEST(
        cudaHostAlloc(&elem, sizeof(KernelElem), cudaHostAllocDefault));
    elem->ngroups = nGroups;
  }

  ~KernelElemBenchmarkSetup() {
    CHECK_EQ(cudaFreeHost(elem), cudaSuccess);
  }

  void initializeStatus(KernelElem::ElemStatus status) {
    for (int i = 0; i < nGroups; i++) {
      elem->status[i] = status;
    }
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

  // Allocate sync structure on device
  CtranAlgoDeviceSync* deviceSync;
  CHECK_EQ(cudaMalloc(&deviceSync, sizeof(CtranAlgoDeviceSync)), cudaSuccess);

  CudaBenchmarkResources resources;
  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Initialize sync structure with NOTIFY_SET for all groups
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
            resources.stream),
        cudaSuccess);
    CHECK_EQ(cudaStreamSynchronize(resources.stream), cudaSuccess);

    // Start timing the receiver kernel
    resources.startTiming();

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
              resources.stream),
          cudaSuccess);
    }

    // Stop timing and measure
    resources.stopTiming();
    totalTimeMs += resources.measureTime();
  }

  float avgTimeUs = (totalTimeMs / iters) * 1000.0f; // Convert ms to us
  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["nGroups"] =
      folly::UserMetric(nGroups, folly::UserMetric::Type::METRIC);

  CHECK_EQ(cudaFree(deviceSync), cudaSuccess);
}

/**
 * Benchmark KernelElem in putNotify with varying number of groups
 */
static void KernelElemPutNotify(
    uint32_t iters,
    int nGroups,
    folly::UserCounters& counters) {
  const int cudaDev = 0;
  const int innerIters = 50;

  CHECK_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  KernelElemBenchmarkSetup kernelElem(nGroups);
  CudaBenchmarkResources resources;
  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Initialize status with INUSE for all groups
    kernelElem.initializeStatus(KernelElem::ElemStatus::INUSE);

    // Start timing the kernelElem kernel
    resources.startTiming();

    {
      std::array<void*, 3> kernArgs;
      kernArgs[0] = &kernelElem.elem;
      kernArgs[1] = (void*)&nGroups;
      kernArgs[2] = (void*)&innerIters;
      dim3 grid = {(unsigned int)nGroups, 1, 1};
      dim3 blocks = {1024, 1, 1};

      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)KernelElemPutNotifyKernel,
              grid,
              blocks,
              kernArgs.data(),
              sizeof(CtranAlgoDeviceState), // Dynamic shared memory size
              resources.stream),
          cudaSuccess);
      for (int j = 0; j < innerIters; j++) {
        kernelElem.elem->post();
        while (!kernelElem.elem->isComplete()) {
        }
      }
    }

    // Stop timing and measure
    resources.stopTiming();
    totalTimeMs += resources.measureTime();
  }

  float avgTimeUs =
      (totalTimeMs / iters / innerIters) * 1000.0f; // Convert ms to us
  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["nGroups"] =
      folly::UserMetric(nGroups, folly::UserMetric::Type::METRIC);
}

/**
 * Benchmark KernelElem in waitNotify with varying number of threads and
 * multiple iterations
 */
static void KernelElemWaitNotify(
    uint32_t iters,
    int nThreads,
    folly::UserCounters& counters) {
  const int nGroups = 1;
  const int cudaDev = 0;
  CHECK_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  const int innerIters = 50;
  KernelElemBenchmarkSetup kernelElem(nGroups);
  CudaBenchmarkResources resources;
  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    kernelElem.initializeStatus(KernelElem::ElemStatus::INUSE);

    // Start timing the kernel
    resources.startTiming();

    {
      std::array<void*, 2> kernArgs;
      kernArgs[0] = &kernelElem.elem;
      kernArgs[1] = (void*)&innerIters;
      dim3 grid = {(unsigned int)nGroups, 1, 1};
      dim3 blocks = {(unsigned int)nThreads, 1, 1};

      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)KernelElemWaitNotifyKernel,
              grid,
              blocks,
              kernArgs.data(),
              sizeof(CtranAlgoDeviceState), // Dynamic shared memory size
              resources.stream),
          cudaSuccess);
      for (int j = 0; j < innerIters; j++) {
        kernelElem.elem->post();
        while (!kernelElem.elem->isComplete()) {
        }
      }
    }

    // Stop timing and measure
    resources.stopTiming();
    totalTimeMs += resources.measureTime();
  }

  float avgTimeUs =
      (totalTimeMs / iters / innerIters) * 1000.0f; // Convert ms to us
  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["nThreads"] =
      folly::UserMetric(nThreads, folly::UserMetric::Type::METRIC);
}

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

#define BENCHMARK_PARAM_COUNTERS(name, param)                     \
  BENCHMARK_IMPL_COUNTERS(                                        \
      FB_CONCATENATE(name, FB_CONCATENATE(_, param)),             \
      FOLLY_PP_STRINGIZE(name) "(" FOLLY_PP_STRINGIZE(param) ")", \
      counters,                                                   \
      iters,                                                      \
      unsigned,                                                   \
      iters) {                                                    \
    name(iters, param, counters);                                 \
  }

// Test with different numbers of thread block groups
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 1);
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 2);
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 4);
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 8);
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 16);
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 32);
BENCHMARK_PARAM_COUNTERS(DevSyncWaitNotify, 64);

// Test with different numbers of thread block groups
BENCHMARK_PARAM_COUNTERS(KernelElemPutNotify, 1);
BENCHMARK_PARAM_COUNTERS(KernelElemPutNotify, 2);
BENCHMARK_PARAM_COUNTERS(KernelElemPutNotify, 4);
BENCHMARK_PARAM_COUNTERS(KernelElemPutNotify, 8);
BENCHMARK_PARAM_COUNTERS(KernelElemPutNotify, 16);
BENCHMARK_PARAM_COUNTERS(KernelElemPutNotify, 32);
BENCHMARK_PARAM_COUNTERS(KernelElemPutNotify, 64);

// Test with different numbers of threads per group
BENCHMARK_PARAM_COUNTERS(KernelElemWaitNotify, 256);
BENCHMARK_PARAM_COUNTERS(KernelElemWaitNotify, 512);
BENCHMARK_PARAM_COUNTERS(KernelElemWaitNotify, 1024);

// Custom main function to handle initialization
int main(int argc, char** argv) {
  // Check if we have at least one CUDA device
  int deviceCount;
  if (cudaGetDeviceCount(&deviceCount) == cudaSuccess) {
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    if (deviceCount < 1) {
      std::cout << "Error: Need at least 1 CUDA device" << std::endl;
      return 1;
    }
  } else {
    std::cout << "Error: No CUDA devices found" << std::endl;
    return 1;
  }

  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  // Cleanup
  cudaSetDevice(0);
  cudaDeviceReset();

  return 0;
}
