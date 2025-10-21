// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ctran/utils/CudaWrap.h"
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
    CHECK_EQ(
        cudaHostAlloc(&elem, sizeof(KernelElem), cudaHostAllocDefault),
        cudaSuccess);
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
 * Benchmark devSyncWaitNotify latency with varying number of groups
 */
static void BM_DevSyncWaitNotify_Latency(benchmark::State& state) {
  const int nGroups = state.range(0);
  const int cudaDev = 0;

  CHECK_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  // Allocate sync structure on device
  CtranAlgoDeviceSync* deviceSync;
  CHECK_EQ(cudaMalloc(&deviceSync, sizeof(CtranAlgoDeviceSync)), cudaSuccess);

  CudaBenchmarkResources resources;

  for (auto _ : state) {
    // Initialize sync structure with NOTIFY_SET for all groups
    CtranAlgoDeviceSync initSync;
    for (int i = 0; i < CTRAN_ALGO_MAX_THREAD_BLOCKS; i++) {
      initSync.syncs[i].stepOnSameBlockIdx = CTRAN_ALGO_NOTIFY_SET;
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
              0,
              resources.stream),
          cudaSuccess);
    }

    // Record stop event and measure time
    resources.stopTiming();
    float devDeltaMs = resources.measureTime();

    // SetIterationTime expects seconds
    state.SetIterationTime(devDeltaMs / 1000.0f);

    // Set custom counter for the benchmark
    state.counters["nGroups"] = nGroups;
  }

  // Cleanup
  CHECK_EQ(cudaFree(deviceSync), cudaSuccess);
}

/**
 * Benchmark KernelElem latency in putNotify with varying number of groups
 */
static void BM_KernelElemPutNotify_Latency(benchmark::State& state) {
  const int nGroups = state.range(0);
  const int cudaDev = 0;
  const int iters = 1;

  CHECK_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  KernelElemBenchmarkSetup kernelElem(nGroups);
  CudaBenchmarkResources resources;

  for (auto _ : state) {
    // Initialize status with INUSE for all groups
    kernelElem.initializeStatus(KernelElem::ElemStatus::INUSE);

    // Start timing the kernelElem kernel
    resources.startTiming();

    // Launch kernelElem kernel
    {
      std::array<void*, 3> kernArgs;
      kernArgs[0] = &kernelElem.elem;
      kernArgs[1] = (void*)&nGroups;
      kernArgs[2] = (void*)&iters;
      dim3 grid = {(unsigned int)nGroups, 1, 1};
      dim3 blocks = {1024, 1, 1};

      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)KernelElemPutNotifyKernel,
              grid,
              blocks,
              kernArgs.data(),
              0,
              resources.stream),
          cudaSuccess);

      for (int i = 0; i < iters; i++) {
        kernelElem.elem->post();
        // wait for kernel to finish
        while (!kernelElem.elem->isComplete()) {
        }
      }
    }

    // Record stop event and measure time
    resources.stopTiming();
    float devDeltaMs = resources.measureTime();

    // SetIterationTime expects seconds
    state.SetIterationTime(devDeltaMs / iters / 1000.0f);

    // Set custom counter for the benchmark
    state.counters["nGroups"] = nGroups;
  }
}

/**
 * Benchmark KernelElem latency in waitNotify with varying number of threads and
 * multiple iterations
 */
static void BM_KernelElemWaitNotify_Latency(benchmark::State& state) {
  const int nGroups = 1;
  const int nThreads = state.range(0);
  const int cudaDev = 0;
  const int iters = 50; // Add multiple iterations like PutNotify

  CHECK_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  KernelElemBenchmarkSetup kernelElem(nGroups);
  CudaBenchmarkResources resources;

  for (auto _ : state) {
    // Initialize status with INUSE for all groups
    kernelElem.initializeStatus(KernelElem::ElemStatus::INUSE);
    // Start timing the kernelElem waitnotify kernel
    resources.startTiming();

    // Launch kernelElem waitnotify kernel
    {
      std::array<void*, 2> kernArgs;
      kernArgs[0] = &kernelElem.elem;
      kernArgs[1] = (void*)&iters;
      dim3 grid = {(unsigned int)nGroups, 1, 1};
      dim3 blocks = {(unsigned int)nThreads, 1, 1};

      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)KernelElemWaitNotifyKernel,
              grid,
              blocks,
              kernArgs.data(),
              0,
              resources.stream),
          cudaSuccess);
      for (int i = 0; i < iters; i++) {
        kernelElem.elem->post();
        // wait for kernel to finish
        while (!kernelElem.elem->isComplete()) {
        }
      }
    }

    // Record stop event and measure time
    resources.stopTiming();
    float devDeltaMs = resources.measureTime();

    // SetIterationTime expects seconds - divide by iters for per-iteration time
    state.SetIterationTime(devDeltaMs / iters / 1000.0f);

    // Set custom counters for the benchmark
    state.counters["nThreads"] = nThreads;
  }
}

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// Test with different numbers of thread block groups
BENCHMARK(BM_DevSyncWaitNotify_Latency)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Test with different numbers of thread block groups
BENCHMARK(BM_KernelElemPutNotify_Latency)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Test with different numbers of threads per group
BENCHMARK(BM_KernelElemWaitNotify_Latency)
    ->RangeMultiplier(2)
    ->Range(64, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

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

  // Initialize and run benchmark
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Cleanup
  cudaSetDevice(0);
  cudaDeviceReset();

  return 0;
}
