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

extern __global__ void
KernelElemPutNotifyKernel(KernelElem* elem, int nGroups, int iters);

extern __global__ void KernelElemWaitNotifyKernel(KernelElem* elem, int iters);

//------------------------------------------------------------------------------
// Common Helper Functions
//------------------------------------------------------------------------------

namespace {

class KernelElemBenchSetup : public CudaBenchBase {
 public:
  KernelElem* elem;
  int nGroups;

  KernelElemBenchSetup(int nGroups) : nGroups(nGroups) {
    CUDACHECK_TEST(
        cudaHostAlloc(&elem, sizeof(KernelElem), cudaHostAllocDefault));
    elem->ngroups = nGroups;
  }

  ~KernelElemBenchSetup() {
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
 * Benchmark KernelElem in putNotify with varying number of groups
 */
static void KernelElemPutNotify(
    uint32_t iters,
    int nGroups,
    folly::UserCounters& counters) {
  const int cudaDev = 0;
  const int innerIters = 50;

  CHECK_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  KernelElemBenchSetup bench(nGroups);
  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Initialize status with INUSE for all groups
    bench.initializeStatus(KernelElem::ElemStatus::INUSE);

    // Start timing the kernelElem kernel
    bench.startTiming();

    {
      std::array<void*, 3> kernArgs;
      kernArgs[0] = &bench.elem;
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
              bench.stream),
          cudaSuccess);
      for (int j = 0; j < innerIters; j++) {
        bench.elem->post();
        while (!bench.elem->isComplete()) {
        }
      }
    }

    // Stop timing and measure
    bench.stopTiming();
    totalTimeMs += bench.measureTime();
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
  KernelElemBenchSetup bench(nGroups);
  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    bench.initializeStatus(KernelElem::ElemStatus::INUSE);

    // Start timing the kernel
    bench.startTiming();

    {
      std::array<void*, 2> kernArgs;
      kernArgs[0] = &bench.elem;
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
              bench.stream),
          cudaSuccess);
      for (int j = 0; j < innerIters; j++) {
        bench.elem->post();
        while (!bench.elem->isComplete()) {
        }
      }
    }

    // Stop timing and measure
    bench.stopTiming();
    totalTimeMs += bench.measureTime();
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

// Test with different numbers of thread block groups
BENCHMARK_SINGLE_PARAM_COUNTERS(KernelElemPutNotify, 1);
BENCHMARK_SINGLE_PARAM_COUNTERS(KernelElemPutNotify, 2);
BENCHMARK_SINGLE_PARAM_COUNTERS(KernelElemPutNotify, 4);
BENCHMARK_SINGLE_PARAM_COUNTERS(KernelElemPutNotify, 8);
BENCHMARK_SINGLE_PARAM_COUNTERS(KernelElemPutNotify, 16);
BENCHMARK_SINGLE_PARAM_COUNTERS(KernelElemPutNotify, 32);
BENCHMARK_SINGLE_PARAM_COUNTERS(KernelElemPutNotify, 64);

// Test with different numbers of threads per group
BENCHMARK_SINGLE_PARAM_COUNTERS(KernelElemWaitNotify, 256);
BENCHMARK_SINGLE_PARAM_COUNTERS(KernelElemWaitNotify, 512);
BENCHMARK_SINGLE_PARAM_COUNTERS(KernelElemWaitNotify, 1024);

int main(int argc, char** argv) {
  CHECK_GE(bench_utils::getNumCudaDevices(), 1);

  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  // Cleanup
  cudaSetDevice(0);
  cudaDeviceReset();

  return 0;
}
