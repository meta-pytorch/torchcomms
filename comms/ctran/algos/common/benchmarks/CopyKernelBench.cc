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

//------------------------------------------------------------------------------
// External Kernel Declaration
//------------------------------------------------------------------------------

template <typename T>
__global__ void
copyKernel(const T* sendbuff, T* recvbuff, size_t count, int nRuns);

//------------------------------------------------------------------------------
// Benchmark Functions
//------------------------------------------------------------------------------

/**
 * Benchmark D2D copyKernel (copy from and to the same device) with varying
 * message size and number of groups
 */
static void d2dCopyKernel(
    uint32_t iters,
    size_t nBytes,
    int nGroups,
    folly::UserCounters& counters) {
  const int nRunsPerIter = 50;

  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CudaBenchBase bench;
  using T = uint8_t;
  const size_t count = nBytes / sizeof(T);
  void* srcPtr = nullptr;
  void* dstPtr = nullptr;
  CHECK_EQ(cudaMalloc(&srcPtr, nBytes), cudaSuccess);
  CHECK_EQ(cudaMalloc(&dstPtr, nBytes), cudaSuccess);
  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Start timing the kernel
    bench.startTiming();

    {
      void* kernArgs[4] = {
          (void*)&srcPtr, (void*)&dstPtr, (void*)&count, (void*)&nRunsPerIter};
      dim3 grid = {(unsigned int)nGroups, 1, 1};
      dim3 blocks = {256, 1, 1};
      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)copyKernel<T>,
              grid,
              blocks,
              kernArgs,
              sizeof(CtranAlgoDeviceState), // Dynamic shared memory size
              bench.stream),
          cudaSuccess);
    }

    // Stop timing and measure
    bench.stopTiming();
    totalTimeMs += bench.measureTime();
  }

  float avgTimeUs =
      (totalTimeMs / iters / nRunsPerIter) * 1000.0f; // Convert ms to us
  float busBwGBps = (nBytes / (float)(1 << 30)) /
      (avgTimeUs / 1e6f); // GB/s = bytes / time_in_seconds
  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["busBwGBps"] =
      folly::UserMetric(busBwGBps, folly::UserMetric::Type::METRIC);
  counters["nGroups"] =
      folly::UserMetric(nGroups, folly::UserMetric::Type::METRIC);

  CHECK_EQ(cudaFree(srcPtr), cudaSuccess);
  CHECK_EQ(cudaFree(dstPtr), cudaSuccess);
}

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// Test with different message sizes and thread block groups
// Format: BENCHMARK_MULTI_PARAM_COUNTERS(function, name, msgSize, nGroups)

// 4MB message size with varying nGroups
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 4MB_1g, 4 * 1024 * 1024, 1);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 4MB_2g, 4 * 1024 * 1024, 2);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 4MB_4g, 4 * 1024 * 1024, 4);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 4MB_8g, 4 * 1024 * 1024, 8);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 4MB_16g, 4 * 1024 * 1024, 16);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 4MB_32g, 4 * 1024 * 1024, 32);

// 8MB message size with varying nGroups
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 8MB_1g, 8 * 1024 * 1024, 1);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 8MB_2g, 8 * 1024 * 1024, 2);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 8MB_4g, 8 * 1024 * 1024, 4);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 8MB_8g, 8 * 1024 * 1024, 8);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 8MB_16g, 8 * 1024 * 1024, 16);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 8MB_32g, 8 * 1024 * 1024, 32);

// 16MB message size with varying nGroups
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 16MB_1g, 16 * 1024 * 1024, 1);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 16MB_2g, 16 * 1024 * 1024, 2);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 16MB_4g, 16 * 1024 * 1024, 4);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 16MB_8g, 16 * 1024 * 1024, 8);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 16MB_16g, 16 * 1024 * 1024, 16);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 16MB_32g, 16 * 1024 * 1024, 32);

// 32MB message size with varying nGroups
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 32MB_1g, 32 * 1024 * 1024, 1);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 32MB_2g, 32 * 1024 * 1024, 2);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 32MB_4g, 32 * 1024 * 1024, 4);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 32MB_8g, 32 * 1024 * 1024, 8);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 32MB_16g, 32 * 1024 * 1024, 16);
BENCHMARK_MULTI_PARAM_COUNTERS(d2dCopyKernel, 32MB_32g, 32 * 1024 * 1024, 32);

int main(int argc, char** argv) {
  CHECK_GE(bench_utils::getNumCudaDevices(), 2);

  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  // Cleanup
  cudaSetDevice(0);
  cudaDeviceReset();
  cudaSetDevice(1);
  cudaDeviceReset();

  return 0;
}
