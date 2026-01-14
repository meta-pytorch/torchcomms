// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <glog/logging.h>

#include "comms/pipes/benchmarks/SelfTransportBench.cuh"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------

// Number of warmup iterations to run before timing (primes GPU clocks, caches)
constexpr int kWarmupIters = 5;

//------------------------------------------------------------------------------
// Benchmark Functions
//------------------------------------------------------------------------------

/**
 * Benchmark P2pSelfTransportDevice::write() for local memory copies
 */
static void selfTransportWrite(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    folly::UserCounters& counters) {
  const int nRunsPerIter = 50;
  const int nThreads = 256;

  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CudaBenchBase bench;

  DeviceBuffer srcBuffer(nBytes);
  DeviceBuffer dstBuffer(nBytes);

  char* srcPtr = static_cast<char*>(srcBuffer.get());
  char* dstPtr = static_cast<char*>(dstBuffer.get());

  dim3 grid{static_cast<unsigned int>(nBlocks), 1, 1};
  dim3 blocks{static_cast<unsigned int>(nThreads), 1, 1};
  void* kernArgs[4] = {
      (void*)&dstPtr, (void*)&srcPtr, (void*)&nBytes, (void*)&nRunsPerIter};

  // Warmup iterations (not timed) - primes GPU clocks and caches
  for (int w = 0; w < kWarmupIters; ++w) {
    CHECK_EQ(
        cudaLaunchKernel(
            (const void*)selfTransportWriteKernel,
            grid,
            blocks,
            kernArgs,
            0,
            bench.stream),
        cudaSuccess);
  }
  CHECK_EQ(cudaStreamSynchronize(bench.stream), cudaSuccess);

  // Timed iterations
  bench.startTiming();
  for (uint32_t i = 0; i < iters; ++i) {
    CHECK_EQ(
        cudaLaunchKernel(
            (const void*)selfTransportWriteKernel,
            grid,
            blocks,
            kernArgs,
            0,
            bench.stream),
        cudaSuccess);
  }
  bench.stopTiming();
  float totalTimeMs = bench.measureTime();

  float avgTimeUs = (totalTimeMs / iters / nRunsPerIter) * 1000.0f;
  float busBwGBps = (nBytes / 1e9f) / (avgTimeUs / 1e6f);

  counters["latency (us)"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["bandwidth (GB/s)"] =
      folly::UserMetric(busBwGBps, folly::UserMetric::Type::METRIC);
}

/**
 * Benchmark cudaMemcpyAsync as a baseline for comparison
 */
static void cudaMemcpyBaseline(
    uint32_t iters,
    size_t nBytes,
    folly::UserCounters& counters) {
  const int nRunsPerIter = 50;

  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CudaBenchBase bench;

  DeviceBuffer srcBuffer(nBytes);
  DeviceBuffer dstBuffer(nBytes);

  char* srcPtr = static_cast<char*>(srcBuffer.get());
  char* dstPtr = static_cast<char*>(dstBuffer.get());

  // Warmup iterations (not timed) - primes GPU clocks and caches
  for (int w = 0; w < kWarmupIters; ++w) {
    for (int run = 0; run < nRunsPerIter; ++run) {
      CHECK_EQ(
          cudaMemcpyAsync(
              dstPtr, srcPtr, nBytes, cudaMemcpyDeviceToDevice, bench.stream),
          cudaSuccess);
    }
  }
  CHECK_EQ(cudaStreamSynchronize(bench.stream), cudaSuccess);

  // Timed iterations
  bench.startTiming();
  for (uint32_t i = 0; i < iters; ++i) {
    for (int run = 0; run < nRunsPerIter; ++run) {
      CHECK_EQ(
          cudaMemcpyAsync(
              dstPtr, srcPtr, nBytes, cudaMemcpyDeviceToDevice, bench.stream),
          cudaSuccess);
    }
  }
  bench.stopTiming();
  float totalTimeMs = bench.measureTime();

  float avgTimeUs = (totalTimeMs / iters / nRunsPerIter) * 1000.0f;
  float busBwGBps = (nBytes / 1e9f) / (avgTimeUs / 1e6f);

  counters["latency (us)"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["bandwidth (GB/s)"] =
      folly::UserMetric(busBwGBps, folly::UserMetric::Type::METRIC);
}

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// Self transport benchmarks - 8MB with different block counts
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_8MB_2blocks,
    8 * 1024 * 1024,
    2);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_8MB_4blocks,
    8 * 1024 * 1024,
    4);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_8MB_8blocks,
    8 * 1024 * 1024,
    8);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_8MB_16blocks,
    8 * 1024 * 1024,
    16);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_8MB_32blocks,
    8 * 1024 * 1024,
    32);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_8MB_64blocks,
    8 * 1024 * 1024,
    64);

// Self transport benchmarks - 64MB with different block counts
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_64MB_2blocks,
    64 * 1024 * 1024,
    2);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_64MB_4blocks,
    64 * 1024 * 1024,
    4);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_64MB_8blocks,
    64 * 1024 * 1024,
    8);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_64MB_16blocks,
    64 * 1024 * 1024,
    16);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_64MB_32blocks,
    64 * 1024 * 1024,
    32);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_64MB_64blocks,
    64 * 1024 * 1024,
    64);

// Self transport benchmarks - 256MB with different block counts
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_256MB_2blocks,
    256 * 1024 * 1024,
    2);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_256MB_4blocks,
    256 * 1024 * 1024,
    4);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_256MB_8blocks,
    256 * 1024 * 1024,
    8);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_256MB_16blocks,
    256 * 1024 * 1024,
    16);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_256MB_32blocks,
    256 * 1024 * 1024,
    32);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_256MB_64blocks,
    256 * 1024 * 1024,
    64);

// Self transport benchmarks - 512MB with different block counts
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_512MB_2blocks,
    512 * 1024 * 1024,
    2);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_512MB_4blocks,
    512 * 1024 * 1024,
    4);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_512MB_8blocks,
    512 * 1024 * 1024,
    8);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_512MB_16blocks,
    512 * 1024 * 1024,
    16);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_512MB_32blocks,
    512 * 1024 * 1024,
    32);
BENCHMARK_MULTI_PARAM_COUNTERS(
    selfTransportWrite,
    size_512MB_64blocks,
    512 * 1024 * 1024,
    64);

BENCHMARK_DRAW_LINE();

// Baseline cudaMemcpy benchmarks for comparison
BENCHMARK_MULTI_PARAM_COUNTERS(cudaMemcpyBaseline, size_8MB, 8 * 1024 * 1024);
BENCHMARK_MULTI_PARAM_COUNTERS(cudaMemcpyBaseline, size_64MB, 64 * 1024 * 1024);
BENCHMARK_MULTI_PARAM_COUNTERS(
    cudaMemcpyBaseline,
    size_256MB,
    256 * 1024 * 1024);
BENCHMARK_MULTI_PARAM_COUNTERS(
    cudaMemcpyBaseline,
    size_512MB,
    512 * 1024 * 1024);

} // namespace comms::pipes::benchmark

int main(int argc, char** argv) {
  CHECK_EQ(cudaSetDevice(0), cudaSuccess);

  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  CHECK_EQ(cudaDeviceReset(), cudaSuccess);

  return 0;
}
