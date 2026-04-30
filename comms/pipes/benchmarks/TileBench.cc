// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <glog/logging.h>

#include "comms/pipes/benchmarks/TileBench.cuh"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

static void run_tile_bench(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    const void* kernel,
    int nBufs,
    float bw_multiplier,
    folly::UserCounters& counters) {
  const int nRunsPerIter = 50;
  const int nThreads = 256;
  size_t nelems = nBytes / sizeof(float);

  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CudaBenchBase bench;

  std::vector<DeviceBuffer> bufs;
  bufs.reserve(nBufs);
  for (int i = 0; i < nBufs; i++) {
    bufs.emplace_back(nBytes);
  }

  std::vector<void*> ptrs;
  ptrs.reserve(nBufs);
  for (auto& b : bufs) {
    ptrs.push_back(b.get());
  }

  std::vector<void*> kernArgs;
  kernArgs.reserve(nBufs + 2);
  for (auto& p : ptrs) {
    kernArgs.push_back(&p);
  }
  kernArgs.push_back(&nelems);
  kernArgs.push_back(const_cast<int*>(&nRunsPerIter));

  bench.startTiming();
  for (uint32_t i = 0; i < iters; ++i) {
    CHECK_EQ(
        cudaLaunchKernel(
            kernel,
            dim3(nBlocks),
            dim3(nThreads),
            kernArgs.data(),
            0,
            bench.stream),
        cudaSuccess);
  }
  bench.stopTiming();
  float totalTimeMs = bench.measureTime();

  float avgTimeUs = (totalTimeMs / iters / nRunsPerIter) * 1000.0f;
  float busBwGBps = (bw_multiplier * nBytes / 1e9f) / (avgTimeUs / 1e6f);

  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["busBwGBps"] =
      folly::UserMetric(busBwGBps, folly::UserMetric::Type::METRIC);
}

static void tile_copy_bench(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    folly::UserCounters& counters) {
  run_tile_bench(
      iters, nBytes, nBlocks, (const void*)tile_copy_kernel, 2, 1.0f, counters);
}

static void tile_reduce_sum_bench(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    folly::UserCounters& counters) {
  run_tile_bench(
      iters,
      nBytes,
      nBlocks,
      (const void*)tile_reduce_sum_kernel,
      3,
      2.0f,
      counters);
}

#define REGISTER_TILE_BENCH_FOR_SIZE(func, sizeMB)   \
  BENCHMARK_MULTI_PARAM_COUNTERS(                    \
      func, sizeMB##MB_4b, sizeMB * 1024 * 1024, 4); \
  BENCHMARK_MULTI_PARAM_COUNTERS(                    \
      func, sizeMB##MB_8b, sizeMB * 1024 * 1024, 8); \
  BENCHMARK_MULTI_PARAM_COUNTERS(func, sizeMB##MB_16b, sizeMB * 1024 * 1024, 16)

#define REGISTER_TILE_BENCH_ALL_SIZES(func) \
  REGISTER_TILE_BENCH_FOR_SIZE(func, 2);    \
  REGISTER_TILE_BENCH_FOR_SIZE(func, 4);    \
  REGISTER_TILE_BENCH_FOR_SIZE(func, 8)

REGISTER_TILE_BENCH_ALL_SIZES(tile_copy_bench);
REGISTER_TILE_BENCH_ALL_SIZES(tile_reduce_sum_bench);

} // namespace comms::pipes::benchmark

int main(int argc, char** argv) {
  CHECK_GE(bench_utils::getNumCudaDevices(), 1);
  CHECK_EQ(cudaSetDevice(0), cudaSuccess);

  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  CHECK_EQ(cudaDeviceReset(), cudaSuccess);
  return 0;
}
