// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <glog/logging.h>

#include "comms/pipes/BarrierState.cuh"
#include "comms/pipes/benchmarks/BarrierBench.cuh"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/utils/CudaRAII.h"

using comms::pipes::BarrierState;
using comms::pipes::getBarrierBufferSize;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

//------------------------------------------------------------------------------
// Benchmark Functions
//------------------------------------------------------------------------------

/**
 * Benchmark P2P barrier synchronization using P2pNvlTransportDevice
 *
 * GPU 0 and GPU 1 perform barrier_sync_threadgroup operations concurrently.
 * Each call to barrier_sync_threadgroup:
 *   - Arrives at peer's barrier via NVLink remote write
 *   - Waits on local barrier until peer arrives
 *
 * This measures the latency of the barrier_sync_threadgroup API over NVLink.
 */
static void barrierBench(
    uint32_t iters,
    int nBlocks,
    bool useBlockGroups,
    folly::UserCounters& counters) {
  const int nSteps = 100;
  const int nThreads = 256;

  const int gpu0 = 0;
  const int gpu1 = 1;

  // Calculate number of barriers needed based on group type
  // For block groups: 1 Barrier per block
  // For warp groups: 8 Barriers per block (256 threads / 32 threads per warp)
  int numBarriers = useBlockGroups ? nBlocks : nBlocks * (nThreads / 32);
  std::size_t barrierBufferSize = getBarrierBufferSize(numBarriers);

  // Allocate barrier buffers on each GPU
  CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
  CudaBenchBase bench0;
  DeviceBuffer barrierBuffer0(barrierBufferSize);
  BarrierState* barrier0 = static_cast<BarrierState*>(barrierBuffer0.get());

  CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
  CudaBenchBase bench1;
  DeviceBuffer barrierBuffer1(barrierBufferSize);
  BarrierState* barrier1 = static_cast<BarrierState*>(barrierBuffer1.get());

  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Reset barriers to 0
    CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
    CHECK_EQ(
        cudaMemsetAsync(barrier0, 0, barrierBufferSize, bench0.stream),
        cudaSuccess);

    CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
    CHECK_EQ(
        cudaMemsetAsync(barrier1, 0, barrierBufferSize, bench1.stream),
        cudaSuccess);

    // Sync both streams before starting
    CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
    CHECK_EQ(cudaStreamSynchronize(bench0.stream), cudaSuccess);
    CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
    CHECK_EQ(cudaStreamSynchronize(bench1.stream), cudaSuccess);

    // Launch kernel on GPU 0
    // GPU 0: local=barrier0, remote=barrier1 (waits on barrier0, arrives on
    // barrier1)
    CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
    bench0.startTiming();
    launchBarrierBenchKernel(
        barrier0, // local barrier (wait here)
        barrier1, // remote barrier (arrive here)
        numBarriers,
        gpu0,
        gpu1,
        nBlocks,
        nThreads,
        nSteps,
        useBlockGroups,
        bench0.stream);
    bench0.stopTiming();

    // Launch kernel on GPU 1
    // GPU 1: local=barrier1, remote=barrier0 (waits on barrier1, arrives on
    // barrier0)
    CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
    bench1.startTiming();
    launchBarrierBenchKernel(
        barrier1, // local barrier (wait here)
        barrier0, // remote barrier (arrive here)
        numBarriers,
        gpu1,
        gpu0,
        nBlocks,
        nThreads,
        nSteps,
        useBlockGroups,
        bench1.stream);
    bench1.stopTiming();

    CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
    float time0 = bench0.measureTime();
    CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
    float time1 = bench1.measureTime();

    // Use the average time
    totalTimeMs += (time0 + time1) / 2;
  }

  // Calculate per-step latency
  float avgTimeUs = (totalTimeMs / iters / nSteps) * 1000.0f;

  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["nGroups"] =
      folly::UserMetric(numBarriers, folly::UserMetric::Type::METRIC);
}

//------------------------------------------------------------------------------
// Benchmark Registration Helper Macros
//------------------------------------------------------------------------------

#define REGISTER_BARRIER_BENCH(nBlocks, useBlockGroups, suffix) \
  BENCHMARK_MULTI_PARAM_COUNTERS(                               \
      barrierBench, nBlocks##b_##suffix, nBlocks, useBlockGroups)

#define REGISTER_BARRIER_BENCH_ALL_GROUPS(useBlockGroups, suffix) \
  REGISTER_BARRIER_BENCH(1, useBlockGroups, suffix);              \
  REGISTER_BARRIER_BENCH(2, useBlockGroups, suffix);              \
  REGISTER_BARRIER_BENCH(4, useBlockGroups, suffix);              \
  REGISTER_BARRIER_BENCH(8, useBlockGroups, suffix);              \
  REGISTER_BARRIER_BENCH(16, useBlockGroups, suffix);             \
  REGISTER_BARRIER_BENCH(32, useBlockGroups, suffix)

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// Barrier benchmarks - warp groups
REGISTER_BARRIER_BENCH_ALL_GROUPS(false, warp);

// Barrier benchmarks - block groups
REGISTER_BARRIER_BENCH_ALL_GROUPS(true, block);

} // namespace comms::pipes::benchmark

int main(int argc, char** argv) {
  CHECK_GE(bench_utils::getNumCudaDevices(), 2);

  // Enable bidirectional P2P access at startup
  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CHECK_EQ(cudaDeviceEnablePeerAccess(1, 0), cudaSuccess);
  CHECK_EQ(cudaSetDevice(1), cudaSuccess);
  CHECK_EQ(cudaDeviceEnablePeerAccess(0, 0), cudaSuccess);

  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  // Cleanup
  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CHECK_EQ(cudaDeviceReset(), cudaSuccess);
  CHECK_EQ(cudaSetDevice(1), cudaSuccess);
  CHECK_EQ(cudaDeviceReset(), cudaSuccess);

  return 0;
}
