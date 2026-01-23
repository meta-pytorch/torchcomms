// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <glog/logging.h>

#include "comms/pipes/Signal.cuh"
#include "comms/pipes/benchmarks/SignalBench.cuh"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/utils/CudaRAII.h"

using comms::pipes::Signal;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

//------------------------------------------------------------------------------
// Benchmark Functions
//------------------------------------------------------------------------------

/**
 * Benchmark P2P signaling using Signal
 *
 * Sender (GPU 0) and Receiver (GPU 1) alternate signaling:
 *   - Sender: wait_until() -> signal()
 *   - Receiver: wait_until() -> signal()
 *
 * This measures the latency of signal/wait operations over NVLink,
 * testing both release semantics (signal) and acquire semantics (wait_until).
 *
 * The Signal array is allocated on Receiver's GPU and accessed by Sender
 * via P2P peer access.
 */
static void signalBench(
    uint32_t iters,
    int nBlocks,
    bool useBlockGroups,
    SignalOp op,
    folly::UserCounters& counters) {
  const int nSteps = 100;
  const int nThreads = 256;

  const void* kernelFunc = nullptr;
  switch (op) {
    case SignalOp::SIGNAL_ADD:
      kernelFunc = (const void*)signalAddBenchKernel;
      break;
    case SignalOp::SIGNAL_SET:
      kernelFunc = (const void*)signalSetBenchKernel;
      break;
  }

  const int gpu0 = 0;
  const int gpu1 = 1;

  // Calculate number of Signals needed based on group type
  // For block groups: 1 Signal per block
  // For warp groups: 8 Signals per block (256 threads / 32 threads per warp)
  int numSignals = useBlockGroups ? nBlocks : nBlocks * (nThreads / 32);

  CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
  CudaBenchBase bench0;
  // Allocate Signal array on receiver device
  DeviceBuffer signalBuffer0(numSignals * sizeof(Signal));
  Signal* signal0 = static_cast<Signal*>(signalBuffer0.get());

  CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
  CudaBenchBase bench1;
  // Allocate Signal array on receiver device
  DeviceBuffer signalBuffer1(numSignals * sizeof(Signal));
  Signal* signal1 = static_cast<Signal*>(signalBuffer1.get());

  // Initialize Signals to 0
  std::vector<Signal> initSignals(numSignals);
  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Reset Signals to 0
    CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
    CHECK_EQ(
        cudaMemcpyAsync(
            signal0,
            initSignals.data(),
            numSignals * sizeof(Signal),
            cudaMemcpyHostToDevice,
            bench0.stream),
        cudaSuccess);

    CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
    CHECK_EQ(
        cudaMemcpyAsync(
            signal1,
            initSignals.data(),
            numSignals * sizeof(Signal),
            cudaMemcpyHostToDevice,
            bench1.stream),
        cudaSuccess);

    // Sync both streams before starting
    CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
    CHECK_EQ(cudaStreamSynchronize(bench0.stream), cudaSuccess);
    CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
    CHECK_EQ(cudaStreamSynchronize(bench1.stream), cudaSuccess);

    // Launch kernel 0
    CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
    {
      bench0.startTiming();
      void* kernArgs[4] = {
          (void*)&signal1,
          (void*)&signal0,
          (void*)&nSteps,
          (void*)&useBlockGroups};
      dim3 grid{static_cast<unsigned int>(nBlocks), 1, 1};
      dim3 blocks{static_cast<unsigned int>(nThreads), 1, 1};
      CHECK_EQ(
          cudaLaunchKernel(
              kernelFunc, grid, blocks, kernArgs, 0, bench0.stream),
          cudaSuccess);
      bench0.stopTiming();
    }

    // Launch kernel 1
    CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
    {
      bench1.startTiming();
      void* kernArgs[4] = {
          (void*)&signal0,
          (void*)&signal1,
          (void*)&nSteps,
          (void*)&useBlockGroups};
      dim3 grid{static_cast<unsigned int>(nBlocks), 1, 1};
      dim3 blocks{static_cast<unsigned int>(nThreads), 1, 1};
      CHECK_EQ(
          cudaLaunchKernel(
              kernelFunc, grid, blocks, kernArgs, 0, bench1.stream),
          cudaSuccess);
      bench1.stopTiming();
    }

    CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
    float time0 = bench0.measureTime();
    CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
    float time1 = bench1.measureTime();

    // Use the average time
    totalTimeMs += (time0 + time1) / 2;
  }

  // Calculate per-step latency (each step has 2 signal/wait round-trips)
  float avgTimeUs = (totalTimeMs / iters / nSteps) * 1000.0f;

  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["nGroups"] =
      folly::UserMetric(numSignals, folly::UserMetric::Type::METRIC);
}

//------------------------------------------------------------------------------
// Benchmark Registration Helper Macros
//------------------------------------------------------------------------------

#define REGISTER_SIGNAL_BENCH(nBlocks, useBlockGroups, op, suffix) \
  BENCHMARK_MULTI_PARAM_COUNTERS(                                  \
      signalBench, nBlocks##b_##suffix, nBlocks, useBlockGroups, op)

#define REGISTER_SIGNAL_BENCH_ALL_GROUPS(useBlockGroups, op, suffix) \
  REGISTER_SIGNAL_BENCH(1, useBlockGroups, op, suffix);              \
  REGISTER_SIGNAL_BENCH(2, useBlockGroups, op, suffix);              \
  REGISTER_SIGNAL_BENCH(4, useBlockGroups, op, suffix);              \
  REGISTER_SIGNAL_BENCH(8, useBlockGroups, op, suffix);              \
  REGISTER_SIGNAL_BENCH(16, useBlockGroups, op, suffix);             \
  REGISTER_SIGNAL_BENCH(32, useBlockGroups, op, suffix)

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// Signal benchmarks - warp groups
REGISTER_SIGNAL_BENCH_ALL_GROUPS(false, SignalOp::SIGNAL_ADD, warp_add);

REGISTER_SIGNAL_BENCH_ALL_GROUPS(false, SignalOp::SIGNAL_SET, warp_set);

// Signal benchmarks - block groups
REGISTER_SIGNAL_BENCH_ALL_GROUPS(true, SignalOp::SIGNAL_ADD, block_add);

REGISTER_SIGNAL_BENCH_ALL_GROUPS(true, SignalOp::SIGNAL_SET, block_set);

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
