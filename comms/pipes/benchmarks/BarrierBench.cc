// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <glog/logging.h>

#include "comms/pipes/ChunkState.cuh"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/SignalState.cuh"
#include "comms/pipes/benchmarks/BarrierBench.cuh"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/utils/CudaRAII.h"

using comms::pipes::ChunkState;
using comms::pipes::DeviceSpan;
using comms::pipes::LocalState;
using comms::pipes::P2pNvlTransportDevice;
using comms::pipes::P2pNvlTransportOptions;
using comms::pipes::RemoteState;
using comms::pipes::SignalState;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

void launchP2pBarrierBench(
    P2pNvlTransportDevice& p2p,
    int nSteps,
    int nBlocks,
    int nThreads,
    bool useBlockGroups,
    cudaStream_t stream) {
  void* args[] = {&p2p, &nSteps, &useBlockGroups};
  dim3 grid{static_cast<unsigned int>(nBlocks), 1, 1};
  dim3 blocks{static_cast<unsigned int>(nThreads), 1, 1};
  CHECK_EQ(
      cudaLaunchKernel(
          (const void*)p2pBarrierThreadGroupBenchKernel,
          grid,
          blocks,
          args,
          0,
          stream),
      cudaSuccess);
}

//------------------------------------------------------------------------------
// Benchmark Functions
//------------------------------------------------------------------------------

/**
 * Benchmark P2P barrier synchronization using P2pNvlTransportDevice::barrier()
 *
 * GPU 0 and GPU 1 synchronize using the barrier API:
 *   - Each GPU signals peer's SignalState via NVLink
 *   - Each GPU waits on its own SignalState for the expected counter value
 *
 * The SignalState for each GPU is allocated on that GPU and accessed by the
 * peer via P2P peer access.
 */
static void p2pBarrierThreadGroupBench(
    uint32_t iters,
    int nBlocks,
    bool useBlockGroups,
    folly::UserCounters& counters) {
  const int nSteps = 100;
  const int nThreads = 256;

  const int gpu0 = 0;
  const int gpu1 = 1;

  // Allocate SignalState on each GPU for barrier synchronization
  // GPU 0's signalState: GPU 1 signals it, GPU 0 waits on it
  CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
  DeviceBuffer signal0Buffer(sizeof(SignalState));
  SignalState* signal0 = static_cast<SignalState*>(signal0Buffer.get());

  // GPU 1's signalState: GPU 0 signals it, GPU 1 waits on it
  CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
  DeviceBuffer signal1Buffer(sizeof(SignalState));
  SignalState* signal1 = static_cast<SignalState*>(signal1Buffer.get());

  // Create P2pNvlTransportOptions (minimal config for barrier-only benchmark)
  P2pNvlTransportOptions options{
      .dataBufferSize = 0,
      .chunkSize = 0,
      .pipelineDepth = 0,
  };

  // Create streams and events for both devices
  CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
  CudaBenchBase bench0;

  CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
  CudaBenchBase bench1;

  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Reset SignalStates to initial state
    SignalState initialSignal;

    CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
    CHECK_EQ(
        cudaMemcpyAsync(
            signal0,
            &initialSignal,
            sizeof(SignalState),
            cudaMemcpyHostToDevice,
            bench0.stream),
        cudaSuccess);

    CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
    CHECK_EQ(
        cudaMemcpyAsync(
            signal1,
            &initialSignal,
            sizeof(SignalState),
            cudaMemcpyHostToDevice,
            bench1.stream),
        cudaSuccess);

    // Sync both streams before starting
    CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
    CHECK_EQ(cudaStreamSynchronize(bench0.stream), cudaSuccess);
    CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
    CHECK_EQ(cudaStreamSynchronize(bench1.stream), cudaSuccess);

    // Create P2pNvlTransportDevice for each GPU
    // GPU 0: local=signal0, remote=signal1
    LocalState local0{
        .dataBuffer = nullptr,
        .stateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
        .signalState = signal0,
    };
    RemoteState remote0{
        .dataBuffer = nullptr,
        .stateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
        .signalState = signal1,
    };
    P2pNvlTransportDevice p2p0(0, 1, options, local0, remote0);

    // GPU 1: local=signal1, remote=signal0
    LocalState local1{
        .dataBuffer = nullptr,
        .stateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
        .signalState = signal1,
    };
    RemoteState remote1{
        .dataBuffer = nullptr,
        .stateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
        .signalState = signal0,
    };
    P2pNvlTransportDevice p2p1(1, 0, options, local1, remote1);

    // Launch both barrier kernels
    CHECK_EQ(cudaSetDevice(gpu0), cudaSuccess);
    bench0.startTiming();
    launchP2pBarrierBench(
        p2p0, nSteps, nBlocks, nThreads, useBlockGroups, bench0.stream);
    bench0.stopTiming();

    CHECK_EQ(cudaSetDevice(gpu1), cudaSuccess);
    bench1.startTiming();
    launchP2pBarrierBench(
        p2p1, nSteps, nBlocks, nThreads, useBlockGroups, bench1.stream);
    bench1.stopTiming();

    float time0 = bench0.measureTime();
    float time1 = bench1.measureTime();

    // Use the max time (since barrier requires both GPUs to sync)
    totalTimeMs += (time0 + time1) / 2;
  }

  // Calculate per-barrier latency
  float avgTimeUs = (totalTimeMs / iters / nSteps) * 1000.0f;

  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["nBlocks"] =
      folly::UserMetric(nBlocks, folly::UserMetric::Type::METRIC);
}

//------------------------------------------------------------------------------
// Benchmark Registration Helper Macros
//------------------------------------------------------------------------------

#define REGISTER_P2P_BARRIER_THREADGROUP_BENCH( \
    nBlocks, useBlockGroups, suffix)            \
  BENCHMARK_MULTI_PARAM_COUNTERS(               \
      p2pBarrierThreadGroupBench,               \
      nBlocks##b_##suffix,                      \
      nBlocks,                                  \
      useBlockGroups)

#define REGISTER_P2P_BARRIER_THREADGROUP_BENCH_ALL_BLOCKS(            \
    useBlockGroups, suffix)                                           \
  REGISTER_P2P_BARRIER_THREADGROUP_BENCH(1, useBlockGroups, suffix);  \
  REGISTER_P2P_BARRIER_THREADGROUP_BENCH(2, useBlockGroups, suffix);  \
  REGISTER_P2P_BARRIER_THREADGROUP_BENCH(4, useBlockGroups, suffix);  \
  REGISTER_P2P_BARRIER_THREADGROUP_BENCH(8, useBlockGroups, suffix);  \
  REGISTER_P2P_BARRIER_THREADGROUP_BENCH(16, useBlockGroups, suffix); \
  REGISTER_P2P_BARRIER_THREADGROUP_BENCH(32, useBlockGroups, suffix)

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// P2P Barrier benchmarks - warp groups
REGISTER_P2P_BARRIER_THREADGROUP_BENCH_ALL_BLOCKS(false, warp);

// P2P Barrier benchmarks - block groups
REGISTER_P2P_BARRIER_THREADGROUP_BENCH_ALL_BLOCKS(true, block);

} // namespace comms::pipes::benchmark

int main(int argc, char** argv) {
  CHECK_GE(bench_utils::getNumCudaDevices(), 2);

  // Enable bidirectional P2P access
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
