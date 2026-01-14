// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <glog/logging.h>

#include "comms/pipes/ChunkState.cuh"
#include "comms/pipes/benchmarks/P2pSyncBench.cuh"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/utils/CudaRAII.h"

using comms::pipes::ChunkState;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

//------------------------------------------------------------------------------
// Benchmark Functions
//------------------------------------------------------------------------------

/**
 * Benchmark P2P synchronization using ChunkState
 *
 * Sender (GPU 0) and Receiver (GPU 1) alternate signaling:
 *   - Sender: waitReadyToSend() -> readyToRecv(step)
 *   - Receiver: waitReadyToRecv(step) -> readyToSend()
 *
 * The ChunkState array is allocated on Receiver's GPU and accessed by Sender
 * via P2P peer access.
 */
static void p2pSyncBench(
    uint32_t iters,
    int nBlocks,
    bool useBlockGroups,
    folly::UserCounters& counters) {
  const int nSteps = 100;
  const int nThreads = 256;

  const int receiverCudaDev = 1;
  const int senderCudaDev = 0;

  // Calculate number of ChunkStates needed based on group type
  // For block groups: 1 ChunkState per block
  // For warp groups: 8 ChunkStates per block (256 threads / 32 threads per
  // warp)
  int numChunkStates = useBlockGroups ? nBlocks : nBlocks * (nThreads / 32);

  // Allocate ChunkState array on receiver device
  CHECK_EQ(cudaSetDevice(receiverCudaDev), cudaSuccess);
  DeviceBuffer chunkStateBuffer(numChunkStates * sizeof(ChunkState));
  ChunkState* chunkStates = static_cast<ChunkState*>(chunkStateBuffer.get());

  // Initialize ChunkStates to READY_TO_SEND state
  std::vector<ChunkState> initStates(numChunkStates);
  CHECK_EQ(
      cudaMemcpy(
          chunkStates,
          initStates.data(),
          numChunkStates * sizeof(ChunkState),
          cudaMemcpyHostToDevice),
      cudaSuccess);

  // Create streams for both devices
  CudaBenchBase receiverBench;

  CHECK_EQ(cudaSetDevice(senderCudaDev), cudaSuccess);
  cudaStream_t senderStream;
  CHECK_EQ(cudaStreamCreate(&senderStream), cudaSuccess);

  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Reset ChunkStates to READY_TO_SEND state
    CHECK_EQ(cudaSetDevice(receiverCudaDev), cudaSuccess);
    CHECK_EQ(
        cudaMemcpyAsync(
            chunkStates,
            initStates.data(),
            numChunkStates * sizeof(ChunkState),
            cudaMemcpyHostToDevice,
            receiverBench.stream),
        cudaSuccess);
    CHECK_EQ(cudaStreamSynchronize(receiverBench.stream), cudaSuccess);

    // Start timing
    receiverBench.startTiming();

    // Launch receiver kernel first (it will wait for sender signals)
    {
      bool isSender = false;
      void* kernArgs[4] = {
          (void*)&chunkStates,
          (void*)&isSender,
          (void*)&nSteps,
          (void*)&useBlockGroups};
      dim3 grid{static_cast<unsigned int>(nBlocks), 1, 1};
      dim3 blocks{static_cast<unsigned int>(nThreads), 1, 1};
      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)p2pSyncKernel,
              grid,
              blocks,
              kernArgs,
              0,
              receiverBench.stream),
          cudaSuccess);
    }

    // Launch sender kernel on sender device
    CHECK_EQ(cudaSetDevice(senderCudaDev), cudaSuccess);
    {
      bool isSender = true;
      void* kernArgs[4] = {
          (void*)&chunkStates,
          (void*)&isSender,
          (void*)&nSteps,
          (void*)&useBlockGroups};
      dim3 grid{static_cast<unsigned int>(nBlocks), 1, 1};
      dim3 blocks{static_cast<unsigned int>(nThreads), 1, 1};
      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)p2pSyncKernel,
              grid,
              blocks,
              kernArgs,
              0,
              senderStream),
          cudaSuccess);
    }

    // Stop timing on receiver (waits for receiver kernel to complete)
    CHECK_EQ(cudaSetDevice(receiverCudaDev), cudaSuccess);
    receiverBench.stopTiming();
    totalTimeMs += receiverBench.measureTime();
  }

  // Cleanup sender stream
  CHECK_EQ(cudaSetDevice(senderCudaDev), cudaSuccess);
  CHECK_EQ(cudaStreamDestroy(senderStream), cudaSuccess);

  // Calculate per-step latency
  float avgTimeUs = (totalTimeMs / iters / nSteps) * 1000.0f;

  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["nGroups"] =
      folly::UserMetric(numChunkStates, folly::UserMetric::Type::METRIC);
}

//------------------------------------------------------------------------------
// Benchmark Registration Helper Macros
//------------------------------------------------------------------------------

#define REGISTER_P2P_SYNC_BENCH(nBlocks, useBlockGroups, suffix) \
  BENCHMARK_MULTI_PARAM_COUNTERS(                                \
      p2pSyncBench, nBlocks##b_##suffix, nBlocks, useBlockGroups)

#define REGISTER_P2P_SYNC_BENCH_ALL_GROUPS(useBlockGroups, suffix) \
  REGISTER_P2P_SYNC_BENCH(1, useBlockGroups, suffix);              \
  REGISTER_P2P_SYNC_BENCH(2, useBlockGroups, suffix);              \
  REGISTER_P2P_SYNC_BENCH(4, useBlockGroups, suffix);              \
  REGISTER_P2P_SYNC_BENCH(8, useBlockGroups, suffix);              \
  REGISTER_P2P_SYNC_BENCH(16, useBlockGroups, suffix);             \
  REGISTER_P2P_SYNC_BENCH(32, useBlockGroups, suffix)

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// P2P Sync benchmarks - warp groups
REGISTER_P2P_SYNC_BENCH_ALL_GROUPS(false, warp);

// P2P Sync benchmarks - block groups
REGISTER_P2P_SYNC_BENCH_ALL_GROUPS(true, block);

} // namespace comms::pipes::benchmark

int main(int argc, char** argv) {
  CHECK_GE(bench_utils::getNumCudaDevices(), 2);

  // Enable P2P access once at startup
  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CHECK_EQ(cudaDeviceEnablePeerAccess(1, 0), cudaSuccess);

  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  // Cleanup
  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CHECK_EQ(cudaDeviceReset(), cudaSuccess);
  CHECK_EQ(cudaSetDevice(1), cudaSuccess);
  CHECK_EQ(cudaDeviceReset(), cudaSuccess);

  return 0;
}
