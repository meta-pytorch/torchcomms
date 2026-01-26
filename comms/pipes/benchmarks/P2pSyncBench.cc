// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <glog/logging.h>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/ChunkState.cuh"
#include "comms/pipes/benchmarks/P2pSyncBench.cuh"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/utils/CudaRAII.h"

using comms::pipes::ChunkState;
using comms::pipes::SyncScope;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

//------------------------------------------------------------------------------
// Benchmark Functions
//------------------------------------------------------------------------------

/**
 * Benchmark P2P synchronization using ChunkState
 *
 * Sender (GPU 0) and Receiver (GPU 1) alternate signaling:
 *   - Sender: wait_ready_to_send() -> ready_to_recv(step)
 *   - Receiver: wait_ready_to_recv(step) -> ready_to_send()
 *
 * The ChunkState array is allocated on Receiver's GPU and accessed by Sender
 * via P2P peer access.
 */
static void p2pSyncBench(
    uint32_t iters,
    int nBlocks,
    SyncScope groupScope,
    folly::UserCounters& counters,
    int clusterSize = 1) {
  const int nSteps = 100;
  const int nThreads = 256;

  const int receiverCudaDev = 1;
  const int senderCudaDev = 0;

  // Calculate number of ChunkStates needed based on group type
  // For block groups: 1 ChunkState per block
  // For multiwarp groups: 2 ChunkStates per block (256 threads / 128 threads
  // per multiwarp)
  // For warp groups: 8 ChunkStates per block (256 threads / 32 threads per
  // warp)
  // For cluster groups: 1 ChunkState per cluster
  int numChunkStates;
  switch (groupScope) {
    case SyncScope::BLOCK:
      numChunkStates = nBlocks;
      break;
    case SyncScope::MULTIWARP:
      numChunkStates = nBlocks * (nThreads / 128); // 4 warps per multiwarp
      break;
    case SyncScope::CLUSTER:
      numChunkStates = nBlocks / clusterSize; // 1 per cluster
      break;
    case SyncScope::WARP:
    default:
      numChunkStates = nBlocks * (nThreads / 32);
      break;
  }

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
          (void*)&groupScope};
      dim3 grid{static_cast<unsigned int>(nBlocks), 1, 1};
      dim3 blocks{static_cast<unsigned int>(nThreads), 1, 1};

      std::optional<dim3> clusterDimOpt =
          (groupScope == SyncScope::CLUSTER && clusterSize > 1)
          ? std::optional{dim3(clusterSize, 1, 1)}
          : std::nullopt;
      CHECK_EQ(
          comms::common::launchKernel(
              (void*)p2pSyncKernel,
              grid,
              blocks,
              kernArgs,
              receiverBench.stream,
              clusterDimOpt),
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
          (void*)&groupScope};
      dim3 grid{static_cast<unsigned int>(nBlocks), 1, 1};
      dim3 blocks{static_cast<unsigned int>(nThreads), 1, 1};

      std::optional<dim3> clusterDimOpt =
          (groupScope == SyncScope::CLUSTER && clusterSize > 1)
          ? std::optional{dim3(clusterSize, 1, 1)}
          : std::nullopt;
      CHECK_EQ(
          comms::common::launchKernel(
              (void*)p2pSyncKernel,
              grid,
              blocks,
              kernArgs,
              senderStream,
              clusterDimOpt),
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

#define REGISTER_P2P_SYNC_BENCH(nBlocks, groupScope, suffix) \
  BENCHMARK_MULTI_PARAM_COUNTERS(                            \
      p2pSyncBench, nBlocks##b_##suffix, nBlocks, groupScope)

#define REGISTER_P2P_SYNC_BENCH_ALL_GROUPS(groupScope, suffix) \
  REGISTER_P2P_SYNC_BENCH(4, groupScope, suffix);              \
  REGISTER_P2P_SYNC_BENCH(8, groupScope, suffix);              \
  REGISTER_P2P_SYNC_BENCH(16, groupScope, suffix);             \
  REGISTER_P2P_SYNC_BENCH(32, groupScope, suffix)

//------------------------------------------------------------------------------
// Cluster Benchmark Wrapper Function
// (wrapper function with hardcoded clusterSize since the macro doesn't
// support extra parameters)
//------------------------------------------------------------------------------

static void p2pSyncBenchCluster(
    uint32_t iters,
    int nBlocks,
    folly::UserCounters& counters) {
  p2pSyncBench(
      iters,
      nBlocks,
      SyncScope::CLUSTER,
      counters,
      comms::common::kDefaultClusterSize);
}

// Cluster benchmarks - nBlocks must be divisible by clusterSize
#define REGISTER_P2P_SYNC_BENCH_CLUSTER(nBlocks, suffix) \
  BENCHMARK_MULTI_PARAM_COUNTERS(                        \
      p2pSyncBenchCluster, nBlocks##b_##suffix, nBlocks)

#define REGISTER_P2P_SYNC_BENCH_ALL_GROUPS_CLUSTER(suffix) \
  REGISTER_P2P_SYNC_BENCH_CLUSTER(4, suffix);              \
  REGISTER_P2P_SYNC_BENCH_CLUSTER(8, suffix);              \
  REGISTER_P2P_SYNC_BENCH_CLUSTER(16, suffix);             \
  REGISTER_P2P_SYNC_BENCH_CLUSTER(32, suffix)

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// P2P Sync benchmarks - warp groups
REGISTER_P2P_SYNC_BENCH_ALL_GROUPS(SyncScope::WARP, warp);

// P2P Sync benchmarks - multiwarp groups
REGISTER_P2P_SYNC_BENCH_ALL_GROUPS(SyncScope::MULTIWARP, multiwarp);

// P2P Sync benchmarks - block groups
REGISTER_P2P_SYNC_BENCH_ALL_GROUPS(SyncScope::BLOCK, block);

// P2P Sync benchmarks - cluster groups (4 blocks per cluster)
REGISTER_P2P_SYNC_BENCH_ALL_GROUPS_CLUSTER(cluster);

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
