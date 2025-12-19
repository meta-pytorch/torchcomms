// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/testinfra/TestXPlatUtils.h"

//------------------------------------------------------------------------------
// Sync Type Enum (must match MultiTbSyncBench.cu)
//------------------------------------------------------------------------------

enum PerfSyncType {
  kBarrier,
  kFence,
  kDispatch,
  kJoin,
  kSignal,
  kSignalWithSync,
  kBcast,
  kClusterSync
};

//------------------------------------------------------------------------------
// External Kernel Declarations
//------------------------------------------------------------------------------

extern __global__ void MultiTbSyncTestResetKernel(
    int* shmCnts,
    int numCounters);

template <PerfSyncType syncType>
extern __global__ void MultiTbSyncTestPerfKernel(
    const int numWorkers,
    const int numIter,
    const int runId,
    int* shmCnt);

//------------------------------------------------------------------------------
// Common Helper Functions
//------------------------------------------------------------------------------

namespace {

std::unordered_map<PerfSyncType, void*> kSyncTypeToPerfKernel = {
    {PerfSyncType::kBarrier,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kBarrier>},
    {PerfSyncType::kFence,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kFence>},
    {PerfSyncType::kDispatch,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kDispatch>},
    {PerfSyncType::kJoin,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kJoin>},
    {PerfSyncType::kSignal,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kSignal>},
    {PerfSyncType::kSignalWithSync,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kSignalWithSync>},
    {PerfSyncType::kBcast,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kBcast>},
    {PerfSyncType::kClusterSync,
     (void*)MultiTbSyncTestPerfKernel<PerfSyncType::kClusterSync>}};

void launchPerfReset(
    const int numWorkers,
    void* resetArgs[2],
    const PerfSyncType syncType) {
  dim3 resetGrid = {1, 1, 1};
  dim3 resetBlock = {1, 1, 1};
  void* resetFn = (void*)MultiTbSyncTestResetKernel;
  // no reset for cluster
  if (syncType != PerfSyncType::kClusterSync) {
    CUDACHECK_TEST(cudaLaunchKernel(resetFn, resetGrid, resetBlock, resetArgs));
  }
}

#ifndef __HIP_PLATFORM_AMD__
void setupClusterLaunchConfig(
    const int numWorkers,
    cudaLaunchAttribute* attribute,
    cudaLaunchConfig_t& launchConfig) {
  if (attribute) {
    launchConfig.attrs = attribute;
    launchConfig.numAttrs = 1;
  }
  launchConfig.gridDim = numWorkers;
  launchConfig.blockDim = 256;
}
#endif

void launchPerfTest(
    const int numWorkers,
    void* execArgs[4],
    const PerfSyncType syncType,
    cudaStream_t stream) {
  void* fn = kSyncTypeToPerfKernel.at(syncType);
  if (syncType == PerfSyncType::kClusterSync) {
#ifndef __HIP_PLATFORM_AMD__
    cudaLaunchConfig_t launchConfig = {0};
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = numWorkers;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    setupClusterLaunchConfig(numWorkers, attribute, launchConfig);
    launchConfig.stream = stream;
    CUDACHECK_TEST(cudaLaunchKernelExC(&launchConfig, fn, execArgs));
#endif
  } else {
    dim3 grid = {(unsigned int)numWorkers, 1, 1};
    dim3 block = {256, 1, 1};
    CUDACHECK_TEST(cudaLaunchKernel(fn, grid, block, execArgs, 0, stream));
  }
}

bool checkClusterSupport(int numWorkers, PerfSyncType syncType) {
  if (syncType != PerfSyncType::kClusterSync) {
    return true;
  }
#ifndef __HIP_PLATFORM_AMD__
  int maxClusterSize = 0;
  cudaLaunchConfig_t launchConfig = {0};
  setupClusterLaunchConfig(numWorkers, nullptr, launchConfig);
  CUDACHECK_TEST(cudaOccupancyMaxPotentialClusterSize(
      &maxClusterSize,
      (const void*)kSyncTypeToPerfKernel.at(syncType),
      &launchConfig));
  return maxClusterSize >= numWorkers;
#else
  return false;
#endif
}

} // anonymous namespace

//------------------------------------------------------------------------------
// Benchmark Functions
//------------------------------------------------------------------------------

/**
 * Benchmark multi-thread-block synchronization primitives with varying
 * sync types and number of workers.
 */
static void MultiTbSyncPerf(
    uint32_t iters,
    int numWorkers,
    PerfSyncType syncType,
    folly::UserCounters& counters) {
  const int numIter = 1000;

  if (!checkClusterSupport(numWorkers, syncType)) {
    counters["skipped"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  CudaBenchBase bench;

  int* shmCnts = nullptr;
  int numCnts = 3; // bcast requires 3
  CUDACHECK_TEST(cudaMalloc((void**)&shmCnts, sizeof(int) * numCnts));

  void* resetArgs[2] = {(void*)&shmCnts, (void*)&numCnts};
  void* execArgs[4] = {
      (void*)&numWorkers,
      (void*)&numIter,
      nullptr, // update in iteration
      (void*)&shmCnts};

  bench.startTiming();
  for (uint32_t i = 0; i < iters; ++i) {
    launchPerfReset(numWorkers, resetArgs, syncType);

    int runId = static_cast<int>(i);
    execArgs[2] = &runId;

    launchPerfTest(numWorkers, execArgs, syncType, bench.stream);
  }
  bench.stopTiming();
  auto totalTimeMs = bench.measureTime();

  float avgTimeNs = totalTimeMs * 1e6 / iters / numIter;
  counters["deviceTimeNs"] =
      folly::UserMetric(avgTimeNs, folly::UserMetric::Type::METRIC);
  counters["numWorkers"] =
      folly::UserMetric(numWorkers, folly::UserMetric::Type::METRIC);

  CUDACHECK_TEST(cudaFree(shmCnts));
}

//------------------------------------------------------------------------------
// Benchmark Wrapper Functions for each sync type
//------------------------------------------------------------------------------

#define DEFINE_SYNC_BENCH(SyncName, SyncEnum)                             \
  static void MultiTbSync##SyncName(                                      \
      uint32_t iters, int numWorkers, folly::UserCounters& counters) {    \
    MultiTbSyncPerf(iters, numWorkers, PerfSyncType::SyncEnum, counters); \
  }

DEFINE_SYNC_BENCH(Barrier, kBarrier)
DEFINE_SYNC_BENCH(Fence, kFence)
DEFINE_SYNC_BENCH(Dispatch, kDispatch)
DEFINE_SYNC_BENCH(Join, kJoin)
DEFINE_SYNC_BENCH(Signal, kSignal)
DEFINE_SYNC_BENCH(SignalWithSync, kSignalWithSync)
DEFINE_SYNC_BENCH(Bcast, kBcast)
DEFINE_SYNC_BENCH(Cluster, kClusterSync)

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

#define REGISTER_SYNC_BENCH(SyncName)                         \
  BENCHMARK_SINGLE_PARAM_COUNTERS(MultiTbSync##SyncName, 2);  \
  BENCHMARK_SINGLE_PARAM_COUNTERS(MultiTbSync##SyncName, 4);  \
  BENCHMARK_SINGLE_PARAM_COUNTERS(MultiTbSync##SyncName, 8);  \
  BENCHMARK_SINGLE_PARAM_COUNTERS(MultiTbSync##SyncName, 16); \
  BENCHMARK_SINGLE_PARAM_COUNTERS(MultiTbSync##SyncName, 32); \
  BENCHMARK_SINGLE_PARAM_COUNTERS(MultiTbSync##SyncName, 64)

REGISTER_SYNC_BENCH(Barrier);
REGISTER_SYNC_BENCH(Fence);
REGISTER_SYNC_BENCH(Dispatch);
REGISTER_SYNC_BENCH(Join);
REGISTER_SYNC_BENCH(Signal);
REGISTER_SYNC_BENCH(SignalWithSync);
REGISTER_SYNC_BENCH(Bcast);
REGISTER_SYNC_BENCH(Cluster);

int main(int argc, char** argv) {
  CHECK_GE(bench_utils::getNumCudaDevices(), 1);

  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  cudaSetDevice(0);
  cudaDeviceReset();

  return 0;
}
