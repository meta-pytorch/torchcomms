// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <glog/logging.h>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/benchmarks/CopyKernelBench.cuh"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

//------------------------------------------------------------------------------
// Shared Benchmark Helpers
//------------------------------------------------------------------------------

/**
 * Compute benchmark metrics and populate folly counters.
 * Shared by all copy kernel benchmark functions.
 */
static void populateBenchCounters(
    folly::UserCounters& counters,
    float totalTimeMs,
    uint32_t iters,
    int nRunsPerIter,
    size_t nBytes,
    int nBlocks,
    int nThreads,
    SyncScope groupScope,
    int clusterSize,
    int hbmMultiplier = 2) {
  float avgTimeUs = (totalTimeMs / iters / nRunsPerIter) * 1000.0f;
  float busBwGBps = (nBytes / 1e9f) / (avgTimeUs / 1e6f);
  float hbmTrafficGBps = busBwGBps * hbmMultiplier;

  size_t nGroups;
  switch (groupScope) {
    case SyncScope::BLOCK:
      nGroups = nBlocks;
      break;
    case SyncScope::MULTIWARP:
      nGroups = nBlocks * (nThreads / 128);
      break;
    case SyncScope::CLUSTER:
      nGroups = nBlocks / clusterSize;
      break;
    case SyncScope::WARP:
    default:
      nGroups = nBlocks * (nThreads / 32);
      break;
  }
  size_t chunkSize = nBytes / nGroups / 1024;

  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["busBwGBps"] =
      folly::UserMetric(busBwGBps, folly::UserMetric::Type::METRIC);
  counters["hbmTrafficGBps"] =
      folly::UserMetric(hbmTrafficGBps, folly::UserMetric::Type::METRIC);
  counters["nGroups"] =
      folly::UserMetric(nGroups, folly::UserMetric::Type::METRIC);
  counters["chunkSizeKB"] =
      folly::UserMetric(chunkSize, folly::UserMetric::Type::METRIC);
}

//------------------------------------------------------------------------------
// Benchmark Functions
//------------------------------------------------------------------------------

/**
 * Benchmark P2P copyKernel (copy from GPU 0 to GPU 1) using IPC memory
 */
static void p2pCopyKernel(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    SyncScope groupScope,
    folly::UserCounters& counters,
    int clusterSize = 1) {
  const int nRunsPerIter = 50;

  const int senderCudaDev = 0;
  const int receiverCudaDev = 1;

  // Allocate destination buffer on receiver device
  CHECK_EQ(cudaSetDevice(receiverCudaDev), cudaSuccess);
  DeviceBuffer dstBuffer(nBytes);
  char* dstPtr = static_cast<char*>(dstBuffer.get());

  // Allocate source buffer on sender device
  CHECK_EQ(cudaSetDevice(senderCudaDev), cudaSuccess);
  DeviceBuffer srcBuffer(nBytes);
  char* srcPtr = static_cast<char*>(srcBuffer.get());

  CudaBenchBase bench;
  float totalTimeMs = 0.0f;
  const int nThreads = 256;
  for (uint32_t i = 0; i < iters; ++i) {
    bench.startTiming();

    void* kernArgs[5] = {
        (void*)&dstPtr,
        (void*)&srcPtr,
        (void*)&nBytes,
        (void*)&nRunsPerIter,
        (void*)&groupScope};
    dim3 grid{static_cast<unsigned int>(nBlocks), 1, 1};
    dim3 blocks{nThreads, 1, 1};

    std::optional<dim3> clusterDimOpt =
        (groupScope == SyncScope::CLUSTER && clusterSize > 1)
        ? std::optional{dim3(clusterSize, 1, 1)}
        : std::nullopt;
    CHECK_EQ(
        comms::common::launchKernel(
            (void*)copyKernel,
            grid,
            blocks,
            kernArgs,
            bench.stream,
            clusterDimOpt),
        cudaSuccess);

    bench.stopTiming();
    totalTimeMs += bench.measureTime();
  }

  populateBenchCounters(
      counters,
      totalTimeMs,
      iters,
      nRunsPerIter,
      nBytes,
      nBlocks,
      nThreads,
      groupScope,
      clusterSize);
}

/**
 * Benchmark D2D copyKernel (copy within the same GPU)
 */
static void d2dCopyKernel(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    SyncScope groupScope,
    folly::UserCounters& counters,
    int clusterSize = 1) {
  const int nRunsPerIter = 50;

  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CudaBenchBase bench;

  DeviceBuffer srcBuffer(nBytes);
  DeviceBuffer dstBuffer(nBytes);

  char* srcPtr = static_cast<char*>(srcBuffer.get());
  char* dstPtr = static_cast<char*>(dstBuffer.get());

  float totalTimeMs = 0.0f;
  const int nThreads = 256;
  for (uint32_t i = 0; i < iters; ++i) {
    bench.startTiming();

    void* kernArgs[5] = {
        (void*)&dstPtr,
        (void*)&srcPtr,
        (void*)&nBytes,
        (void*)&nRunsPerIter,
        (void*)&groupScope};
    dim3 grid{static_cast<unsigned int>(nBlocks), 1, 1};
    dim3 blocks{nThreads, 1, 1};

    std::optional<dim3> clusterDimOpt =
        (groupScope == SyncScope::CLUSTER && clusterSize > 1)
        ? std::optional{dim3(clusterSize, 1, 1)}
        : std::nullopt;
    CHECK_EQ(
        comms::common::launchKernel(
            (void*)copyKernel,
            grid,
            blocks,
            kernArgs,
            bench.stream,
            clusterDimOpt),
        cudaSuccess);

    bench.stopTiming();
    totalTimeMs += bench.measureTime();
  }

  populateBenchCounters(
      counters,
      totalTimeMs,
      iters,
      nRunsPerIter,
      nBytes,
      nBlocks,
      nThreads,
      groupScope,
      clusterSize);
}

//------------------------------------------------------------------------------
// Benchmark Registration Helper Macros
//------------------------------------------------------------------------------

#define REGISTER_COPY_BENCH_FOR_SIZE(func, sizeMB, groupScope, suffix)    \
  BENCHMARK_MULTI_PARAM_COUNTERS(                                         \
      func, sizeMB##MB_4b_##suffix, sizeMB * 1024 * 1024, 4, groupScope); \
  BENCHMARK_MULTI_PARAM_COUNTERS(                                         \
      func, sizeMB##MB_8b_##suffix, sizeMB * 1024 * 1024, 8, groupScope); \
  BENCHMARK_MULTI_PARAM_COUNTERS(                                         \
      func, sizeMB##MB_16b_##suffix, sizeMB * 1024 * 1024, 16, groupScope)

#define REGISTER_COPY_BENCH_ALL_SIZES(func, groupScope, suffix) \
  REGISTER_COPY_BENCH_FOR_SIZE(func, 2, groupScope, suffix);    \
  REGISTER_COPY_BENCH_FOR_SIZE(func, 4, groupScope, suffix);    \
  REGISTER_COPY_BENCH_FOR_SIZE(func, 8, groupScope, suffix)

//------------------------------------------------------------------------------
// Cluster Benchmark Wrapper Functions
// (wrapper functions with hardcoded clusterSize since the macro doesn't
// support extra parameters)
//------------------------------------------------------------------------------

static void p2pCopyKernelCluster(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    folly::UserCounters& counters) {
  p2pCopyKernel(
      iters,
      nBytes,
      nBlocks,
      SyncScope::CLUSTER,
      counters,
      comms::common::kDefaultClusterSize);
}

static void d2dCopyKernelCluster(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    folly::UserCounters& counters) {
  d2dCopyKernel(
      iters,
      nBytes,
      nBlocks,
      SyncScope::CLUSTER,
      counters,
      comms::common::kDefaultClusterSize);
}

// Cluster benchmarks - nBlocks must be divisible by clusterSize, so we use 4,
// 8, 16 blocks
#define REGISTER_COPY_BENCH_FOR_SIZE_CLUSTER(func, sizeMB, suffix) \
  BENCHMARK_MULTI_PARAM_COUNTERS(                                  \
      func, sizeMB##MB_4b_##suffix, sizeMB * 1024 * 1024, 4);      \
  BENCHMARK_MULTI_PARAM_COUNTERS(                                  \
      func, sizeMB##MB_8b_##suffix, sizeMB * 1024 * 1024, 8);      \
  BENCHMARK_MULTI_PARAM_COUNTERS(                                  \
      func, sizeMB##MB_16b_##suffix, sizeMB * 1024 * 1024, 16)

#define REGISTER_COPY_BENCH_ALL_SIZES_CLUSTER(func, suffix) \
  REGISTER_COPY_BENCH_FOR_SIZE_CLUSTER(func, 2, suffix);    \
  REGISTER_COPY_BENCH_FOR_SIZE_CLUSTER(func, 4, suffix);    \
  REGISTER_COPY_BENCH_FOR_SIZE_CLUSTER(func, 8, suffix)

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// D2D (same device) benchmarks - warp groups
REGISTER_COPY_BENCH_ALL_SIZES(d2dCopyKernel, SyncScope::WARP, warp);

// P2P (cross device) benchmarks - warp groups
REGISTER_COPY_BENCH_ALL_SIZES(p2pCopyKernel, SyncScope::WARP, warp);

// D2D (same device) benchmarks - multiwarp groups
REGISTER_COPY_BENCH_ALL_SIZES(d2dCopyKernel, SyncScope::MULTIWARP, multiwarp);

// P2P (cross device) benchmarks - multiwarp groups
REGISTER_COPY_BENCH_ALL_SIZES(p2pCopyKernel, SyncScope::MULTIWARP, multiwarp);

// D2D (same device) benchmarks - block groups
REGISTER_COPY_BENCH_ALL_SIZES(d2dCopyKernel, SyncScope::BLOCK, block);

// P2P (cross device) benchmarks - block groups
REGISTER_COPY_BENCH_ALL_SIZES(p2pCopyKernel, SyncScope::BLOCK, block);

// D2D (same device) benchmarks - cluster groups (2 blocks per cluster)
REGISTER_COPY_BENCH_ALL_SIZES_CLUSTER(d2dCopyKernelCluster, cluster);

// P2P (cross device) benchmarks - cluster groups (2 blocks per cluster)
REGISTER_COPY_BENCH_ALL_SIZES_CLUSTER(p2pCopyKernelCluster, cluster);

//------------------------------------------------------------------------------
// Dual-Dest Benchmark Functions
// Compare sequential 2x memcpy vs 1x dual-dest memcpy
//------------------------------------------------------------------------------

/**
 * Shared implementation for dual-dest benchmarks. Allocates src, dst1, dst2
 * on device 0 and launches the given kernel for timing.
 *
 * @param kernel Kernel function pointer (sequentialCopyKernel or
 *               dualDestCopyKernel)
 */
static void d2dDualDestBenchImpl(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    SyncScope groupScope,
    folly::UserCounters& counters,
    int clusterSize,
    void* kernel,
    int hbmMultiplier) {
  const int nRunsPerIter = 50;

  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CudaBenchBase bench;

  DeviceBuffer srcBuffer(nBytes);
  DeviceBuffer dst1Buffer(nBytes);
  DeviceBuffer dst2Buffer(nBytes);

  char* srcPtr = static_cast<char*>(srcBuffer.get());
  char* dst1Ptr = static_cast<char*>(dst1Buffer.get());
  char* dst2Ptr = static_cast<char*>(dst2Buffer.get());

  float totalTimeMs = 0.0f;
  const int nThreads = 256;
  for (uint32_t i = 0; i < iters; ++i) {
    bench.startTiming();

    void* kernArgs[6] = {
        (void*)&dst1Ptr,
        (void*)&dst2Ptr,
        (void*)&srcPtr,
        (void*)&nBytes,
        (void*)&nRunsPerIter,
        (void*)&groupScope};
    dim3 grid{static_cast<unsigned int>(nBlocks), 1, 1};
    dim3 blocks{nThreads, 1, 1};

    std::optional<dim3> clusterDimOpt =
        (groupScope == SyncScope::CLUSTER && clusterSize > 1)
        ? std::optional{dim3(clusterSize, 1, 1)}
        : std::nullopt;
    CHECK_EQ(
        comms::common::launchKernel(
            kernel, grid, blocks, kernArgs, bench.stream, clusterDimOpt),
        cudaSuccess);

    bench.stopTiming();
    totalTimeMs += bench.measureTime();
  }

  populateBenchCounters(
      counters,
      totalTimeMs,
      iters,
      nRunsPerIter,
      nBytes,
      nBlocks,
      nThreads,
      groupScope,
      clusterSize,
      hbmMultiplier);
}

/**
 * Benchmark sequential copy: two separate memcpy_vectorized calls
 * (src->dst1 then src->dst2). HBM traffic: 2 reads + 2 writes = 4x nBytes.
 */
static void d2dSequentialCopy(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    SyncScope groupScope,
    folly::UserCounters& counters,
    int clusterSize = 1) {
  d2dDualDestBenchImpl(
      iters,
      nBytes,
      nBlocks,
      groupScope,
      counters,
      clusterSize,
      (void*)sequentialCopyKernel,
      /*hbmMultiplier=*/4);
}

/**
 * Benchmark dual-dest copy: single memcpy_vectorized_multi_dest<2> call
 * (src->dst1+dst2). HBM traffic: 1 read + 2 writes = 3x nBytes.
 */
static void d2dDualDestCopy(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    SyncScope groupScope,
    folly::UserCounters& counters,
    int clusterSize = 1) {
  d2dDualDestBenchImpl(
      iters,
      nBytes,
      nBlocks,
      groupScope,
      counters,
      clusterSize,
      (void*)dualDestCopyKernel,
      /*hbmMultiplier=*/3);
}

//------------------------------------------------------------------------------
// Dual-Dest Cluster Benchmark Wrapper Functions
//------------------------------------------------------------------------------

static void d2dSequentialCopyCluster(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    folly::UserCounters& counters) {
  d2dSequentialCopy(
      iters,
      nBytes,
      nBlocks,
      SyncScope::CLUSTER,
      counters,
      comms::common::kDefaultClusterSize);
}

static void d2dDualDestCopyCluster(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    folly::UserCounters& counters) {
  d2dDualDestCopy(
      iters,
      nBytes,
      nBlocks,
      SyncScope::CLUSTER,
      counters,
      comms::common::kDefaultClusterSize);
}

//------------------------------------------------------------------------------
// Dual-Dest Benchmark Registration
//------------------------------------------------------------------------------

// D2D sequential copy benchmarks - warp groups
REGISTER_COPY_BENCH_ALL_SIZES(d2dSequentialCopy, SyncScope::WARP, warp);

// D2D dual-dest copy benchmarks - warp groups
REGISTER_COPY_BENCH_ALL_SIZES(d2dDualDestCopy, SyncScope::WARP, warp);

// D2D sequential copy benchmarks - multiwarp groups
REGISTER_COPY_BENCH_ALL_SIZES(
    d2dSequentialCopy,
    SyncScope::MULTIWARP,
    multiwarp);

// D2D dual-dest copy benchmarks - multiwarp groups
REGISTER_COPY_BENCH_ALL_SIZES(d2dDualDestCopy, SyncScope::MULTIWARP, multiwarp);

// D2D sequential copy benchmarks - block groups
REGISTER_COPY_BENCH_ALL_SIZES(d2dSequentialCopy, SyncScope::BLOCK, block);

// D2D dual-dest copy benchmarks - block groups
REGISTER_COPY_BENCH_ALL_SIZES(d2dDualDestCopy, SyncScope::BLOCK, block);

// D2D sequential copy benchmarks - cluster groups
REGISTER_COPY_BENCH_ALL_SIZES_CLUSTER(d2dSequentialCopyCluster, cluster);

// D2D dual-dest copy benchmarks - cluster groups
REGISTER_COPY_BENCH_ALL_SIZES_CLUSTER(d2dDualDestCopyCluster, cluster);

//------------------------------------------------------------------------------
// Tri-Dest Benchmark Functions
// Compare sequential 3x memcpy vs 1x tri-dest memcpy
//------------------------------------------------------------------------------

/**
 * Shared implementation for tri-dest benchmarks. Allocates src, dst1, dst2,
 * dst3 on device 0 and launches the given kernel for timing.
 *
 * @param kernel Kernel function pointer (sequentialTriCopyKernel or
 *               triDestCopyKernel)
 */
static void d2dTriDestBenchImpl(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    SyncScope groupScope,
    folly::UserCounters& counters,
    int clusterSize,
    void* kernel,
    int hbmMultiplier) {
  const int nRunsPerIter = 50;

  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CudaBenchBase bench;

  DeviceBuffer srcBuffer(nBytes);
  DeviceBuffer dst1Buffer(nBytes);
  DeviceBuffer dst2Buffer(nBytes);
  DeviceBuffer dst3Buffer(nBytes);

  char* srcPtr = static_cast<char*>(srcBuffer.get());
  char* dst1Ptr = static_cast<char*>(dst1Buffer.get());
  char* dst2Ptr = static_cast<char*>(dst2Buffer.get());
  char* dst3Ptr = static_cast<char*>(dst3Buffer.get());

  float totalTimeMs = 0.0f;
  const int nThreads = 256;
  for (uint32_t i = 0; i < iters; ++i) {
    bench.startTiming();

    void* kernArgs[7] = {
        (void*)&dst1Ptr,
        (void*)&dst2Ptr,
        (void*)&dst3Ptr,
        (void*)&srcPtr,
        (void*)&nBytes,
        (void*)&nRunsPerIter,
        (void*)&groupScope};
    dim3 grid{static_cast<unsigned int>(nBlocks), 1, 1};
    dim3 blocks{nThreads, 1, 1};

    std::optional<dim3> clusterDimOpt =
        (groupScope == SyncScope::CLUSTER && clusterSize > 1)
        ? std::optional{dim3(clusterSize, 1, 1)}
        : std::nullopt;
    CHECK_EQ(
        comms::common::launchKernel(
            kernel, grid, blocks, kernArgs, bench.stream, clusterDimOpt),
        cudaSuccess);

    bench.stopTiming();
    totalTimeMs += bench.measureTime();
  }

  populateBenchCounters(
      counters,
      totalTimeMs,
      iters,
      nRunsPerIter,
      nBytes,
      nBlocks,
      nThreads,
      groupScope,
      clusterSize,
      hbmMultiplier);
}

/**
 * Benchmark sequential tri-copy: three separate memcpy_vectorized calls
 * (src->dst1 then src->dst2 then src->dst3). HBM traffic: 3 reads + 3 writes =
 * 6x nBytes.
 */
static void d2dSequentialTriCopy(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    SyncScope groupScope,
    folly::UserCounters& counters,
    int clusterSize = 1) {
  d2dTriDestBenchImpl(
      iters,
      nBytes,
      nBlocks,
      groupScope,
      counters,
      clusterSize,
      (void*)sequentialTriCopyKernel,
      /*hbmMultiplier=*/6);
}

/**
 * Benchmark tri-dest copy: single memcpy_vectorized_multi_dest<3> call
 * (src->dst1+dst2+dst3). HBM traffic: 1 read + 3 writes = 4x nBytes.
 */
static void d2dTriDestCopy(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    SyncScope groupScope,
    folly::UserCounters& counters,
    int clusterSize = 1) {
  d2dTriDestBenchImpl(
      iters,
      nBytes,
      nBlocks,
      groupScope,
      counters,
      clusterSize,
      (void*)triDestCopyKernel,
      /*hbmMultiplier=*/4);
}

//------------------------------------------------------------------------------
// Tri-Dest Cluster Benchmark Wrapper Functions
//------------------------------------------------------------------------------

static void d2dSequentialTriCopyCluster(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    folly::UserCounters& counters) {
  d2dSequentialTriCopy(
      iters,
      nBytes,
      nBlocks,
      SyncScope::CLUSTER,
      counters,
      comms::common::kDefaultClusterSize);
}

static void d2dTriDestCopyCluster(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    folly::UserCounters& counters) {
  d2dTriDestCopy(
      iters,
      nBytes,
      nBlocks,
      SyncScope::CLUSTER,
      counters,
      comms::common::kDefaultClusterSize);
}

//------------------------------------------------------------------------------
// Tri-Dest Benchmark Registration
//------------------------------------------------------------------------------

// D2D sequential tri-copy benchmarks - warp groups
REGISTER_COPY_BENCH_ALL_SIZES(d2dSequentialTriCopy, SyncScope::WARP, warp);

// D2D tri-dest copy benchmarks - warp groups
REGISTER_COPY_BENCH_ALL_SIZES(d2dTriDestCopy, SyncScope::WARP, warp);

// D2D sequential tri-copy benchmarks - multiwarp groups
REGISTER_COPY_BENCH_ALL_SIZES(
    d2dSequentialTriCopy,
    SyncScope::MULTIWARP,
    multiwarp);

// D2D tri-dest copy benchmarks - multiwarp groups
REGISTER_COPY_BENCH_ALL_SIZES(d2dTriDestCopy, SyncScope::MULTIWARP, multiwarp);

// D2D sequential tri-copy benchmarks - block groups
REGISTER_COPY_BENCH_ALL_SIZES(d2dSequentialTriCopy, SyncScope::BLOCK, block);

// D2D tri-dest copy benchmarks - block groups
REGISTER_COPY_BENCH_ALL_SIZES(d2dTriDestCopy, SyncScope::BLOCK, block);

// D2D sequential tri-copy benchmarks - cluster groups
REGISTER_COPY_BENCH_ALL_SIZES_CLUSTER(d2dSequentialTriCopyCluster, cluster);

// D2D tri-dest copy benchmarks - cluster groups
REGISTER_COPY_BENCH_ALL_SIZES_CLUSTER(d2dTriDestCopyCluster, cluster);

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
