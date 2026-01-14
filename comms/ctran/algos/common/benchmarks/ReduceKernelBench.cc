// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "comms/ctran/algos/common/benchmarks/ReduceKernelBench.h"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/commSpecs.h"

//------------------------------------------------------------------------------
// External Kernel Declaration
//------------------------------------------------------------------------------

template <typename T, commRedOp_t redOp>
extern __global__ void LocalReduceKernel(ReduceKernelBenchArg arg, int iters);

//------------------------------------------------------------------------------
// Common Helper Functions
//------------------------------------------------------------------------------

namespace {

class ReduceKernelBenchSetup : public CudaBenchBase {
 public:
  ReduceKernelBenchArg arg{};

  ReduceKernelBenchSetup(
      int cudaDev,
      size_t count,
      int nGroups,
      int nSrcs,
      int nDsts)
      : cudaDev_(cudaDev),
        count_(count),
        nGroups_(nGroups),
        nSrcs_(nSrcs),
        nDsts_(nDsts) {
    CUDACHECK_TEST(cudaSetDevice(cudaDev_));

    size_t bytesPerBuffer = count_ * sizeof(int);
    size_t totalSrcBytes = bytesPerBuffer * nSrcs_;
    size_t totalDstBytes = bytesPerBuffer * nDsts_;

    // Allocate source buffer on GPU
    CUDACHECK_TEST(cudaMalloc(&srcBuf_, totalSrcBytes));

    // Allocate destination buffer on GPU
    CUDACHECK_TEST(cudaMalloc(&dstBuf_, totalDstBytes));

    // Set up reduce arguments
    arg.count = count_;
    arg.nsrcs = nSrcs_;
    arg.ndsts = nDsts_;

    for (int i = 0; i < nSrcs_; i++) {
      arg.srcs[i] = reinterpret_cast<char*>(srcBuf_) + i * bytesPerBuffer;
    }
    for (int i = 0; i < nDsts_; i++) {
      arg.dsts[i] = reinterpret_cast<char*>(dstBuf_) + i * bytesPerBuffer;
    }
  }

  ~ReduceKernelBenchSetup() {
    CUDACHECK_TEST(cudaSetDevice(cudaDev_));
    if (srcBuf_) {
      CHECK_EQ(cudaFree(srcBuf_), cudaSuccess);
    }
    if (dstBuf_) {
      CHECK_EQ(cudaFree(dstBuf_), cudaSuccess);
    }
  }

 private:
  int cudaDev_;
  size_t count_;
  int nGroups_;
  int nSrcs_;
  int nDsts_;
  void* srcBuf_{nullptr};
  void* dstBuf_{nullptr};
};

// Template function pointers for different reduction operations
template <typename T, commRedOp_t redOp>
void* getReduceKernelFn() {
  return reinterpret_cast<void*>(LocalReduceKernel<T, redOp>);
}

} // anonymous namespace

//------------------------------------------------------------------------------
// Benchmark Functions
//------------------------------------------------------------------------------

/**
 * Benchmark reduce kernel with varying message sizes and reduction operations.
 *
 * Parameters:
 * - msgSizeMB: message size in megabytes
 * - nGroups: number of thread block groups
 * - nSrcs: number of source buffers
 * - nDsts: number of destination buffers
 * - op: reduction operation (sum or max)
 */
template <size_t msgSizeMB, int nGroups, int nSrcs, int nDsts, commRedOp_t op>
static void ReduceKernelPerf(uint32_t iters, folly::UserCounters& counters) {
  const int cudaDev = 0;
  const int innerIters = 50;
  constexpr size_t msgSizeBytes = msgSizeMB * 1024 * 1024;
  constexpr size_t count = msgSizeBytes / sizeof(int);

  CHECK_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  ReduceKernelBenchSetup bench(cudaDev, count, nGroups, nSrcs, nDsts);
  float totalTimeMs = 0.0f;

  void* fn = getReduceKernelFn<int, op>();

  dim3 grid = {(unsigned int)nGroups, 1, 1};
  dim3 blocks = {1024, 1, 1};

  for (uint32_t i = 0; i < iters; ++i) {
    bench.startTiming();

    {
      std::array<void*, 2> kernArgs;
      kernArgs[0] = &bench.arg;
      kernArgs[1] = (void*)&innerIters;

      CUDACHECK_TEST(
          cudaLaunchKernel(fn, grid, blocks, kernArgs.data(), 0, bench.stream));
    }

    bench.stopTiming();
    totalTimeMs += bench.measureTime();
  }

  // Calculate timing
  float avgTimeUs = (totalTimeMs / iters / innerIters) * 1000.0f;
  float avgTimeSec = avgTimeUs / 1e6;

  // Calculate process bandwidth (GB/s) - based on message size only
  float processBwGBs = (msgSizeBytes / 1e9) / avgTimeSec;

  // Report metrics
  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["processBwGBs"] =
      folly::UserMetric(processBwGBs, folly::UserMetric::Type::METRIC);
  counters["msgSizeMB"] =
      folly::UserMetric(msgSizeMB, folly::UserMetric::Type::METRIC);
  counters["nGroups"] =
      folly::UserMetric(nGroups, folly::UserMetric::Type::METRIC);
  counters["nSrcs"] = folly::UserMetric(nSrcs, folly::UserMetric::Type::METRIC);
  counters["nDsts"] = folly::UserMetric(nDsts, folly::UserMetric::Type::METRIC);
}

//------------------------------------------------------------------------------
// Benchmark Registration Macros
//------------------------------------------------------------------------------

// Macro to define and register a benchmark
// msgSizeMB: message size in megabytes
// nGroups: number of thread block groups
// nSrcs: number of source buffers
// nDsts: number of destination buffers
#define DEFINE_REDUCE_BENCH(msgSizeMB, nGroups, nSrcs, nDsts)                      \
  static void                                                                      \
  ReduceKernelPerf_##msgSizeMB##MB_##nGroups##g_##nSrcs##src_##nDsts##dst_Sum(     \
      uint32_t iters, folly::UserCounters& counters) {                             \
    ReduceKernelPerf<msgSizeMB, nGroups, nSrcs, nDsts, commSum>(                   \
        iters, counters);                                                          \
  }                                                                                \
  BENCHMARK_COUNTERS(                                                              \
      ReduceKernelPerf_##msgSizeMB##MB_##nGroups##g_##nSrcs##src_##nDsts##dst_Sum, \
      counters) {                                                                  \
    ReduceKernelPerf_##msgSizeMB##MB_##nGroups##g_##nSrcs##src_##nDsts##dst_Sum(   \
        1, counters);                                                              \
  }

// Macro to define all src/dst combinations for a given message size and groups
// Only 2 src, 1 dst and 2 src, 2 dst are needed
#define DEFINE_REDUCE_BENCH_ALL_SRCDST(msgSizeMB, nGroups) \
  DEFINE_REDUCE_BENCH(msgSizeMB, nGroups, 2, 1)            \
  DEFINE_REDUCE_BENCH(msgSizeMB, nGroups, 2, 2)

// Macro to define all group counts for a given message size
#define DEFINE_REDUCE_BENCH_ALL_GROUPS(msgSizeMB) \
  DEFINE_REDUCE_BENCH_ALL_SRCDST(msgSizeMB, 2)    \
  DEFINE_REDUCE_BENCH_ALL_SRCDST(msgSizeMB, 4)    \
  DEFINE_REDUCE_BENCH_ALL_SRCDST(msgSizeMB, 8)

//------------------------------------------------------------------------------
// Benchmark Definitions
//------------------------------------------------------------------------------

// 4MB message size with 2, 4, 8 groups
DEFINE_REDUCE_BENCH_ALL_GROUPS(4)

// 8MB message size with 2, 4, 8 groups
DEFINE_REDUCE_BENCH_ALL_GROUPS(8)

// 16MB message size with 2, 4, 8 groups
DEFINE_REDUCE_BENCH_ALL_GROUPS(16)

int main(int argc, char** argv) {
  CHECK_GE(bench_utils::getNumCudaDevices(), 1);

  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  // Cleanup
  cudaSetDevice(0);
  cudaDeviceReset();

  return 0;
}
