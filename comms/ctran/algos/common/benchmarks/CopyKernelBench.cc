// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/testinfra/TestXPlatUtils.h"

using namespace ctran;
using ctran::utils::CtranIpcDesc;
using ctran::utils::CtranIpcMem;
using ctran::utils::CtranIpcRemMem;

//------------------------------------------------------------------------------
// External Kernel Declaration
//------------------------------------------------------------------------------

template <typename T>
__global__ void
copyKernel(const T* sendbuff, T* recvbuff, size_t count, int nRuns);

template <typename T>
__global__ void copyKernel2Dst(
    const T* sendbuff,
    T* recvbuff1,
    T* recvbuff2,
    size_t count,
    int nRuns);

//------------------------------------------------------------------------------
// Common Helper Functions
//------------------------------------------------------------------------------

namespace {
class CopyBenchSetup : public CudaBenchBase {
 public:
  CtranIpcDesc ipcDesc;
  CopyBenchSetup(int cudaDev, size_t nBytes = 0) : cudaDev_(cudaDev) {
    if (nBytes == 0) {
      return;
    }
    CHECK_EQ(cudaSetDevice(cudaDev_), cudaSuccess);
    ipcMem_ = std::make_unique<CtranIpcMem>(
        nBytes, cudaDev_, &dummyLogMetaData_, "Benchmark");
    CHECK_EQ(ipcMem_->ipcExport(ipcDesc), commSuccess);
  }

  void* importRemoteDeviceSyncPtr(const CtranIpcDesc& remoteIpcDesc) {
    CHECK_EQ(cudaSetDevice(cudaDev_), cudaSuccess);
    ipcRemMem_ = std::make_unique<CtranIpcRemMem>(
        remoteIpcDesc, cudaDev_, &dummyLogMetaData_, "Benchmark");
    return ipcRemMem_->getBase();
  }

  ~CopyBenchSetup() {
    CHECK_EQ(cudaSetDevice(cudaDev_), cudaSuccess);
    if (ipcRemMem_) {
      CHECK_EQ(ipcRemMem_->release(), commSuccess);
    }
    if (ipcMem_) {
      CHECK_EQ(ipcMem_->free(), commSuccess);
    }
  }

 private:
  int cudaDev_;
  std::unique_ptr<CtranIpcMem> ipcMem_;
  std::unique_ptr<CtranIpcRemMem> ipcRemMem_;
  const struct CommLogData dummyLogMetaData_ = {
      0,
      0xfaceb00c12345678 /*Dummy placeholder value for commHash*/,
      "BenchComm",
      0,
      0};
};

} // anonymous namespace

//------------------------------------------------------------------------------
// Benchmark Functions
//------------------------------------------------------------------------------

/**
 * Benchmark P2P copyKernel (copy from and to different devices) with varying
 * message size and number of groups: the copy is done in 1 step (IPC buffer is
 * the same size as copy data size).
 */
static void p2pCopyKernel(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    folly::UserCounters& counters) {
  const int nRunsPerIter = 50;

  const int senderCudaDev = 0;
  CHECK_EQ(cudaSetDevice(senderCudaDev), cudaSuccess);
  CopyBenchSetup senderBench(senderCudaDev);
  using T = uint8_t;
  const size_t count = nBytes / sizeof(T);
  void* srcPtr = nullptr;
  CHECK_EQ(cudaMalloc(&srcPtr, nBytes), cudaSuccess);

  const int receiverCudaDev = 1;
  CHECK_EQ(cudaSetDevice(receiverCudaDev), cudaSuccess);
  CopyBenchSetup receiverBench(receiverCudaDev, nBytes);

  CHECK_EQ(cudaSetDevice(senderCudaDev), cudaSuccess);
  auto dstPtr = senderBench.importRemoteDeviceSyncPtr(receiverBench.ipcDesc);

  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Start timing the kernel
    senderBench.startTiming();

    {
      void* kernArgs[4] = {
          (void*)&srcPtr, (void*)&dstPtr, (void*)&count, (void*)&nRunsPerIter};
      dim3 grid = {(unsigned int)nBlocks, 1, 1};
      dim3 blocks = {256, 1, 1};
      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)copyKernel<T>,
              grid,
              blocks,
              kernArgs,
              sizeof(CtranAlgoDeviceState), // Dynamic shared memory size
              senderBench.stream),
          cudaSuccess);
    }

    // Stop timing and measure
    senderBench.stopTiming();
    totalTimeMs += senderBench.measureTime();
  }

  float avgTimeUs =
      (totalTimeMs / iters / nRunsPerIter) * 1000.0f; // Convert ms to us
  float busBwGBps =
      (nBytes / 1e9f) / (avgTimeUs / 1e6f); // GB/s = bytes / time_in_seconds
  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["busBwGBps"] =
      folly::UserMetric(busBwGBps, folly::UserMetric::Type::METRIC);
  counters["nBlocks"] =
      folly::UserMetric(nBlocks, folly::UserMetric::Type::METRIC);

  CHECK_EQ(cudaFree(srcPtr), cudaSuccess);
}

/**
 * Benchmark D2D copyKernel (copy from and to the same device) with varying
 * message size and number of groups
 */
static void d2dCopyKernel(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    folly::UserCounters& counters) {
  const int nRunsPerIter = 50;

  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CudaBenchBase bench;
  using T = uint8_t;
  const size_t count = nBytes / sizeof(T);
  void* srcPtr = nullptr;
  void* dstPtr = nullptr;
  CHECK_EQ(cudaMalloc(&srcPtr, nBytes), cudaSuccess);
  CHECK_EQ(cudaMalloc(&dstPtr, nBytes), cudaSuccess);
  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    // Start timing the kernel
    bench.startTiming();

    {
      void* kernArgs[4] = {
          (void*)&srcPtr, (void*)&dstPtr, (void*)&count, (void*)&nRunsPerIter};
      dim3 grid = {(unsigned int)nBlocks, 1, 1};
      dim3 blocks = {256, 1, 1};
      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)copyKernel<T>,
              grid,
              blocks,
              kernArgs,
              sizeof(CtranAlgoDeviceState), // Dynamic shared memory size
              bench.stream),
          cudaSuccess);
    }

    // Stop timing and measure
    bench.stopTiming();
    totalTimeMs += bench.measureTime();
  }

  float avgTimeUs =
      (totalTimeMs / iters / nRunsPerIter) * 1000.0f; // Convert ms to us
  float busBwGBps =
      (nBytes / 1e9f) / (avgTimeUs / 1e6f); // GB/s = bytes / time_in_seconds
  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["busBwGBps"] =
      folly::UserMetric(busBwGBps, folly::UserMetric::Type::METRIC);
  counters["nBlocks"] =
      folly::UserMetric(nBlocks, folly::UserMetric::Type::METRIC);

  CHECK_EQ(cudaFree(srcPtr), cudaSuccess);
  CHECK_EQ(cudaFree(dstPtr), cudaSuccess);
}

/**
 * Benchmark D2D copyKernel2Dst (dual-destination copy on the same device) with
 * varying message size and number of groups
 */
static void d2dCopyKernel2Dst(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    folly::UserCounters& counters) {
  const int nRunsPerIter = 50;

  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CudaBenchBase bench;
  using T = uint8_t;
  const size_t count = nBytes / sizeof(T);
  void* srcPtr = nullptr;
  void* dst1Ptr = nullptr;
  void* dst2Ptr = nullptr;
  CHECK_EQ(cudaMalloc(&srcPtr, nBytes), cudaSuccess);
  CHECK_EQ(cudaMalloc(&dst1Ptr, nBytes), cudaSuccess);
  CHECK_EQ(cudaMalloc(&dst2Ptr, nBytes), cudaSuccess);
  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    bench.startTiming();

    {
      void* kernArgs[5] = {
          (void*)&srcPtr,
          (void*)&dst1Ptr,
          (void*)&dst2Ptr,
          (void*)&count,
          (void*)&nRunsPerIter};
      dim3 grid = {(unsigned int)nBlocks, 1, 1};
      dim3 blocks = {256, 1, 1};
      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)copyKernel2Dst<T>,
              grid,
              blocks,
              kernArgs,
              sizeof(CtranAlgoDeviceState),
              bench.stream),
          cudaSuccess);
    }

    bench.stopTiming();
    totalTimeMs += bench.measureTime();
  }

  float avgTimeUs =
      (totalTimeMs / iters / nRunsPerIter) * 1000.0f; // Convert ms to us
  float busBwGBps =
      (nBytes / 1e9f) / (avgTimeUs / 1e6f); // GB/s = bytes / time_in_seconds
  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["busBwGBps"] =
      folly::UserMetric(busBwGBps, folly::UserMetric::Type::METRIC);
  counters["nBlocks"] =
      folly::UserMetric(nBlocks, folly::UserMetric::Type::METRIC);

  CHECK_EQ(cudaFree(srcPtr), cudaSuccess);
  CHECK_EQ(cudaFree(dst1Ptr), cudaSuccess);
  CHECK_EQ(cudaFree(dst2Ptr), cudaSuccess);
}

/**
 * Benchmark P2P copyKernel2Dst (dual-destination copy across devices) with
 * varying message size and number of groups: src and dst1 are local on GPU 0,
 * dst2 is an IPC-imported remote buffer on GPU 1.
 */
static void p2pCopyKernel2Dst(
    uint32_t iters,
    size_t nBytes,
    int nBlocks,
    folly::UserCounters& counters) {
  const int nRunsPerIter = 50;

  const int senderCudaDev = 0;
  CHECK_EQ(cudaSetDevice(senderCudaDev), cudaSuccess);
  CopyBenchSetup senderBench(senderCudaDev);
  using T = uint8_t;
  const size_t count = nBytes / sizeof(T);
  void* srcPtr = nullptr;
  void* dst1Ptr = nullptr;
  CHECK_EQ(cudaMalloc(&srcPtr, nBytes), cudaSuccess);
  CHECK_EQ(cudaMalloc(&dst1Ptr, nBytes), cudaSuccess);

  const int receiverCudaDev = 1;
  CHECK_EQ(cudaSetDevice(receiverCudaDev), cudaSuccess);
  CopyBenchSetup receiverBench(receiverCudaDev, nBytes);

  CHECK_EQ(cudaSetDevice(senderCudaDev), cudaSuccess);
  auto dst2Ptr = senderBench.importRemoteDeviceSyncPtr(receiverBench.ipcDesc);

  float totalTimeMs = 0.0f;

  for (uint32_t i = 0; i < iters; ++i) {
    senderBench.startTiming();

    {
      void* kernArgs[5] = {
          (void*)&srcPtr,
          (void*)&dst1Ptr,
          (void*)&dst2Ptr,
          (void*)&count,
          (void*)&nRunsPerIter};
      dim3 grid = {(unsigned int)nBlocks, 1, 1};
      dim3 blocks = {256, 1, 1};
      CHECK_EQ(
          cudaLaunchKernel(
              (const void*)copyKernel2Dst<T>,
              grid,
              blocks,
              kernArgs,
              sizeof(CtranAlgoDeviceState),
              senderBench.stream),
          cudaSuccess);
    }

    senderBench.stopTiming();
    totalTimeMs += senderBench.measureTime();
  }

  float avgTimeUs =
      (totalTimeMs / iters / nRunsPerIter) * 1000.0f; // Convert ms to us
  float busBwGBps =
      (nBytes / 1e9f) / (avgTimeUs / 1e6f); // GB/s = bytes / time_in_seconds
  counters["deviceTimeUs"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
  counters["busBwGBps"] =
      folly::UserMetric(busBwGBps, folly::UserMetric::Type::METRIC);
  counters["nBlocks"] =
      folly::UserMetric(nBlocks, folly::UserMetric::Type::METRIC);

  CHECK_EQ(cudaFree(srcPtr), cudaSuccess);
  CHECK_EQ(cudaFree(dst1Ptr), cudaSuccess);
}

//------------------------------------------------------------------------------
// Benchmark Registration Helper Macros
//------------------------------------------------------------------------------

// Helper macro to register benchmarks for all group counts for a given size
#define REGISTER_COPY_BENCH_FOR_SIZE(func, sizeMB)     \
  BENCHMARK_MULTI_PARAM_COUNTERS(                      \
      func, sizeMB##MB_1b, sizeMB * 1024 * 1024, 1);   \
  BENCHMARK_MULTI_PARAM_COUNTERS(                      \
      func, sizeMB##MB_2b, sizeMB * 1024 * 1024, 2);   \
  BENCHMARK_MULTI_PARAM_COUNTERS(                      \
      func, sizeMB##MB_4b, sizeMB * 1024 * 1024, 4);   \
  BENCHMARK_MULTI_PARAM_COUNTERS(                      \
      func, sizeMB##MB_8b, sizeMB * 1024 * 1024, 8);   \
  BENCHMARK_MULTI_PARAM_COUNTERS(                      \
      func, sizeMB##MB_16b, sizeMB * 1024 * 1024, 16); \
  BENCHMARK_MULTI_PARAM_COUNTERS(func, sizeMB##MB_32b, sizeMB * 1024 * 1024, 32)

// Helper macro to register benchmarks for all standard sizes
#define REGISTER_COPY_BENCH_ALL_SIZES(func) \
  REGISTER_COPY_BENCH_FOR_SIZE(func, 4);    \
  REGISTER_COPY_BENCH_FOR_SIZE(func, 8);    \
  REGISTER_COPY_BENCH_FOR_SIZE(func, 16);   \
  REGISTER_COPY_BENCH_FOR_SIZE(func, 32)

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// Register d2dCopyKernel benchmarks for all sizes (4MB, 8MB, 16MB, 32MB)
// and all group counts (1, 2, 4, 8, 16, 32)
REGISTER_COPY_BENCH_ALL_SIZES(d2dCopyKernel);

// Register p2pCopyKernel benchmarks for all sizes (4MB, 8MB, 16MB, 32MB)
// and all group counts (1, 2, 4, 8, 16, 32)
REGISTER_COPY_BENCH_ALL_SIZES(p2pCopyKernel);

// Register d2dCopyKernel2Dst benchmarks for all sizes (4MB, 8MB, 16MB, 32MB)
// and all group counts (1, 2, 4, 8, 16, 32)
REGISTER_COPY_BENCH_ALL_SIZES(d2dCopyKernel2Dst);

// Register p2pCopyKernel2Dst benchmarks for all sizes (4MB, 8MB, 16MB, 32MB)
// and all group counts (1, 2, 4, 8, 16, 32)
REGISTER_COPY_BENCH_ALL_SIZES(p2pCopyKernel2Dst);

int main(int argc, char** argv) {
  CHECK_GE(bench_utils::getNumCudaDevices(), 2);

  // Initialize CUDA driver library to load cuMem* functions for IPC support
  CHECK_EQ(ctran::utils::commCudaLibraryInit(), commSuccess);
  CHECK_EQ(ctran::utils::CtranIpcSupport(), true);

  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  // Cleanup
  cudaSetDevice(0);
  cudaDeviceReset();
  cudaSetDevice(1);
  cudaDeviceReset();

  return 0;
}
