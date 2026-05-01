// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// GPU reproducer for the per-CTA byte-ownership invariant violation
// between `copyUnroll<4, T>` (`fbcode/comms/ctran/algos/DevCommon.cuh`)
// and `localReduceVectorized` (`fbcode/comms/ctran/algos/localReduce.cuh`).
//
// D69774173 gave `copyUnroll`'s tail a single
// designated CTA so per-byte ownership is stable across calls on the
// same buffer. The same fix was never applied to `localReduce.cuh`.
//
// In ring AllReduce the bug surfaces at the RS→AG transition: RS-last-step
// writes `recvbuff` (and `tmpSendBuf`) via `localReduce` (line 121); AG
// steps touch the same buffers via `copyUnroll`-via-`ctranKernCopy*`
// (lines 127, 137, 268, 278). Inter-round sync is per-CTA only via
// `GpeKernelSync.completeFlag[blockIdx]`. When per-CTA ownership of two
// writers disagrees, a CTA in the next round can read or overwrite bytes
// a different CTA in the prior round hasn't yet committed.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "comms/ctran/algos/tests/LocalReduceTailRaceUTKernels.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

class LocalReduceTailRaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }
};

TEST_F(
    LocalReduceTailRaceTest,
    DISABLED_DelayedBlockZeroExposesBlockOneStaleRead) {
  // count=2200, blockDim=128, gridDim=8, fp32:
  // - copyUnroll-tail designated CTA = (2048/2048) % 8 = 1, so block 1
  //   reads ALL of [2048, 2200) in phase 2.
  // - Pre-fix localReduceVectorized-tail: block 0 writes [2048, 2176),
  //   block 1 writes [2176, 2200) (grid-strided over linearThreadId).
  // - Post-fix: block 1 writes [2048, 2200) (single-CTA tail mirrors
  //   copyUnroll's predicate).
  constexpr size_t count = 2200;
  constexpr int gridDim = 8;
  constexpr int blockDim = 128;
  // 50 ms is well above any plausible time block 1 needs to reach phase 2.
  constexpr unsigned long long block0DelayNs = 50ULL * 1000ULL * 1000ULL;
  using T = int32_t;
  const size_t bytes = count * sizeof(T);

  std::vector<T> srcHost(count);
  for (size_t i = 0; i < count; ++i) {
    srcHost[i] = static_cast<T>(i + 1);
  }

  T* srcDev{nullptr};
  T* bufDev{nullptr};
  T* outDev{nullptr};
  int* flagDev{nullptr};

  CUDACHECK_TEST(cudaMalloc(&srcDev, bytes));
  CUDACHECK_TEST(cudaMalloc(&bufDev, bytes));
  CUDACHECK_TEST(cudaMalloc(&outDev, bytes));
  CUDACHECK_TEST(cudaMalloc(&flagDev, gridDim * sizeof(int)));

  CUDACHECK_TEST(
      cudaMemcpy(srcDev, srcHost.data(), bytes, cudaMemcpyHostToDevice));
  // Sentinel for `buf`. Stale phase-2 reads will return this value
  // (broadcasts to T=int32 as 0xCDCDCDCD).
  CUDACHECK_TEST(cudaMemset(bufDev, 0xCD, bytes));
  CUDACHECK_TEST(cudaMemset(outDev, 0, bytes));
  CUDACHECK_TEST(cudaMemset(flagDev, 0, gridDim * sizeof(int)));

  dim3 grid{gridDim, 1, 1};
  dim3 block{blockDim, 1, 1};
  size_t countLocal = count;
  unsigned long long delayLocal = block0DelayNs;
  void* args[] = {
      &bufDev, &outDev, &srcDev, &countLocal, &flagDev, &delayLocal};
  CUDACHECK_TEST(cudaLaunchKernel(
      reinterpret_cast<void*>(multiWriterTailRaceKernel<T>),
      grid,
      block,
      args));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<T> outHost(count);
  CUDACHECK_TEST(
      cudaMemcpy(outHost.data(), outDev, bytes, cudaMemcpyDeviceToHost));

  size_t firstStale = count;
  for (size_t i = 0; i < count; ++i) {
    if (outHost[i] != srcHost[i]) {
      firstStale = i;
      break;
    }
  }

  EXPECT_EQ(firstStale, count)
      << "stale read at out[" << firstStale
      << "] = " << static_cast<uint32_t>(outHost[firstStale]) << " (expected "
      << static_cast<uint32_t>(srcHost[firstStale]) << ")"
      << " — block 1 read a slot block 0 had not yet written in phase 1";

  CUDACHECK_TEST(cudaFree(srcDev));
  CUDACHECK_TEST(cudaFree(bufDev));
  CUDACHECK_TEST(cudaFree(outDev));
  CUDACHECK_TEST(cudaFree(flagDev));
}
