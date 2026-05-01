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

namespace {

// Common harness for the multi-writer race test. Allocates buffers,
// launches `multiWriterTailRaceKernel<T, Unroll16>`, and verifies that
// every `out[i]` matches `src[i]`. A mismatch means phase 2 read a slot
// phase 1 had not yet committed (block 0 was sleeping); the stale value
// is the `0xCD` sentinel `buf` was initialized to.
template <typename T, int Unroll16>
void runMultiWriterRaceTest(
    size_t count,
    int gridDim,
    int blockDim,
    unsigned long long block0DelayNs) {
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

  dim3 grid{static_cast<unsigned>(gridDim), 1, 1};
  dim3 block{static_cast<unsigned>(blockDim), 1, 1};
  size_t countLocal = count;
  unsigned long long delayLocal = block0DelayNs;
  void* args[] = {
      &bufDev, &outDev, &srcDev, &countLocal, &flagDev, &delayLocal};
  CUDACHECK_TEST(cudaLaunchKernel(
      reinterpret_cast<void*>(multiWriterTailRaceKernel<T, Unroll16>),
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
      << " — phase-2 reader read a slot phase-1 writer had not yet committed"
      << " (Unroll16=" << Unroll16 << ")";

  CUDACHECK_TEST(cudaFree(srcDev));
  CUDACHECK_TEST(cudaFree(bufDev));
  CUDACHECK_TEST(cudaFree(outDev));
  CUDACHECK_TEST(cudaFree(flagDev));
}

// 50 ms is well above any plausible time another block needs to reach
// phase 2 once block 0 has started its delay loop.
constexpr unsigned long long kBlock0DelayNs = 50ULL * 1000ULL * 1000ULL;

} // namespace

class LocalReduceTailRaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }
};

// Matched-unroll (production) configuration.
//
// count=2200, blockDim=128, gridDim=8, fp32, Unroll16=4:
// - copyUnroll<4>-tail designated CTA = (2048/2048) % 8 = 1, so block 1
//   reads ALL of [2048, 2200) in phase 2.
// - Pre-fix localReduceVectorized-tail: block 0 writes [2048, 2176),
//   block 1 writes [2176, 2200) (grid-strided over linearThreadId).
// - Post-fix: block 1 writes [2048, 2200) (single-CTA tail mirrors
//   copyUnroll's predicate); block 1's phase-2 read returns block 1's
//   own phase-1 write regardless of block 0's progress.
TEST_F(LocalReduceTailRaceTest, DelayedBlockZeroExposesBlockOneStaleRead) {
  runMultiWriterRaceTest<int32_t, /*Unroll16=*/4>(
      /*count=*/2200,
      /*gridDim=*/8,
      /*blockDim=*/128,
      kBlock0DelayNs);
}

// Mismatched-unroll configuration.
//
// `localReduceVectorized` hardcodes `kUnroll=4`; `copyUnroll` is templated
// on `Unroll16` and EVERY in-tree call site instantiates with `Unroll16=4`
// (`copy<T>`, `ctranKernCopyRaw`, `ctranKernCopyMultiDestRaw`). The match
// is by convention only — nothing prevents a future caller from using a
// different unroll factor. This test demonstrates what happens if that
// happens: at perfectly aligned counts (no tail at all on either side),
// the per-CTA partitions in the MAIN LOOP disagree, so cross-CTA stale
// reads still occur once the writer is delayed.
//
// count=2200, blockDim=128, gridDim=8, fp32, Unroll16=2:
// - localReduceVectorized<4>: numPerBlock = 128*(16/4)*4 = 2048. Block 0
//   owns [0, 2048) in main loop; tail [2048, 2200) per the post-fix
//   single-CTA pattern (designated = 1).
// - copyUnroll<2>: numPerBlock = 128*(16/4)*2 = 1024. Block 0 owns
//   [0, 1024); block 1 owns [1024, 2048); block 2 owns the tail
//   [2048, 2200). Reading position 1024+ requires block 1 (and block 2)
//   to observe block 0's phase-1 write — but block 0 is sleeping.
//
// Marked DISABLED so CI stays green — the failure here motivates the
// next diff in this stack, which adds a single shared unroll constant
// plus a `static_assert` so the mismatched instantiation cannot compile.
TEST_F(
    LocalReduceTailRaceTest,
    DISABLED_DifferentUnrollFromLocalReduceExposesStaleRead) {
  runMultiWriterRaceTest<int32_t, /*Unroll16=*/2>(
      /*count=*/2200,
      /*gridDim=*/8,
      /*blockDim=*/128,
      kBlock0DelayNs);
}
