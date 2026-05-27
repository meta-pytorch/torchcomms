// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// GPU reproducer for the per-CTA byte-ownership invariant violation
// between `copyUnroll<4, T>` (`fbcode/comms/ctran/algos/DevCommon.cuh`)
// and `localReduce*` (`fbcode/comms/ctran/algos/localReduce.cuh`).
//
// Invariant: every byte of a shared buffer must map to the same CTA in
// both the writer and the next reader on that buffer. When that holds,
// each CTA's phase-2 read sees only its own phase-1 write — no
// cross-CTA dependency for memory visibility, so a per-CTA self-fence
// (release-store + acquire-load on the same CTA's own flag) is
// sufficient. D69774173 gave `copyUnroll`'s tail a single designated
// CTA so this invariant holds across calls on the same buffer. The
// matching fix was never applied to `localReduce.cuh`.
//
// In ring AllReduce the bug surfaces at the RS->AG transition: RS-last
// writes `recvbuff` (and `tmpSendBuf`) via `localReduce`; AG steps
// touch the same buffers via `copyUnroll`-via-`ctranKernCopy*`. Cross-
// round visibility is mediated by the GPE host thread via separate
// sync structs (`recvRedCopySync`, `revSendCopySync`, ...), not by per-
// CTA flags; but the host gate at `AllReduceRing.cc:689`
// (`progressRevSendCheckSendBuf`) only checks `revSendTrans.done`, not
// `recvRedCopy.isComplete`, so the host relay does not fully serialize
// writer and reader CTAs across this edge. When per-CTA byte ownership
// of the two ops disagrees, a CTA in the next round can read bytes a
// different CTA in the prior round hasn't yet committed.
//
// The test isolates the invariant: it uses no host relay at all and
// only intra-CTA fences (see `LocalReduceTailRaceUTKernels.cu`), so
// the byte-ownership invariant is the only thing standing between the
// two ops. Pre-fix the invariant fails; post-fix it holds.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <vector>
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/algos/tests/LocalReduceTailRaceUTKernels.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

class LocalReduceTailRaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }
};

namespace {

// Launches `multiWriterTailRaceKernel<int32_t, UseFallback>` with the
// production `GpeKernelSync` host-pinned allocation and pre-post, then
// asserts that every `out[i] == src[i]` (i.e. no stale 0xCD-init read
// reached phase 2).
template <bool UseFallback>
void runMultiWriterTailRaceTest(
    size_t count,
    int delayedBlock,
    const std::string& failureContext) {
  constexpr int gridDim = 8;
  constexpr int blockDim = 128;
  // 50 ms is well above any plausible time the other blocks need to
  // reach phase 2.
  constexpr unsigned long long delayNs = 50ULL * 1000ULL * 1000ULL;
  using T = int32_t;
  const size_t bytes = count * sizeof(T);

  std::vector<T> srcHost(count);
  for (size_t i = 0; i < count; ++i) {
    srcHost[i] = static_cast<T>(i + 1);
  }

  T* srcDev{nullptr};
  T* bufDev{nullptr};
  T* outDev{nullptr};

  CUDACHECK_TEST(cudaMalloc(&srcDev, bytes));
  CUDACHECK_TEST(cudaMalloc(&bufDev, bytes));
  CUDACHECK_TEST(cudaMalloc(&outDev, bytes));

  // Allocate the REAL `ctran::algos::GpeKernelSync` in pinned host memory
  // — same allocation pattern as production (and as D103255142). The
  // kernel uses `GpeKernelSyncDev::complete` / `waitPost` (the production
  // sync APIs) over `sync->completeFlag[]` / `sync->postFlag[]`. We
  // pre-post step=1 for every worker BEFORE launch so the kernel's
  // `waitPost` succeeds without a host poll loop — the only intentionally
  // race-relevant interaction is the per-CTA release/acquire pair the
  // kernel performs around the buf write/read.
  void* syncPtr = nullptr;
  CUDACHECK_TEST(cudaHostAlloc(
      &syncPtr, sizeof(ctran::algos::GpeKernelSync), cudaHostAllocDefault));
  auto* sync = new (syncPtr) ctran::algos::GpeKernelSync(gridDim);
  sync->post(/*step=*/1);

  CUDACHECK_TEST(
      cudaMemcpy(srcDev, srcHost.data(), bytes, cudaMemcpyHostToDevice));
  // Sentinel for `buf`. Stale phase-2 reads will return this value
  // (broadcasts to T=int32 as 0xCDCDCDCD).
  CUDACHECK_TEST(cudaMemset(bufDev, 0xCD, bytes));
  CUDACHECK_TEST(cudaMemset(outDev, 0, bytes));

  dim3 grid{gridDim, 1, 1};
  dim3 block{blockDim, 1, 1};
  size_t countLocal = count;
  int delayedLocal = delayedBlock;
  unsigned long long delayLocal = delayNs;
  void* args[] = {
      &bufDev,
      &outDev,
      &srcDev,
      &countLocal,
      &sync,
      &delayedLocal,
      &delayLocal};
  CUDACHECK_TEST(cudaLaunchKernel(
      reinterpret_cast<void*>(multiWriterTailRaceKernel<T, UseFallback>),
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
      << static_cast<uint32_t>(srcHost[firstStale]) << ") — " << failureContext;

  CUDACHECK_TEST(cudaFree(srcDev));
  CUDACHECK_TEST(cudaFree(bufDev));
  CUDACHECK_TEST(cudaFree(outDev));
  sync->~GpeKernelSync();
  CUDACHECK_TEST(cudaFreeHost(syncPtr));
}

} // namespace

TEST_F(LocalReduceTailRaceTest, DelayedBlockZeroExposesBlockOneStaleRead) {
  // count=2200, blockDim=128, gridDim=8, fp32 (vectorized path):
  // - copyUnroll-tail designated CTA = (2048/2048) % 8 = 1, so block 1
  //   reads ALL of [2048, 2200) in phase 2.
  // - Pre-fix localReduceVectorized-tail: block 0 writes [2048, 2176),
  //   block 1 writes [2176, 2200) (grid-strided over linearThreadId).
  //   Delaying block 0 makes block 1 read 0xCD in [2048, 2176).
  // - Post-fix: block 1 writes [2048, 2200) (single-CTA tail mirrors
  //   copyUnroll's predicate); delay has no effect.
  runMultiWriterTailRaceTest</*UseFallback=*/false>(
      /*count=*/2200,
      /*delayedBlock=*/0,
      "block 1 read a slot block 0 had not yet written in phase 1");
}

TEST_F(LocalReduceTailRaceTest, FallbackPathPartitionMismatchExposesStaleRead) {
  // count=2048, blockDim=128, gridDim=8, fp32 (fallback path, aligned on
  // numPerBlock so this isolates the MAIN-loop partition disagreement;
  // tail coverage is already exercised by the vectorized test above):
  // - copyUnroll<4, fp32> per-CTA chunk = 128 * 4 * 4 = 2048. With
  //   count=2048, block 0's main loop covers [0, 2048).
  // - Pre-fix localReduceFallback per-CTA chunk = 128 * 8 = 1024.
  //   Block 0 writes [0, 1024); block 1 writes [1024, 2048).
  //   Delaying block 1 makes block 0's phase-2 read of [1024, 2048) hit
  //   0xCD.
  // - Post-fix (ctaPartition<T, 4>): fallback's per-CTA chunk also = 2048.
  //   Block 0 writes [0, 2048) entirely; delaying block 1 has no effect.
  runMultiWriterTailRaceTest</*UseFallback=*/true>(
      /*count=*/2048,
      /*delayedBlock=*/1,
      "block 0 read a slot fallback assigned to block 1 under its "
      "(pre-fix) numPerBlock; block 1 was sleeping in phase 1");
}
