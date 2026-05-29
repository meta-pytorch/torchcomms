// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/algos/localReduce.cuh"
#include "comms/ctran/algos/tests/LocalReduceTailRaceUTKernels.cuh"

namespace {

// Spin-sleep on a single designated block via a clock-cycle busy wait.
// Cross-platform (NVIDIA + AMD/HIP); `__nanosleep` is NVIDIA-only.
// Cycle-to-ns conversion is approximate (~2 GHz upper bound) which is
// fine since the goal is just "delay enough that other blocks reach
// phase 2 first".
__device__ __forceinline__ void blockNDelay(
    int targetBlock,
    unsigned long long totalNs) {
  if (blockIdx.x != targetBlock || threadIdx.x != 0) {
    return;
  }
  constexpr unsigned long long kCyclesPerNs = 2;
  const long long targetCycles = static_cast<long long>(totalNs * kCyclesPerNs);
  const long long start = clock64();
  while (clock64() - start < targetCycles) {
    // busy wait
  }
}

} // namespace

template <typename T, bool UseFallback>
__global__ void multiWriterTailRaceKernel(
    T* buf,
    T* out,
    const T* src,
    size_t count,
    ctran::algos::GpeKernelSync* sync,
    int delayedBlockIdx,
    unsigned long long delayNs) {
  // Delay one block so other blocks reach phase 2 before it finishes
  // phase 1. The delay runs only on the target block's thread 0; the rest
  // of that block waits at the barrier below. Other blocks pass through
  // immediately and proceed to phase 1.
  blockNDelay(delayedBlockIdx, delayNs);
  __syncthreads();

  // Phase 1: real `localReduce*` writes `buf` from `src`. With NSrcs=1 +
  // commSum the reduction is identity, so `buf` should equal `src` for
  // every byte once all CTAs' writes have committed.
  {
    const T* srcs[1] = {src};
    T* dsts[1] = {buf};
    if constexpr (UseFallback) {
      localReduceFallback<T, commSum>(
          /*nsrcs=*/1, srcs, /*ndsts=*/1, dsts, count, blockIdx.x, gridDim.x);
    } else {
      localReduceVectorized<T, commSum, 1, 1>(
          srcs, dsts, count, blockIdx.x, gridDim.x);
    }
  }

  // Intra-CTA release-then-acquire on this CTA's own flag pair. Purpose
  // is purely a per-CTA fence between phase-1 writes and phase-2 reads
  // on `buf` (without it, compiler/GPU may reorder across the phases).
  // Each call only touches `sync->completeFlag[blockIdx.x]` /
  // `sync->postFlag[blockIdx.x]`; no CTA reads another CTA's slot, so
  // these calls provide NO cross-CTA visibility. Host pre-posted step=1
  // for every worker before launch, so `waitPost` returns past the
  // acquire fence immediately.
  //
  // GpeKernelSync is a GPE-thread<->kernel primitive
  // (`comms/ctran/algos/common/GpeKernelSync.h`), not a CTA<->CTA
  // primitive — we reuse its device-side primitives here only because
  // they give us the release/acquire fence shape we want for the
  // intra-CTA phase boundary. The test deliberately omits any cross-CTA
  // barrier so that the byte-ownership invariant between phase-1 writer
  // and phase-2 reader is the only thing standing between the two
  // ops — pre-fix that invariant fails for some byte ranges, and the
  // delayed-block setup turns the resulting stale read into a
  // deterministic failure.
  ctran::algos::GpeKernelSyncDev::complete(sync, blockIdx.x, /*step=*/1);
  ctran::algos::GpeKernelSyncDev::waitPost(sync, blockIdx.x, /*step=*/1);

  // Phase 2: real `copyUnroll<4, T>` reads `buf` and writes `out`. Each
  // CTA reads only the bytes it owns under copyUnroll's per-CTA
  // partition. If the phase-1 writer's per-CTA partition assigned some
  // of those bytes to a different CTA, this CTA's read may hit init
  // sentinel because the writer CTA was still sleeping in `blockNDelay`
  // and hadn't issued its phase-1 writes yet.
  copyUnroll<4, T>(out, buf, count, blockIdx.x, gridDim.x);
}

#define DECL_MULTI_WRITER_KERN(T)                               \
  template __global__ void multiWriterTailRaceKernel<T, false>( \
      T * buf,                                                  \
      T * out,                                                  \
      const T* src,                                             \
      size_t count,                                             \
      ctran::algos::GpeKernelSync* sync,                        \
      int delayedBlockIdx,                                      \
      unsigned long long delayNs);                              \
  template __global__ void multiWriterTailRaceKernel<T, true>(  \
      T * buf,                                                  \
      T * out,                                                  \
      const T* src,                                             \
      size_t count,                                             \
      ctran::algos::GpeKernelSync* sync,                        \
      int delayedBlockIdx,                                      \
      unsigned long long delayNs)

DECL_MULTI_WRITER_KERN(int32_t);
