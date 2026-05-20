// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/algos/localReduce.cuh"
#include "comms/ctran/algos/tests/LocalReduceTailRaceUTKernels.cuh"

namespace {

// Spin-sleep on block 0 only via a clock-cycle busy wait. Cross-platform
// (NVIDIA + AMD/HIP); `__nanosleep` is NVIDIA-only. Cycle-to-ns
// conversion is approximate (~2 GHz upper bound) which is fine since
// the goal is just "delay enough that block 1 reaches phase 2 first".
__device__ __forceinline__ void block0Delay(unsigned long long totalNs) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
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

template <typename T>
__global__ void multiWriterTailRaceKernel(
    T* buf,
    T* out,
    const T* src,
    size_t count,
    ctran::algos::GpeKernelSync* sync,
    unsigned long long block0DelayNs) {
  // Delay block 0 so other blocks reach phase 2 before block 0 finishes
  // phase 1. The delay runs only on block 0's thread 0; the rest of
  // block 0 waits at the barrier below. Other blocks pass through
  // immediately and proceed to phase 1.
  block0Delay(block0DelayNs);
  __syncthreads();

  // Phase 1: real `localReduceVectorized` writes `buf` from `src`. With
  // NSrcs=1 + commSum the reduction is identity, so `buf` should equal
  // `src` for every byte once all CTAs' writes have committed.
  {
    const T* srcs[1] = {src};
    T* dsts[1] = {buf};
    localReduceVectorized<T, commSum, 1, 1>(
        srcs, dsts, count, blockIdx.x, gridDim.x);
  }

  // Per-CTA release + per-CTA acquire-on-own-flag, using the REAL
  // production sync API. `complete` does `__syncthreads()` +
  // `st.release.sys.global` on `sync->completeFlag[blockIdx.x]`.
  // `waitPost` polls `sync->postFlag[blockIdx.x]` with `ld.acquire.sys.global`
  // until it observes a value >= step. The host pre-posted step=1 for
  // every worker before kernel launch, so `waitPost` returns immediately
  // — but the acquire fence still fires. There is intentionally NO
  // cross-CTA acquire here (no host `isComplete` between phases),
  // matching the per-CTA-only sync pattern that opens the bug window.
  ctran::algos::GpeKernelSyncDev::complete(sync, blockIdx.x, /*step=*/1);
  ctran::algos::GpeKernelSyncDev::waitPost(sync, blockIdx.x, /*step=*/1);

  // Phase 2: real `copyUnroll<4, T>` reads `buf` and writes `out`. Each
  // CTA reads only the bytes it owns under copyUnroll's per-CTA
  // partition. If localReduceVectorized's per-CTA partition assigned
  // some of those bytes to a different CTA in phase 1 (the bug), this
  // CTA's read may hit init sentinel because the writer CTA was still
  // sleeping in `block0Delay` and hadn't issued its phase-1 writes yet.
  copyUnroll<4, T>(out, buf, count, blockIdx.x, gridDim.x);
}

#define DECL_MULTI_WRITER_KERN(T)                        \
  template __global__ void multiWriterTailRaceKernel<T>( \
      T * buf,                                           \
      T * out,                                           \
      const T* src,                                      \
      size_t count,                                      \
      ctran::algos::GpeKernelSync* sync,                 \
      unsigned long long block0DelayNs)

DECL_MULTI_WRITER_KERN(int32_t);
