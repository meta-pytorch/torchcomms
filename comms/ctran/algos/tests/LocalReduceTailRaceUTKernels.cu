// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/localReduce.cuh"
#include "comms/ctran/algos/tests/LocalReduceTailRaceUTKernels.cuh"

namespace {

// Per-CTA release on `flag[blockIdx.x]`. System-scope release-store, so
// any other CTA that does a system-scope acquire load on the same
// address observes writes this CTA committed before the release.
__device__ __forceinline__ void perCtaRelease(int* flag) {
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("st.release.sys.global.s32 [%0], %1;"
                 :
                 : "l"(flag), "r"(1)
                 : "memory");
  }
}

// Per-CTA acquire on the SAME CTA's flag. NO cross-CTA acquire — same
// shape as the inter-round sync `GpeKernelSync.checkPost` provides.
__device__ __forceinline__ void perCtaAcquireOwn(int* flag) {
  if (threadIdx.x == 0) {
    int v;
    do {
      asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
                   : "=r"(v)
                   : "l"(flag)
                   : "memory");
    } while (v == 0);
  }
  __syncthreads();
}

// Spin-sleep on block 0 only, using `__nanosleep` (Volta+ / sm_70+).
// `__nanosleep`'s argument is a 32-bit ns count; loop to reach larger
// total delays.
__device__ __forceinline__ void block0Delay(unsigned long long totalNs) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  constexpr unsigned int kStepNs = 1000000U; // 1 ms per __nanosleep call
  unsigned long long remaining = totalNs;
  while (remaining > 0) {
    unsigned int step = remaining > kStepNs ? kStepNs : (unsigned int)remaining;
    __nanosleep(step);
    remaining -= step;
  }
}

} // namespace

template <typename T, int Unroll16>
__global__ void multiWriterTailRaceKernel(
    T* buf,
    T* out,
    const T* src,
    size_t count,
    int* perCtaFlag,
    unsigned long long block0DelayNs) {
  // Delay block 0 so block 1 reaches phase 2 before block 0 finishes
  // phase 1. The delay runs only on block 0's thread 0; the rest of
  // block 0 waits at the barrier below. Other blocks pass through
  // immediately and proceed to phase 1.
  block0Delay(block0DelayNs);
  __syncthreads();

  // Phase 1: real `localReduceVectorized` writes `buf` from `src`. With
  // NSrcs=1 + commSum the reduction is identity, so `buf` should equal
  // `src` for every byte once all CTAs' writes have committed.
  // localReduceVectorized's tail uses kUnroll=4 internally.
  {
    const T* srcs[1] = {src};
    T* dsts[1] = {buf};
    localReduceVectorized<T, commSum, 1, 1>(
        srcs, dsts, count, blockIdx.x, gridDim.x);
  }

  perCtaRelease(&perCtaFlag[blockIdx.x]);
  perCtaAcquireOwn(&perCtaFlag[blockIdx.x]);

  // Phase 2: real `copyUnroll<Unroll16, T>` reads `buf` and writes `out`.
  // Each CTA reads only the bytes it owns under copyUnroll's per-CTA
  // partition. If `Unroll16` differs from localReduceVectorized's
  // hardcoded kUnroll=4, the per-CTA partitions disagree even at
  // perfectly aligned counts, and the block-0 delay surfaces stale reads.
  copyUnroll<Unroll16, T>(out, buf, count, blockIdx.x, gridDim.x);
}

#define DECL_MULTI_WRITER_KERN(T, U16)                        \
  template __global__ void multiWriterTailRaceKernel<T, U16>( \
      T * buf,                                                \
      T * out,                                                \
      const T* src,                                           \
      size_t count,                                           \
      int* perCtaFlag,                                        \
      unsigned long long block0DelayNs)

DECL_MULTI_WRITER_KERN(int32_t, 4);
// Unroll16=2 is intentionally NOT what production uses; instantiated to
// drive the unroll-mismatch test case.
DECL_MULTI_WRITER_KERN(int32_t, 2);
