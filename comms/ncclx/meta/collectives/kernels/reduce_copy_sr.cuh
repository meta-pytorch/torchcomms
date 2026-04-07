// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_REDUCE_COPY_SR_CUH_
#define NCCL_REDUCE_COPY_SR_CUH_

#include "meta/collectives/kernels/reduce_copy_common.cuh"

#include "comms/utils/kernels/rng/philox_rng.cuh"
#include "comms/utils/kernels/stochastic_rounding/stochastic_rounding.cuh"

namespace meta::comms::ncclx::kernels::simplecopy {

// Stochastic rounding with pre-computed random bits.
// Converts a BytePack of AccType elements to DstType using pre-computed
// Philox random values. Does NOT call the RNG — caller is responsible for
// generating r0..r3 via philox_randint4x() BEFORE calling this.
template <typename AccType, typename DstType, int EltPerPack>
struct ApplyStochasticRound;

// --- float -> __nv_bfloat16, 1 element ---
template <>
struct ApplyStochasticRound<float, __nv_bfloat16, 1> {
  __device__ __forceinline__ static BytePack<sizeof(__nv_bfloat16)>
  cast(BytePack<sizeof(float)> a, uint32_t r0, uint32_t, uint32_t, uint32_t) {
    float val = fromPack<float>(a);
    return toPack(stochastic_round_bf16_software(val, r0));
  }
};

// --- float -> __nv_bfloat16, 2 elements ---
template <>
struct ApplyStochasticRound<float, __nv_bfloat16, 2> {
  __device__ __forceinline__ static BytePack<2 * sizeof(__nv_bfloat16)> cast(
      BytePack<2 * sizeof(float)> a,
      uint32_t r0,
      uint32_t r1,
      uint32_t,
      uint32_t) {
    float2 vals = fromPack<float2>(a);
#if __CUDA_ARCH__ >= 1000
    uint32_t rand_bits = r0 ^ (r1 << 16);
    return toPack(stochastic_round_bf16x2_blackwell(vals, rand_bits));
#else
    return toPack(stochastic_round_bf16x2_software(vals, r0, r1));
#endif
  }
};

// --- float -> __nv_bfloat16, 4 elements ---
template <>
struct ApplyStochasticRound<float, __nv_bfloat16, 4> {
  __device__ __forceinline__ static BytePack<4 * sizeof(__nv_bfloat16)> cast(
      BytePack<4 * sizeof(float)> a,
      uint32_t r0,
      uint32_t r1,
      uint32_t r2,
      uint32_t r3) {
    float4 vals = fromPack<float4>(a);
    BytePack<4 * sizeof(__nv_bfloat16)> result;
#if __CUDA_ARCH__ >= 1000
    uint32_t rand_lo = r0 ^ (r1 << 16);
    uint32_t rand_hi = r2 ^ (r3 << 16);
    result.half[0] = toPack(stochastic_round_bf16x2_blackwell(
        make_float2(vals.x, vals.y), rand_lo));
    result.half[1] = toPack(stochastic_round_bf16x2_blackwell(
        make_float2(vals.z, vals.w), rand_hi));
#else
    result.half[0] = toPack(
        stochastic_round_bf16x2_software(make_float2(vals.x, vals.y), r0, r1));
    result.half[1] = toPack(
        stochastic_round_bf16x2_software(make_float2(vals.z, vals.w), r2, r3));
#endif
    return result;
  }
};

// Store accumulator to destination with stochastic rounding using
// pre-computed random bits. When AccType == DstType, stores directly.
// randR0..randR3[u] contain the pre-computed Philox random values for
// unroll step u.
template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    int FirstDstIdx,
    typename IterType>
static __device__ __forceinline__ void storeFirstDestinationSR(
    BytePack<EltPerPack * sizeof(AccType)>* acc,
    IterType& iter,
    uint32_t* randR0,
    uint32_t* randR1,
    uint32_t* randR2,
    uint32_t* randR3) {
  using DstType = typename IterType::template PtrType<FirstDstIdx>;
  static_assert(
      sizeof(DstType) <= sizeof(AccType),
      "DstType must be of lower or same precision as AccType (by size).");

  if constexpr (std::is_same_v<AccType, DstType>) {
    // Same precision — no stochastic rounding needed, store directly.
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      st_global<EltPerPack * sizeof(DstType)>(
          iter.template get<FirstDstIdx>(), acc[u]);
      iter.template advanceUnroll<FirstDstIdx>();
    }
  } else {
    // Lower precision destination — apply stochastic rounding with
    // pre-computed random bits.
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      auto dstPack = ApplyStochasticRound<AccType, DstType, EltPerPack>::cast(
          acc[u], randR0[u], randR1[u], randR2[u], randR3[u]);
      st_global<EltPerPack * sizeof(DstType)>(
          iter.template get<FirstDstIdx>(), dstPack);
      iter.template advanceUnroll<FirstDstIdx>();
    }
  }
}

// Inner reduce-copy with stochastic rounding on the store path.
// Pipeline: pre-compute RNG → load sources → reduce → stochastic store.
// The RNG computation has NO data dependency on the loads, so the GPU can
// overlap Philox computation with memory access latency.
template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    size_t DstPtrCount,
    typename IntThread,
    typename IntBytes,
    typename... Ts>
__device__ __forceinline__ void reduceCopyPacksSR(
    IntThread nThreads,
    IntThread& thread,
    IntBytes& nEltsBehind,
    IntBytes& nEltsAhead,
    uint64_t randomSeed,
    uint64_t randomBaseOffset,
    Ts*... ptrs) {
  static_assert(
      std::is_signed<IntBytes>::value,
      "IntBytes must be a signed integral type.");
  static_assert(EltPerPack > 0, "EltPerPack must be greater than 0");
  static_assert(DstPtrCount > 0, "DstPtrCount must be greater than 0");
  static_assert(DstPtrCount == 1, "Currently only support 1 dst pointer");

  constexpr auto kSrcPtrCount = sizeof...(Ts) - DstPtrCount;
  static_assert(kSrcPtrCount > 0, "There must be at least one src pointer");
  static_assert(kSrcPtrCount <= 3, "We only support up to 3 src pointers");

  constexpr size_t kDstStartIdx = sizeof...(Ts) - DstPtrCount;

  // Capture element offset BEFORE CopyIterator modifies nEltsBehind.
  IntThread warp = thread / WARP_SIZE;
  IntThread lane = thread % WARP_SIZE;
  constexpr int kEltPerHunk = Unroll * WARP_SIZE * EltPerPack;
  uint64_t threadEltOffset = uint64_t(nEltsBehind) +
      uint64_t(warp) * kEltPerHunk + uint64_t(lane) * EltPerPack;

  CopyIterator<Unroll, EltPerPack, IntBytes, IntThread, Ts...> iter(
      nThreads, thread, nEltsBehind, nEltsAhead, ptrs...);

  IntThread nWarps = nThreads / WARP_SIZE;

  while (iter.hasWork()) {
    // ---- Phase 1: Load first source into accumulator ----
    BytePack<EltPerPack * sizeof(AccType)> acc[Unroll];
    loadFirstSource<Unroll, EltPerPack, AccType>(acc, iter);

    // ---- Phase 2: Load + reduce remaining sources ----
    ReduceSources<1, kSrcPtrCount>::template apply<Unroll, EltPerPack, AccType>(
        acc, iter);

    // ---- Philox RNG: Pre-compute random bits (NO data dependency on acc) ----
    // Placed here so the compiler/GPU can overlap RNG ALU ops with
    // the preceding memory loads (no dependency on acc values).
    uint32_t randR0[Unroll], randR1[Unroll], randR2[Unroll], randR3[Unroll];
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      uint64_t elemOffset = randomBaseOffset + threadEltOffset +
          uint64_t(u) * WARP_SIZE * EltPerPack;
      philox_randint4x(
          randomSeed, elemOffset, randR0[u], randR1[u], randR2[u], randR3[u]);
    }

    // ---- Phase 3: Stochastic round + store ----
    // Uses pre-computed random bits from above.
    storeFirstDestinationSR<Unroll, EltPerPack, AccType, kDstStartIdx>(
        acc, iter, randR0, randR1, randR2, randR3);

    iter.advance();
    threadEltOffset += uint64_t(nWarps) * kEltPerHunk;
  }
}

// Reduce-sum with stochastic rounding on the store path.
// Same as reduceCopyMixed but uses stochastic rounding when DstType is
// lower precision than AccType. Philox RNG is pre-computed before memory
// loads for latency hiding. Each element at absolute offset N uses
// philox(randomSeed, randomBaseOffset + N).
template <
    int Unroll,
    typename AccType,
    typename DstType,
    typename IntBytes,
    typename... SrcTs>
__device__ __forceinline__ void reduceCopySR(
    int thread,
    int nThreads,
    DstType* dstPtr,
    IntBytes nElts,
    uint64_t randomSeed,
    uint64_t randomBaseOffset,
    SrcTs*... srcPtrs) {
  int lane = thread % WARP_SIZE;
  constexpr int BigPackSize = 16;
  constexpr size_t kNSrcs = sizeof...(SrcTs);
  static_assert(kNSrcs > 0, "reduceCopySR requires at least one source");

  IntBytes nEltsBehind = 0;
  IntBytes nEltsAhead = nElts;

  if constexpr (BigPackSize > sizeof(AccType)) {
    bool aligned = true;
    if (lane == 0) {
      aligned &= 0 == cvta_to_global(dstPtr) % BigPackSize;
      ((aligned &= 0 == cvta_to_global(srcPtrs) % BigPackSize), ...);
    }
    aligned = __all_sync(~0u, aligned);
    if (aligned) {
      reduceCopyPacksSR<Unroll, BigPackSize / sizeof(AccType), AccType, 1>(
          nThreads,
          thread,
          nEltsBehind,
          nEltsAhead,
          randomSeed,
          randomBaseOffset,
          srcPtrs...,
          dstPtr);
      if (nEltsAhead == 0)
        return;

      reduceCopyPacksSR<1, BigPackSize / sizeof(AccType), AccType, 1>(
          nThreads,
          thread,
          nEltsBehind,
          nEltsAhead,
          randomSeed,
          randomBaseOffset,
          srcPtrs...,
          dstPtr);
      if (nEltsAhead == 0)
        return;
    }
  }

  reduceCopyPacksSR<Unroll * (16 / sizeof(AccType)) / 2, 1, AccType, 1>(
      nThreads,
      thread,
      nEltsBehind,
      nEltsAhead,
      randomSeed,
      randomBaseOffset,
      srcPtrs...,
      dstPtr);
  if (nEltsAhead == 0)
    return;

  reduceCopyPacksSR<1, 1, AccType, 1>(
      nThreads,
      thread,
      nEltsBehind,
      nEltsAhead,
      randomSeed,
      randomBaseOffset,
      srcPtrs...,
      dstPtr);
}

} // namespace meta::comms::ncclx::kernels::simplecopy

#endif // NCCL_REDUCE_COPY_SR_CUH_
