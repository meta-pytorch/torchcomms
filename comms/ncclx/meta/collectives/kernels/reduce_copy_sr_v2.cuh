// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Stochastic rounding with Philox RNG.
// Each Philox call produces 128 bits (4 uint32 = 8 uint16) of random values.
// PhiloxResult union allows addressing the output at any width directly.

#ifndef NCCL_COPY_KERNEL_V2_CUH_
#define NCCL_COPY_KERNEL_V2_CUH_

#include "meta/collectives/kernels/reduce_copy_common.cuh"

#include "comms/utils/kernels/rng/philox_rng.cuh"
#include "comms/utils/kernels/stochastic_rounding/stochastic_rounding.cuh"

namespace meta::comms::ncclx::kernels::simplecopy_v2 {

// =========================================================================
// ApplyStochasticRound — 16-bit random value interface
// =========================================================================

template <typename AccType, typename DstType, int EltPerPack>
struct ApplyStochasticRound;

template <>
struct ApplyStochasticRound<float, __nv_bfloat16, 1> {
  __device__ __forceinline__ static BytePack<sizeof(__nv_bfloat16)> cast(
      BytePack<sizeof(float)> a,
      uint16_t rand16) {
    float val = fromPack<float>(a);
    return toPack(stochastic_round_bf16<kHasHardwareSR>(val, rand16));
  }
};

template <>
struct ApplyStochasticRound<float, __nv_bfloat16, 2> {
  __device__ __forceinline__ static BytePack<2 * sizeof(__nv_bfloat16)> cast(
      BytePack<2 * sizeof(float)> a,
      uint32_t rand_packed) {
    float2 vals = fromPack<float2>(a);
    return toPack(stochastic_round_bf16x2<kHasHardwareSR>(vals, rand_packed));
  }
};

template <>
struct ApplyStochasticRound<float, __nv_bfloat16, 4> {
  __device__ __forceinline__ static BytePack<4 * sizeof(__nv_bfloat16)>
  cast(BytePack<4 * sizeof(float)> a, uint32_t rand_lo, uint32_t rand_hi) {
    float4 vals = fromPack<float4>(a);
    BytePack<4 * sizeof(__nv_bfloat16)> result;
    result.half[0] = toPack(
        stochastic_round_bf16x2<kHasHardwareSR>(
            make_float2(vals.x, vals.y), rand_lo));
    result.half[1] = toPack(
        stochastic_round_bf16x2<kHasHardwareSR>(
            make_float2(vals.z, vals.w), rand_hi));
    return result;
  }
};

// =========================================================================
// PhiloxRNGGenerator — encapsulated RNG
// =========================================================================

template <int Unroll, int EltPerPack>
struct PhiloxRNGGenerator {
  // 4 uint32_t random values converts to 8 uint16_t random values.
  static constexpr int philoxCalls = std::max((Unroll * EltPerPack + 7) / 8, 1);

  union {
    PhiloxResult calls[philoxCalls];
    uint32_t u32[philoxCalls * 4];
    uint16_t u16[philoxCalls * 8];
  } rand;

  __device__ __forceinline__ void generate(
      uint64_t randomSeed,
      uint64_t randomBaseOffset,
      uint64_t threadEltBase) {
#pragma unroll philoxCalls
    for (int u = 0; u < philoxCalls; u++) {
      uint64_t philox_off = randomBaseOffset + threadEltBase +
          uint64_t(u) * WARP_SIZE * EltPerPack;
      rand.calls[u] = philox_randint4x(randomSeed, philox_off);
    }
  }
};

// =========================================================================
// storeFirstDestinationSR — store with stochastic rounding
// =========================================================================

template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    int FirstDstIdx,
    typename IterType>
static __device__ __forceinline__ void storeFirstDestinationSR(
    BytePack<EltPerPack * sizeof(AccType)>* acc,
    IterType& iter,
    const PhiloxRNGGenerator<Unroll, EltPerPack>& rng) {
  using DstType = typename IterType::template PtrType<FirstDstIdx>;
  static_assert(
      sizeof(DstType) <= sizeof(AccType),
      "DstType must be of lower or same precision as AccType (by size).");

  if constexpr (std::is_same_v<AccType, DstType>) {
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      st_global<EltPerPack * sizeof(DstType)>(
          iter.template get<FirstDstIdx>(), acc[u]);
      iter.template advanceUnroll<FirstDstIdx>();
    }
  } else {
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      BytePack<EltPerPack * sizeof(DstType)> dstPack;
      if constexpr (EltPerPack == 4) {
        dstPack = ApplyStochasticRound<AccType, DstType, EltPerPack>::cast(
            acc[u], rng.rand.u32[2 * u], rng.rand.u32[2 * u + 1]);
      } else if constexpr (EltPerPack == 2) {
        dstPack = ApplyStochasticRound<AccType, DstType, EltPerPack>::cast(
            acc[u], rng.rand.u32[u]);
      } else {
        dstPack = ApplyStochasticRound<AccType, DstType, EltPerPack>::cast(
            acc[u], rng.rand.u16[u]);
      }
      st_global<EltPerPack * sizeof(DstType)>(
          iter.template get<FirstDstIdx>(), dstPack);
      iter.template advanceUnroll<FirstDstIdx>();
    }
  }
}

// =========================================================================
// reduceCopyPacksSR — inner loop with optimized RNG
// =========================================================================

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

  IntThread warp = thread / WARP_SIZE;
  IntThread lane = thread % WARP_SIZE;
  constexpr int kEltPerHunk = Unroll * WARP_SIZE * EltPerPack;
  uint64_t threadEltBase = uint64_t(nEltsBehind) +
      uint64_t(warp) * kEltPerHunk + uint64_t(lane) * EltPerPack;

  meta::comms::ncclx::kernels::
      CopyIterator<Unroll, EltPerPack, IntBytes, IntThread, Ts...>
      iter(nThreads, thread, nEltsBehind, nEltsAhead, ptrs...);

  IntThread nWarps = nThreads / WARP_SIZE;

  while (iter.hasWork()) {
    BytePack<EltPerPack * sizeof(AccType)> acc[Unroll];
    meta::comms::ncclx::kernels::loadFirstSource<Unroll, EltPerPack, AccType>(
        acc, iter);

    meta::comms::ncclx::kernels::ReduceSources<1, kSrcPtrCount>::
        template apply<Unroll, EltPerPack, AccType>(acc, iter);

    PhiloxRNGGenerator<Unroll, EltPerPack> rng;
    rng.generate(randomSeed, randomBaseOffset, threadEltBase);

    storeFirstDestinationSR<Unroll, EltPerPack, AccType, kDstStartIdx>(
        acc, iter, rng);

    iter.advance();
    threadEltBase += uint64_t(nWarps) * kEltPerHunk;
  }
}

// =========================================================================
// reduceCopySR — top-level with multi-pass alignment
// =========================================================================

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

} // namespace meta::comms::ncclx::kernels::simplecopy_v2

#endif // NCCL_COPY_KERNEL_V2_CUH_
