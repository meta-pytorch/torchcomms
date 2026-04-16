// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Optimized stochastic rounding with 16-bit random values and
// warp shuffle exchange for efficient Philox RNG utilization.
//
// Key optimization: each Philox call produces 4 uint32 = 8 uint16 random
// values, enough for 8 elements. Groups of G = 8/EltPerPack consecutive
// lanes share a Philox call. Warp shuffles redistribute the outputs so
// each lane calls Philox once per G unroll steps instead of once per step.
//
// Determinism mapping (independent of Unroll/EltPerPack):
//   abs_E = randomBaseOffset + element_position
//   Philox call: philox(seed, abs_E / 8)  ->  r0, r1, r2, r3
//   Channel: (abs_E % 8) / 2  ->  selects r[channel]
//   Half: abs_E % 2  ->  0: low 16 bits, 1: high 16 bits

#ifndef NCCL_COPY_KERNEL_V2_CUH_
#define NCCL_COPY_KERNEL_V2_CUH_

#include "meta/collectives/kernels/reduce_copy_common.cuh"

#include "comms/utils/kernels/rng/philox_rng.cuh"
#include "comms/utils/kernels/stochastic_rounding/stochastic_rounding.cuh"

namespace meta::comms::ncclx::kernels::simplecopy_v2 {

// Select one of 4 uint32 values by runtime index (0-3).
// Compiles to 3 predicated moves — no array, no memory access.
__device__ __forceinline__ uint32_t
philox_select(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, int idx) {
  uint32_t lo = (idx & 1) ? r1 : r0;
  uint32_t hi = (idx & 1) ? r3 : r2;
  return (idx & 2) ? hi : lo;
}

// Select one of 8 uint32 values by runtime index (0-7).
// Compiles to 7 predicated moves — no array, no memory access.
__device__ __forceinline__ uint32_t delta_select8(
    uint32_t v0,
    uint32_t v1,
    uint32_t v2,
    uint32_t v3,
    uint32_t v4,
    uint32_t v5,
    uint32_t v6,
    uint32_t v7,
    int idx) {
  uint32_t a0 = (idx & 1) ? v1 : v0;
  uint32_t a1 = (idx & 1) ? v3 : v2;
  uint32_t a2 = (idx & 1) ? v5 : v4;
  uint32_t a3 = (idx & 1) ? v7 : v6;
  uint32_t b0 = (idx & 2) ? a1 : a0;
  uint32_t b1 = (idx & 2) ? a3 : a2;
  return (idx & 4) ? b1 : b0;
}

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
// PhiloxWarpExchange — encapsulated RNG with warp shuffle redistribution
// =========================================================================

template <int Unroll, int EltPerPack>
struct PhiloxWarpExchange {
  static constexpr int G = 8 / EltPerPack;
  static constexpr bool kCanExchange = (Unroll >= G);
  static_assert(
      !kCanExchange || (Unroll % G == 0),
      "When exchange is used, Unroll must be a multiple of G");

  // Per-unroll-step random storage:
  //   EPP=4: rand_a[u] = packed lo pair, rand_b[u] = packed hi pair
  //   EPP=2: rand_a[u] = packed uint32 for 2 elements
  //   EPP=1: rand_a[u] = uint32 containing channel value (extract half later)
  //   rand_b is only used for EPP=4.
  uint32_t rand_a[Unroll];
  uint32_t rand_b[Unroll];

  // For EPP=1: which half of each uint32 to use (0=low16, 1=high16).
  // Constant across unroll steps for a given thread.
  int half;

  __device__ __forceinline__ void generate(
      uint64_t randomSeed,
      uint64_t randomBaseOffset,
      uint64_t threadEltBase,
      int lane) {
    if constexpr (EltPerPack == 1) {
      half = int((randomBaseOffset + threadEltBase) % 2);
    }

    if constexpr (kCanExchange) {
      uint64_t warpBase =
          randomBaseOffset + threadEltBase - uint64_t(lane) * EltPerPack;
      if (warpBase % 8 == 0) {
        generateExchange(randomSeed, randomBaseOffset, threadEltBase, lane);
        return;
      }
    }
    generateSimple(randomSeed, randomBaseOffset, threadEltBase);
  }

 private:
  __device__ __forceinline__ void generateSimple(
      uint64_t randomSeed,
      uint64_t randomBaseOffset,
      uint64_t threadEltBase) {
    uint32_t r0, r1, r2, r3;
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      uint64_t abs_E = randomBaseOffset + threadEltBase +
          uint64_t(u) * WARP_SIZE * EltPerPack;
      uint64_t philox_off = abs_E / 8;
      philox_randint4x(randomSeed, philox_off, r0, r1, r2, r3);

      int channel = int(abs_E % 8) / 2;
      rand_a[u] = philox_select(r0, r1, r2, r3, channel);
      if constexpr (EltPerPack == 4) {
        rand_b[u] = philox_select(r0, r1, r2, r3, channel + 1);
      }
    }
  }

  // All writes to rand_a/rand_b use compile-time indices only (rnd is
  // unrolled, slot literals 0..G-1 are constants). Runtime lane identity
  // is resolved via predicated-move selects, keeping everything in registers.

  __device__ __forceinline__ void exchangeEPP4(
      int rnd,
      int local_id,
      uint32_t r0,
      uint32_t r1,
      uint32_t r2,
      uint32_t r3) {
    // G=2: lane 0 uses (r0,r1), lane 1 uses (r2,r3)
    uint32_t my_a = (local_id == 0) ? r0 : r2;
    uint32_t my_b = (local_id == 0) ? r1 : r3;
    uint32_t send_a = (local_id == 0) ? r2 : r0;
    uint32_t send_b = (local_id == 0) ? r3 : r1;

    uint32_t recv_a = __shfl_xor_sync(0xFFFFFFFF, send_a, 1, G);
    uint32_t recv_b = __shfl_xor_sync(0xFFFFFFFF, send_b, 1, G);

    // Static indices: rnd*2+0, rnd*2+1 are compile-time
    rand_a[rnd * 2 + 0] = (local_id == 0) ? my_a : recv_a;
    rand_b[rnd * 2 + 0] = (local_id == 0) ? my_b : recv_b;
    rand_a[rnd * 2 + 1] = (local_id == 1) ? my_a : recv_a;
    rand_b[rnd * 2 + 1] = (local_id == 1) ? my_b : recv_b;
  }

  __device__ __forceinline__ void exchangeEPP2(
      int rnd,
      int local_id,
      uint32_t r0,
      uint32_t r1,
      uint32_t r2,
      uint32_t r3) {
    // G=4: each lane uses r[local_id]
    uint32_t own = philox_select(r0, r1, r2, r3, local_id);

    uint32_t send_d1 = philox_select(r0, r1, r2, r3, local_id ^ 1);
    uint32_t recv_d1 = __shfl_xor_sync(0xFFFFFFFF, send_d1, 1, G);

    uint32_t send_d2 = philox_select(r0, r1, r2, r3, local_id ^ 2);
    uint32_t recv_d2 = __shfl_xor_sync(0xFFFFFFFF, send_d2, 2, G);

    uint32_t send_d3 = philox_select(r0, r1, r2, r3, local_id ^ 3);
    uint32_t recv_d3 = __shfl_xor_sync(0xFFFFFFFF, send_d3, 3, G);

    // Slot k needs the result from delta = (local_id ^ k).
    // philox_select picks from {own(d=0), recv_d1, recv_d2, recv_d3}.
    // Static indices: rnd*4+{0,1,2,3} are compile-time.
    rand_a[rnd * 4 + 0] =
        philox_select(own, recv_d1, recv_d2, recv_d3, local_id);
    rand_a[rnd * 4 + 1] =
        philox_select(own, recv_d1, recv_d2, recv_d3, local_id ^ 1);
    rand_a[rnd * 4 + 2] =
        philox_select(own, recv_d1, recv_d2, recv_d3, local_id ^ 2);
    rand_a[rnd * 4 + 3] =
        philox_select(own, recv_d1, recv_d2, recv_d3, local_id ^ 3);
  }

  __device__ __forceinline__ void exchangeEPP1(
      int rnd,
      int local_id,
      uint32_t r0,
      uint32_t r1,
      uint32_t r2,
      uint32_t r3) {
    // G=8: each lane uses r[local_id/2]
    int channel = local_id / 2;
    uint32_t own = philox_select(r0, r1, r2, r3, channel);

    uint32_t recv_d1 = __shfl_xor_sync(
        0xFFFFFFFF, philox_select(r0, r1, r2, r3, (local_id ^ 1) / 2), 1, G);
    uint32_t recv_d2 = __shfl_xor_sync(
        0xFFFFFFFF, philox_select(r0, r1, r2, r3, (local_id ^ 2) / 2), 2, G);
    uint32_t recv_d3 = __shfl_xor_sync(
        0xFFFFFFFF, philox_select(r0, r1, r2, r3, (local_id ^ 3) / 2), 3, G);
    uint32_t recv_d4 = __shfl_xor_sync(
        0xFFFFFFFF, philox_select(r0, r1, r2, r3, (local_id ^ 4) / 2), 4, G);
    uint32_t recv_d5 = __shfl_xor_sync(
        0xFFFFFFFF, philox_select(r0, r1, r2, r3, (local_id ^ 5) / 2), 5, G);
    uint32_t recv_d6 = __shfl_xor_sync(
        0xFFFFFFFF, philox_select(r0, r1, r2, r3, (local_id ^ 6) / 2), 6, G);
    uint32_t recv_d7 = __shfl_xor_sync(
        0xFFFFFFFF, philox_select(r0, r1, r2, r3, (local_id ^ 7) / 2), 7, G);

    // Slot k needs the result from delta = (local_id ^ k).
    // delta_select8 picks from {own(d=0), recv_d1..recv_d7}.
    // Static indices: rnd*8+{0..7} are compile-time.
    rand_a[rnd * 8 + 0] = delta_select8(
        own,
        recv_d1,
        recv_d2,
        recv_d3,
        recv_d4,
        recv_d5,
        recv_d6,
        recv_d7,
        local_id);
    rand_a[rnd * 8 + 1] = delta_select8(
        own,
        recv_d1,
        recv_d2,
        recv_d3,
        recv_d4,
        recv_d5,
        recv_d6,
        recv_d7,
        local_id ^ 1);
    rand_a[rnd * 8 + 2] = delta_select8(
        own,
        recv_d1,
        recv_d2,
        recv_d3,
        recv_d4,
        recv_d5,
        recv_d6,
        recv_d7,
        local_id ^ 2);
    rand_a[rnd * 8 + 3] = delta_select8(
        own,
        recv_d1,
        recv_d2,
        recv_d3,
        recv_d4,
        recv_d5,
        recv_d6,
        recv_d7,
        local_id ^ 3);
    rand_a[rnd * 8 + 4] = delta_select8(
        own,
        recv_d1,
        recv_d2,
        recv_d3,
        recv_d4,
        recv_d5,
        recv_d6,
        recv_d7,
        local_id ^ 4);
    rand_a[rnd * 8 + 5] = delta_select8(
        own,
        recv_d1,
        recv_d2,
        recv_d3,
        recv_d4,
        recv_d5,
        recv_d6,
        recv_d7,
        local_id ^ 5);
    rand_a[rnd * 8 + 6] = delta_select8(
        own,
        recv_d1,
        recv_d2,
        recv_d3,
        recv_d4,
        recv_d5,
        recv_d6,
        recv_d7,
        local_id ^ 6);
    rand_a[rnd * 8 + 7] = delta_select8(
        own,
        recv_d1,
        recv_d2,
        recv_d3,
        recv_d4,
        recv_d5,
        recv_d6,
        recv_d7,
        local_id ^ 7);
  }

  __device__ __forceinline__ void generateExchange(
      uint64_t randomSeed,
      uint64_t randomBaseOffset,
      uint64_t threadEltBase,
      int lane) {
    constexpr int nRounds = Unroll / G;
    int local_id = lane % G;
    uint64_t groupBase = threadEltBase - uint64_t(local_id) * EltPerPack;
    uint32_t r0, r1, r2, r3;

#pragma unroll nRounds
    for (int rnd = 0; rnd < nRounds; rnd++) {
      uint64_t ref_E =
          groupBase + uint64_t(rnd * G + local_id) * WARP_SIZE * EltPerPack;
      uint64_t philox_off = (randomBaseOffset + ref_E) / 8;
      philox_randint4x(randomSeed, philox_off, r0, r1, r2, r3);

      if constexpr (EltPerPack == 4) {
        exchangeEPP4(rnd, local_id, r0, r1, r2, r3);
      } else if constexpr (EltPerPack == 2) {
        exchangeEPP2(rnd, local_id, r0, r1, r2, r3);
      } else {
        exchangeEPP1(rnd, local_id, r0, r1, r2, r3);
      }
    }
  }
};

// =========================================================================
// storeFirstDestinationSR — store with stochastic rounding using exchange
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
    const PhiloxWarpExchange<Unroll, EltPerPack>& rng) {
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
            acc[u], rng.rand_a[u], rng.rand_b[u]);
      } else if constexpr (EltPerPack == 2) {
        dstPack = ApplyStochasticRound<AccType, DstType, EltPerPack>::cast(
            acc[u], rng.rand_a[u]);
      } else {
        uint16_t r16 = (rng.half == 0)
            ? static_cast<uint16_t>(rng.rand_a[u])
            : static_cast<uint16_t>(rng.rand_a[u] >> 16);
        dstPack = ApplyStochasticRound<AccType, DstType, EltPerPack>::cast(
            acc[u], r16);
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

    PhiloxWarpExchange<Unroll, EltPerPack> rng;
    rng.generate(randomSeed, randomBaseOffset, threadEltBase, lane);

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
