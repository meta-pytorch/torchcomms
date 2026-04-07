// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "meta/collectives/kernels/copy_iterator.cuh"
#include "meta/collectives/kernels/reduce_copy_common.cuh"

namespace meta::comms::ncclx::kernels::simplecopy {

template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    int FirstDstIdx,
    typename IterType>
static __device__ __forceinline__ void storeFirstDestination(
    BytePack<EltPerPack * sizeof(AccType)>* acc,
    IterType& iter) {
  using DstType = typename IterType::template PtrType<FirstDstIdx>;
  // Enforce that we only store to equal-or-lower precision destinations from
  // the accumulator type (e.g., fp32 -> bf16, fp32 -> fp32).
  static_assert(
      sizeof(DstType) <= sizeof(AccType),
      "DstType must be of lower or same precision as AccType (by size).");

  if constexpr (std::is_same_v<AccType, DstType>) {
    // If the accumulator and dst types are the same, we can just store
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      st_global<EltPerPack * sizeof(DstType)>(
          iter.template get<FirstDstIdx>(), acc[u]);
      iter.template advanceUnroll<FirstDstIdx>();
    }
  } else {
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      auto destPack = applyCast<AccType, DstType>(acc[u]);
      st_global<EltPerPack * sizeof(DstType)>(
          iter.template get<FirstDstIdx>(), destPack);
      iter.template advanceUnroll<FirstDstIdx>();
    }
  }
}

// Inner reduce-copy using packs of EltPerPack elements.
// Ts... are ordered as {Src0Type, [Src1Type, [Src2Type,]] DstType}.
// The last DstPtrCount pointers are destinations, the rest are sources.
template <
    int Unroll,
    int EltPerPack,
    typename AccType, // What type to use for the accumulator
    size_t DstPtrCount, // The last N pointers are dsts, others are srcs
    typename IntThread,
    typename IntBytes,
    typename... Ts>
__device__ __forceinline__ void reduceCopyPacks(
    IntThread nThreads,
    IntThread& thread,
    IntBytes& nEltsBehind,
    IntBytes& nEltsAhead,
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

  CopyIterator<Unroll, EltPerPack, IntBytes, IntThread, Ts...> iter(
      nThreads, thread, nEltsBehind, nEltsAhead, ptrs...);

  using Iter = decltype(iter);
  using Src0Type = typename Iter::template PtrType<0>;
  using DstType = typename Iter::template PtrType<kDstStartIdx>;

  while (iter.hasWork()) {
    BytePack<EltPerPack * sizeof(AccType)> acc[Unroll];

    loadFirstSource<Unroll, EltPerPack, AccType>(acc, iter);

    ReduceSources<1, kSrcPtrCount>::template apply<Unroll, EltPerPack, AccType>(
        acc, iter);

    storeFirstDestination<Unroll, EltPerPack, AccType, kDstStartIdx>(acc, iter);

    iter.advance();
  }
}

// Reduce-sum nElts elements from multiple sources into a single destination.
// Destination pointer comes first, followed by variadic source pointers.
// Uses the same multi-pass vectorized packing strategy as copy():
//   1. Try 16-byte packs if all pointers are aligned (bulk + tail).
//   2. Fall back to sizeof(AccType)-byte packs (bulk + tail).
template <int Unroll, typename AccType, typename IntBytes, typename... SrcTs>
__device__ __forceinline__ void reduceCopy(
    int thread,
    int nThreads,
    void* dstPtr,
    IntBytes nElts,
    SrcTs*... srcPtrs) {
  int lane = thread % WARP_SIZE;
  constexpr int BigPackSize = 16;
  constexpr size_t kNSrcs = sizeof...(SrcTs);
  static_assert(kNSrcs > 0, "reduceCopy requires at least one source");

  IntBytes nEltsBehind = 0;
  IntBytes nEltsAhead = nElts;

  AccType* dst = static_cast<AccType*>(dstPtr);

  if constexpr (BigPackSize > sizeof(AccType)) {
    bool aligned = true;
    if (lane == 0) {
      aligned &= 0 == cvta_to_global(dstPtr) % BigPackSize;
      // Check alignment of all source pointers.
      ((aligned &= 0 == cvta_to_global(srcPtrs) % BigPackSize), ...);
    }
    aligned = __all_sync(~0u, aligned);
    if (aligned) {
      reduceCopyPacks<Unroll, BigPackSize / sizeof(AccType), AccType, 1>(
          nThreads, thread, nEltsBehind, nEltsAhead, srcPtrs..., dst);
      if (nEltsAhead == 0)
        return;

      reduceCopyPacks<1, BigPackSize / sizeof(AccType), AccType, 1>(
          nThreads, thread, nEltsBehind, nEltsAhead, srcPtrs..., dst);
      if (nEltsAhead == 0)
        return;
    }
  }

  reduceCopyPacks<Unroll * (16 / sizeof(AccType)) / 2, 1, AccType, 1>(
      nThreads, thread, nEltsBehind, nEltsAhead, srcPtrs..., dst);
  if (nEltsAhead == 0)
    return;

  reduceCopyPacks<1, 1, AccType, 1>(
      nThreads, thread, nEltsBehind, nEltsAhead, srcPtrs..., dst);
}

// Copy nElts elements of type T from srcPtr to dstPtr.
// Uses a multi-pass vectorized packing strategy:
//   1. Try 16-byte packs if pointers are aligned (bulk + tail).
//   2. Fall back to sizeof(T)-byte packs (bulk + tail).
template <int Unroll, typename T, typename IntBytes>
__device__ __forceinline__ void
copy(int thread, int nThreads, void* srcPtr, void* dstPtr, IntBytes nElts) {
  int lane = thread % WARP_SIZE;
  constexpr int BigPackSize = 16;

  IntBytes nEltsBehind = 0;
  IntBytes nEltsAhead = nElts;

  T* src = static_cast<T*>(srcPtr);
  T* dst = static_cast<T*>(dstPtr);

  if constexpr (BigPackSize > sizeof(T)) {
    // Check that both pointers are BigPackSize-aligned.
    bool aligned = true;
    if (lane == 0) {
      aligned &= 0 == cvta_to_global(srcPtr) % BigPackSize;
      aligned &= 0 == cvta_to_global(dstPtr) % BigPackSize;
    }
    aligned = __all_sync(~0u, aligned);
    if (aligned) {
      reduceCopyPacks<Unroll, BigPackSize / sizeof(T), T, 1>(
          nThreads, thread, nEltsBehind, nEltsAhead, src, dst);
      if (nEltsAhead == 0)
        return;

      reduceCopyPacks<1, BigPackSize / sizeof(T), T, 1>(
          nThreads, thread, nEltsBehind, nEltsAhead, src, dst);
      if (nEltsAhead == 0)
        return;
    }
  }

  reduceCopyPacks<Unroll * (16 / sizeof(T)) / 2, 1, T, 1>(
      nThreads, thread, nEltsBehind, nEltsAhead, src, dst);
  if (nEltsAhead == 0)
    return;

  reduceCopyPacks<1, 1, T, 1>(
      nThreads, thread, nEltsBehind, nEltsAhead, src, dst);
}

// Reduce-sum nElts elements from multiple sources into a single destination.
// Destination pointer comes first, followed by variadic source pointers.
// Uses the same multi-pass vectorized packing strategy as copy():
//   1. Try 16-byte packs if all pointers are aligned (bulk + tail).
//   2. Fall back to sizeof(AccType)-byte packs (bulk + tail).
template <
    int Unroll,
    typename AccType,
    typename DstType,
    typename IntBytes,
    typename... SrcTs>
__device__ __forceinline__ void reduceCopyMixed(
    int thread,
    int nThreads,
    DstType* dstPtr,
    IntBytes nElts,
    SrcTs*... srcPtrs) {
  int lane = thread % WARP_SIZE;
  constexpr int BigPackSize = 16;
  constexpr size_t kNSrcs = sizeof...(SrcTs);
  static_assert(kNSrcs > 0, "reduceCopy requires at least one source");

  IntBytes nEltsBehind = 0;
  IntBytes nEltsAhead = nElts;

  if constexpr (BigPackSize > sizeof(AccType)) {
    bool aligned = true;
    if (lane == 0) {
      aligned &= 0 == cvta_to_global(dstPtr) % BigPackSize;
      // Check alignment of all source pointers.
      ((aligned &= 0 == cvta_to_global(srcPtrs) % BigPackSize), ...);
    }
    aligned = __all_sync(~0u, aligned);
    if (aligned) {
      reduceCopyPacks<Unroll, BigPackSize / sizeof(AccType), AccType, 1>(
          nThreads, thread, nEltsBehind, nEltsAhead, srcPtrs..., dstPtr);
      if (nEltsAhead == 0)
        return;

      reduceCopyPacks<1, BigPackSize / sizeof(AccType), AccType, 1>(
          nThreads, thread, nEltsBehind, nEltsAhead, srcPtrs..., dstPtr);
      if (nEltsAhead == 0)
        return;
    }
  }

  reduceCopyPacks<Unroll * (16 / sizeof(AccType)) / 2, 1, AccType, 1>(
      nThreads, thread, nEltsBehind, nEltsAhead, srcPtrs..., dstPtr);
  if (nEltsAhead == 0)
    return;

  reduceCopyPacks<1, 1, AccType, 1>(
      nThreads, thread, nEltsBehind, nEltsAhead, srcPtrs..., dstPtr);
}

} // namespace meta::comms::ncclx::kernels::simplecopy
