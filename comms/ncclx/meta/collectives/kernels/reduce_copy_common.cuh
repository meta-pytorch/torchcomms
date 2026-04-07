// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_REDUCE_COPY_COMMON_CUH_
#define NCCL_REDUCE_COPY_COMMON_CUH_

#include "meta/collectives/kernels/copy_iterator.cuh"
#include "reduce_kernel.h" // FuncSum, applyReduce, applyCast

namespace meta::comms::ncclx::kernels {

// Compile-time iteration over source pointer indices [Begin, End).
// Loads each source and reduces into the accumulator.
template <size_t Begin, size_t End>
struct ReduceSources {
  template <int Unroll, int EltPerPack, typename AccType, typename IterType>
  static __device__ __forceinline__ void apply(
      BytePack<EltPerPack * sizeof(AccType)>* acc,
      IterType& iter) {
    using SrcType = typename IterType::template PtrType<Begin>;
    // Enforce that we only accumulate from equal-or-lower precision sources
    // into the accumulator type (e.g., fp16/bf16 -> fp32, fp32 -> fp32).
    static_assert(
        sizeof(SrcType) <= sizeof(AccType),
        "SrcType must be of lower or same precision as AccType (by size).");

    BytePack<EltPerPack * sizeof(SrcType)> tmp[Unroll];

    auto redFn = FuncSum<AccType>{};

    // Load from the current source pointer
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      tmp[u] = ld_volatile_global<EltPerPack * sizeof(SrcType)>(
          iter.template get<Begin>());
      iter.template advanceUnroll<Begin>();
    }

    // Reduce into the accumulation buffer
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      acc[u] = applyReduce(redFn, acc[u], applyCast<SrcType, AccType>(tmp[u]));
    }

    // Recurse to the next source pointer.
    ReduceSources<Begin + 1, End>::template apply<Unroll, EltPerPack, AccType>(
        acc, iter);
  }
};

template <size_t End>
struct ReduceSources<End, End> {
  template <int Unroll, int EltPerPack, typename AccType, typename IterType>
  static __device__ __forceinline__ void apply(
      BytePack<EltPerPack * sizeof(AccType)>*,
      IterType&) {}
};

template <int Unroll, int EltPerPack, typename AccType, typename IterType>
static __device__ __forceinline__ void loadFirstSource(
    BytePack<EltPerPack * sizeof(AccType)>* acc,
    IterType& iter) {
  using SrcType = typename IterType::template PtrType<0>;
  // Enforce that we only accumulate from equal-or-lower precision sources into
  // the accumulator type (e.g., fp16/bf16 -> fp32, fp32 -> fp32).
  static_assert(
      sizeof(SrcType) <= sizeof(AccType),
      "SrcType must be of lower or same precision as AccType (by size).");

  if constexpr (std::is_same_v<AccType, SrcType>) {
    // If the accumulator and source types are the same, we can just load
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      acc[u] = ld_volatile_global<EltPerPack * sizeof(SrcType)>(
          iter.template get<0>());
      iter.template advanceUnroll<0>();
    }
  } else {
    // Otherwise, we need to use extra registers to cast the source to the
    // accumulator type.
    BytePack<EltPerPack * sizeof(SrcType)> tmp[Unroll];
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      tmp[u] = ld_volatile_global<EltPerPack * sizeof(SrcType)>(
          iter.template get<0>());
      iter.template advanceUnroll<0>();
    }

#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      acc[u] = applyCast<SrcType, AccType>(tmp[u]);
    }
  }
}

} // namespace meta::comms::ncclx::kernels

#endif // NCCL_REDUCE_COPY_COMMON_CUH_
