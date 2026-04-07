// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_COPY_ITERATOR_CUH_
#define NCCL_COPY_ITERATOR_CUH_

#include "device.h" // WARP_SIZE
#include "op128.h" // BytePack, cvta_to_global, ld_volatile_global, st_global

#include <type_traits>
#include <utility>

namespace meta::comms::ncclx::kernels {

// Extract the I-th type from a parameter pack (avoids <tuple> dependency).
template <size_t I, typename... Pack>
struct NthType;
template <typename T, typename... Rest>
struct NthType<0, T, Rest...> {
  using type = T;
};
template <size_t I, typename T, typename... Rest>
struct NthType<I, T, Rest...> {
  using type = typename NthType<I - 1, Rest...>::type;
};
template <size_t I, typename... Pack>
using NthType_t = typename NthType<I, Pack...>::type;

// Encapsulates iteration bookkeeping for copyPacks: hunk counting, per-thread
// byte offsets, pointer advancement, and warp rotation.  Supports an arbitrary
// number of pointers, each with its own element type (Ts...).
//
// All methods are __forceinline__ so the struct is a source-level abstraction
// only — it compiles to identical register usage as the original inline code.
template <
    int Unroll,
    int EltPerPack,
    typename IntBytes,
    typename IntThread,
    typename... Ts>
struct CopyIterator {
  static constexpr size_t N = sizeof...(Ts);
  static_assert(N > 0, "CopyIterator requires at least one pointer type");

  uintptr_t ptrs[N];
  IntBytes threadEltsAhead;
  IntBytes nHunksAhead;
  IntThread nThreads;

  static constexpr int kEltPerHunk = Unroll * WARP_SIZE * EltPerPack;
  static constexpr bool kDoPartialIter = Unroll == 1;

  // The element type associated with the I-th pointer.
  // std::remove_cv strips const/volatile so that downstream applyCast/fromPack
  // operate on unqualified types (e.g. __nv_bfloat16 rather than
  // const __nv_bfloat16), avoiding deleted-default-constructor errors in the
  // fromPack union.
  template <size_t I>
  using PtrType = std::remove_cv_t<NthType_t<I, Ts...>>;

  // Get the element offset for the current thread from the beginning of the
  // overall element offset
  __device__ __forceinline__ int getThreadEltOffset(int warp, int lane) {
    // warp * kEltPerHunk -> Warp offset in elements from the beginning of the
    // overall offset
    // lane * EltPerPack -> Offset for this specific thread
    return warp * kEltPerHunk + lane * EltPerPack;
  }

  __device__ __forceinline__ CopyIterator(
      IntThread nThreads_,
      IntThread thread,
      IntBytes& nEltsBehind,
      IntBytes& nEltsAhead,
      Ts*... ptrArgs)
      : nThreads{nThreads_} {
    int warp = thread / WARP_SIZE;
    int lane = thread % WARP_SIZE;

    IntBytes threadEltsBehind = nEltsBehind + getThreadEltOffset(warp, lane);
    threadEltsAhead = nEltsAhead - getThreadEltOffset(warp, lane);

    nHunksAhead = nEltsAhead / kEltPerHunk;

    nEltsBehind += nHunksAhead * kEltPerHunk;
    nEltsAhead -= nHunksAhead * kEltPerHunk;

    if (kDoPartialIter && EltPerPack <= nEltsAhead) {
      nHunksAhead += 1;
      nEltsBehind += nEltsAhead - (nEltsAhead % EltPerPack);
      nEltsAhead = nEltsAhead % EltPerPack;
    }

    // Put nHunksAhead in regard to the current warp
    nHunksAhead -= warp;

    // Initialize each pointer with its type-specific byte stride.
    size_t i = 0;
    ((ptrs[i++] = cvta_to_global(ptrArgs) + threadEltsBehind * sizeof(Ts)),
     ...);
  }

  // Access the I-th pointer.
  template <size_t I>
  __device__ __forceinline__ uintptr_t& get() {
    static_assert(I < N, "Pointer index out of range");
    return ptrs[I];
  }

  template <size_t I>
  __device__ __forceinline__ const uintptr_t& get() const {
    static_assert(I < N, "Pointer index out of range");
    return ptrs[I];
  }

  __device__ __forceinline__ bool hasWork() const {
    if constexpr (kDoPartialIter)
      return EltPerPack <= threadEltsAhead;
    else
      return 0 < nHunksAhead;
  }

  // Advance the I-th pointer by one unrolled warp-stride, using the I-th
  // pointer's type for the byte calculation automatically.
  template <size_t I>
  __device__ __forceinline__ void advanceUnroll() {
    ptrs[I] += WARP_SIZE * EltPerPack * sizeof(PtrType<I>);
  }

  // Advance an arbitrary uintptr_t by one unrolled warp-stride with an
  // explicitly specified element type.
  template <typename PtrT>
  __device__ __forceinline__ void advanceUnroll(uintptr_t& ptr) {
    ptr += WARP_SIZE * EltPerPack * sizeof(PtrT);
  }

  // Advance all pointers past the current set of warps (warp rotation).
  __device__ __forceinline__ void advance() {
    int nWarps = nThreads / WARP_SIZE;
    advanceAll(nWarps, std::index_sequence_for<Ts...>{});
    threadEltsAhead -= nWarps * kEltPerHunk;
    nHunksAhead -= nWarps;
  }

 private:
  template <size_t... Is>
  __device__ __forceinline__ void advanceAll(
      int nWarps,
      std::index_sequence<Is...>) {
    ((ptrs[Is] += (nWarps - 1) * kEltPerHunk * sizeof(PtrType<Is>)), ...);
  }
};

} // namespace meta::comms::ncclx::kernels

#endif // NCCL_COPY_ITERATOR_CUH_
