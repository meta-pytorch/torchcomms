// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_COMMON_KERNEL_QUANTIZE_H_
#define NCCL_COMMON_KERNEL_QUANTIZE_H_

#include <cassert>

#include "common_kernel.h"
#include "meta/collectives/kernels/stochastic_cast.cuh"
#include "reduce_kernel.h"

////////////////////////////////////////////////////////////////////////////////
// Mixed-precision reduceCopy for quantized collectives.
//
// This module provides functions for reduce-copy operations where sources
// and destinations may have different precisions:
// - AccumType: Higher precision type for input/output and accumulation (e.g.,
// float)
// - TransportType: Lower precision type for transport buffers (e.g., bf16)
//
// The pipeline:
// 1. Loads each source in its native precision
// 2. Converts to AccumType for reduction
// 3. Reduces in AccumType
// 4. Converts to destination precision (with optional stochastic rounding)
// 5. Stores to destination
////////////////////////////////////////////////////////////////////////////////

// Type conversion helpers for mixed-precision operations
template <typename DstType, typename SrcType>
__device__ __forceinline__ DstType convertType(SrcType val);

template <>
__device__ __forceinline__ float convertType<float, float>(float val) {
  return val;
}

template <>
__device__ __forceinline__ float convertType<float, __nv_bfloat16>(
    __nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <>
__device__ __forceinline__ float convertType<float, __half>(__half val) {
  return __half2float(val);
}

template <>
__device__ __forceinline__ __nv_bfloat16
convertType<__nv_bfloat16, float>(float val) {
  return __float2bfloat16(val);
}

template <>
__device__ __forceinline__ __nv_bfloat16
convertType<__nv_bfloat16, __nv_bfloat16>(__nv_bfloat16 val) {
  return val;
}

template <>
__device__ __forceinline__ __half convertType<__half, float>(float val) {
  return __float2half(val);
}

template <>
__device__ __forceinline__ __half convertType<__half, __half>(__half val) {
  return val;
}

// Stochastic rounding conversion from AccumType to TransportType
// Uses random noise to round probabilistically based on the fractional part
template <typename TransportType, typename AccumType>
__device__ __forceinline__ TransportType
convertWithStochasticRounding(AccumType val, uint32_t randomBits);

template <>
__device__ __forceinline__ __nv_bfloat16
convertWithStochasticRounding<__nv_bfloat16, float>(
    float val,
    uint32_t randomBits) {
  // BF16 has 7 mantissa bits, float has 23 mantissa bits
  // We need to add noise to the 16 bits that will be truncated
  union {
    float f;
    uint32_t u;
  } pun;
  pun.f = val;
  // Add random noise to the lower 16 bits (the part that gets truncated)
  // This implements stochastic rounding: round up with probability proportional
  // to the fractional part
  uint32_t noise = randomBits & 0xFFFF;
  pun.u += noise;
  return __float2bfloat16(pun.f);
}

template <>
__device__ __forceinline__ __half
convertWithStochasticRounding<__half, float>(float val, uint32_t randomBits) {
  // FP16 has 10 mantissa bits, float has 23 mantissa bits
  // We need to add noise to the 13 bits that will be truncated
  union {
    float f;
    uint32_t u;
  } pun;
  pun.f = val;
  // Add random noise to the lower 13 bits
  uint32_t noise = randomBits & 0x1FFF;
  pun.u += noise;
  return __float2half(pun.f);
}

// Load a single element from source, converting to AccumType
template <typename AccumType, typename SrcType>
__device__ __forceinline__ AccumType loadAndConvert(SrcType* ptr, int64_t idx) {
  return convertType<AccumType, SrcType>(ptr[idx]);
}

// Store a single element to destination, converting from AccumType
template <typename DstType, typename AccumType>
__device__ __forceinline__ void
convertAndStore(DstType* ptr, int64_t idx, AccumType val) {
  DstType dst = convertType<DstType, AccumType>(val);
  ptr[idx] = dst;
}

// Store with stochastic rounding
template <typename DstType, typename AccumType>
__device__ __forceinline__ void convertAndStoreStochastic(
    DstType* ptr,
    int64_t idx,
    AccumType val,
    uint32_t randomBits) {
  DstType dst =
      convertWithStochasticRounding<DstType, AccumType>(val, randomBits);
  ptr[idx] = dst;
}

// Apply reduction in AccumType precision
template <typename RedFn, typename AccumType>
__device__ __forceinline__ AccumType
reduceAccum(RedFn& redFn, AccumType a, AccumType b) {
  BytePack<sizeof(AccumType)> packA = toPack(a);
  BytePack<sizeof(AccumType)> packB = toPack(b);
  BytePack<sizeof(AccumType)> result = applyReduce(redFn, packA, packB);
  return fromPack<AccumType>(result);
}

// Convert a pack of TransportType elements to a pack of AccumType elements.
// Used by the vectorized path to convert after a packed load.
template <typename AccumType, typename TransportType, int PackElts>
__device__ __forceinline__ BytePack<PackElts * sizeof(AccumType)>
convertPackToAccum(BytePack<PackElts * sizeof(TransportType)> src) {
  BytePack<PackElts * sizeof(AccumType)> result;
#pragma unroll
  for (int i = 0; i < PackElts; i++) {
    TransportType srcElt;
    memcpy(
        &srcElt,
        reinterpret_cast<const char*>(&src) + i * sizeof(TransportType),
        sizeof(TransportType));
    AccumType dstElt = convertType<AccumType, TransportType>(srcElt);
    memcpy(
        reinterpret_cast<char*>(&result) + i * sizeof(AccumType),
        &dstElt,
        sizeof(AccumType));
  }
  return result;
}

// Load PackElts elements one-by-one from a potentially misaligned global
// address, converting each from SrcType to AccumType, and assemble into
// an AccumType-sized BytePack. Each ld_volatile_global<sizeof(SrcType)> is
// naturally aligned for a valid SrcType pointer (sizeof-aligned).
template <typename AccumType, typename SrcType, int PackElts>
__device__ __forceinline__ BytePack<PackElts * sizeof(AccumType)>
ld_volatile_global_elements(uintptr_t addr) {
  BytePack<PackElts * sizeof(AccumType)> result;
#pragma unroll
  for (int i = 0; i < PackElts; i++) {
    BytePack<sizeof(SrcType)> elem =
        ld_volatile_global<sizeof(SrcType)>(addr + i * sizeof(SrcType));
    AccumType val = convertType<AccumType, SrcType>(fromPack<SrcType>(elem));
    memcpy(
        reinterpret_cast<char*>(&result) + i * sizeof(AccumType),
        &val,
        sizeof(AccumType));
  }
  return result;
}

// Store PackElts elements one-by-one to a potentially misaligned global
// address from a DstType-sized BytePack.
template <typename DstType, int PackElts>
__device__ __forceinline__ void st_global_elements(
    uintptr_t addr,
    BytePack<PackElts * sizeof(DstType)> pack) {
#pragma unroll
  for (int i = 0; i < PackElts; i++) {
    BytePack<sizeof(DstType)> elem;
    memcpy(
        &elem,
        reinterpret_cast<const char*>(&pack) + i * sizeof(DstType),
        sizeof(DstType));
    st_global<sizeof(DstType)>(addr + i * sizeof(DstType), elem);
  }
}

// Core mixed-precision reduce-copy implementation with compile-time type
// selection.
//
// Optimizations borrowed from reduceCopy (common_kernel.h):
//   1. Vectorized loads/stores via ld_volatile_global<N>/st_global<N> (up to
//   128-bit)
//   2. Pre-computed global-segment addresses via cvta_to_global()
//   3. Hunk-strided work distribution across warps
//   4. Separate load and reduce phases for memory pipelining
//   5. Batched stochastic rounding via Apply_StochasticCast<..., PackElts>
template <
    int Unroll,
    typename RedFn,
    typename AccumType,
    typename TransportType,
    bool Src0IsAccumType,
    bool Src1IsAccumType,
    bool Dst0IsAccumType,
    typename IntElts>
__device__ __forceinline__ void reduceCopyMixedImpl(
    int thread,
    int nThreads,
    uint64_t redArg,
    int nSrcs,
    void** srcs,
    int nDsts,
    void** dsts,
    IntElts nElts,
    uint64_t randomSeed,
    uint64_t randomBaseOffset) {
  static_assert(
      std::is_signed<IntElts>::value,
      "IntElts must be a signed integral type.");

  if (nDsts == 0 || nElts <= 0) {
    return;
  }

  // Doesn't allow nSrcs == 0, that would be an invalid case
  if (nSrcs == 0) {
    __trap();
  }

  // nThreads must be at least WARP_SIZE; the warp-based work distribution
  // below divides by nWarps, so fewer threads causes division by zero.
  // We also expect nThreads to be a multiple of WARP_SIZE.
  if (nThreads < WARP_SIZE || nThreads % WARP_SIZE != 0) {
    __trap();
  }

  using SrcType0 = typename std::
      conditional<Src0IsAccumType, AccumType, TransportType>::type;
  using SrcType1 = typename std::
      conditional<Src1IsAccumType, AccumType, TransportType>::type;
  using DstType0 = typename std::
      conditional<Dst0IsAccumType, AccumType, TransportType>::type;

  // Pack multiple elements per memory transaction for better bandwidth.
  // PackElts=4 gives 128-bit (16-byte) loads for float, 64-bit (8-byte) for
  // bf16.
  constexpr int PackElts = 16 / sizeof(AccumType);
  constexpr int AccumPackBytes = PackElts * sizeof(AccumType);
  constexpr int SrcPack0Bytes = PackElts * sizeof(SrcType0);
  constexpr int SrcPack1Bytes = PackElts * sizeof(SrcType1);
  constexpr int DstPack0Bytes = PackElts * sizeof(DstType0);

  // A "hunk" is the contiguous data a warp processes per loop iteration.
  constexpr int ElemsPerHunk = Unroll * WARP_SIZE * PackElts;

  int nWarps = nThreads / WARP_SIZE;
  int warp = thread / WARP_SIZE;
  int lane = thread % WARP_SIZE;

  RedFn redFn(redArg);

  // Pre-compute global-segment addresses for vectorized PTX loads/stores.
  uintptr_t src0Addr = nSrcs > 0 ? cvta_to_global(srcs[0]) : 0;
  uintptr_t src1Addr = nSrcs > 1 ? cvta_to_global(srcs[1]) : 0;
  uintptr_t dst0Addr = nDsts > 0 ? cvta_to_global(dsts[0]) : 0;

  // This thread's initial element offset within the first hunk.
  IntElts threadEltBase =
      IntElts(warp) * IntElts(ElemsPerHunk) + IntElts(lane) * IntElts(PackElts);

  // Number of complete hunks and elements they cover.
  IntElts nHunksTotal = nElts / IntElts(ElemsPerHunk);
  IntElts packedElts = nHunksTotal * IntElts(ElemsPerHunk);

  // Per-buffer alignment checks. Misaligned buffers use element-wise
  // loads/stores via ld_volatile_global_elements/st_global_elements;
  // aligned buffers use vectorized ld_volatile_global<N>/st_global<N>.
  // The vectorized loop always runs regardless of alignment.
  bool src0Aligned = nSrcs == 0 || (src0Addr % SrcPack0Bytes == 0);
  bool src1Aligned = nSrcs <= 1 || (src1Addr % SrcPack1Bytes == 0);
  bool dst0Aligned = nDsts == 0 || (dst0Addr % DstPack0Bytes == 0);

  // Hunk-strided loop: warps interleave through hunks for balanced work.
  IntElts hunksRemaining = nHunksTotal - IntElts(warp);
  IntElts eltOffset = threadEltBase;
  IntElts strideElts = IntElts(nWarps) * IntElts(ElemsPerHunk);

  while (hunksRemaining > 0) {
    BytePack<AccumPackBytes> acc[Unroll];

    // Load source 0 into acc[]
    if (nSrcs > 0) {
#pragma unroll Unroll
      for (int u = 0; u < Unroll; u++) {
        IntElts eidx = eltOffset + IntElts(u) * IntElts(WARP_SIZE * PackElts);
        uintptr_t loadAddr = src0Addr + eidx * IntElts(sizeof(SrcType0));
        if (src0Aligned) {
          if constexpr (Src0IsAccumType) {
            acc[u] = ld_volatile_global<AccumPackBytes>(loadAddr);
          } else {
            BytePack<SrcPack0Bytes> srcPack =
                ld_volatile_global<SrcPack0Bytes>(loadAddr);
            acc[u] =
                convertPackToAccum<AccumType, TransportType, PackElts>(srcPack);
          }
        } else {
          acc[u] = ld_volatile_global_elements<AccumType, SrcType0, PackElts>(
              loadAddr);
        }
      }
    }

    // Load source 1 into tmp[], then reduce into acc[]
    if (nSrcs > 1) {
      BytePack<AccumPackBytes> tmp[Unroll];
#pragma unroll Unroll
      for (int u = 0; u < Unroll; u++) {
        IntElts eidx = eltOffset + IntElts(u) * IntElts(WARP_SIZE * PackElts);
        uintptr_t loadAddr = src1Addr + eidx * IntElts(sizeof(SrcType1));
        if (src1Aligned) {
          if constexpr (Src1IsAccumType) {
            tmp[u] = ld_volatile_global<AccumPackBytes>(loadAddr);
          } else {
            BytePack<SrcPack1Bytes> srcPack =
                ld_volatile_global<SrcPack1Bytes>(loadAddr);
            tmp[u] =
                convertPackToAccum<AccumType, TransportType, PackElts>(srcPack);
          }
        } else {
          tmp[u] = ld_volatile_global_elements<AccumType, SrcType1, PackElts>(
              loadAddr);
        }
      }
#pragma unroll Unroll
      for (int u = 0; u < Unroll; u++) {
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    // Store to destination
    if (nDsts > 0) {
#pragma unroll Unroll
      for (int u = 0; u < Unroll; u++) {
        IntElts eidx = eltOffset + IntElts(u) * IntElts(WARP_SIZE * PackElts);
        uintptr_t storeAddr = dst0Addr + eidx * IntElts(sizeof(DstType0));
        if constexpr (Dst0IsAccumType) {
          if (dst0Aligned) {
            st_global<AccumPackBytes>(storeAddr, acc[u]);
          } else {
            st_global_elements<DstType0, PackElts>(storeAddr, acc[u]);
          }
        } else {
          uint64_t elemIdx = randomBaseOffset + static_cast<uint64_t>(eidx);
          BytePack<DstPack0Bytes> dstPack =
              Apply_StochasticCast<AccumType, DstType0, PackElts>::cast(
                  acc[u], randomSeed, elemIdx);
          if (dst0Aligned) {
            st_global<DstPack0Bytes>(storeAddr, dstPack);
          } else {
            st_global_elements<DstType0, PackElts>(storeAddr, dstPack);
          }
        }
      }
    }

    eltOffset += strideElts;
    hunksRemaining -= IntElts(nWarps);
  }

  // Handle remaining elements (< ElemsPerHunk) with scalar access.
  for (IntElts idx = packedElts + IntElts(thread); idx < nElts;
       idx += IntElts(nThreads)) {
    AccumType acc;
    bool hasValue = false;

    if (nSrcs > 0) {
      acc = loadAndConvert<AccumType, SrcType0>(
          static_cast<SrcType0*>(srcs[0]), idx);
      hasValue = true;
    }

    if (nSrcs > 1) {
      AccumType val1 = loadAndConvert<AccumType, SrcType1>(
          static_cast<SrcType1*>(srcs[1]), idx);
      if (hasValue) {
        acc = reduceAccum(redFn, acc, val1);
      } else {
        acc = val1;
        hasValue = true;
      }
    }

    if (hasValue) {
      DstType0* dst0Ptr = static_cast<DstType0*>(dsts[0]);
      if constexpr (Dst0IsAccumType) {
        dst0Ptr[idx] = acc;
      } else {
        uint64_t offset = randomBaseOffset + static_cast<uint64_t>(idx);
        BytePack<sizeof(AccumType)> accPack = toPack(acc);
        BytePack<sizeof(DstType0)> dstPack =
            Apply_StochasticCast<AccumType, DstType0, 1>::cast(
                accPack, randomSeed, offset);
        dst0Ptr[idx] = fromPack<DstType0>(dstPack);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Clean dispatch interface for patReduce
//
// This function takes runtime booleans for source/destination types and
// dispatches to the appropriate template instantiation internally.
// The caller doesn't need to handle the dispatch logic.
//
// Parameters:
//   thread, nThreads: Thread indexing
//   redArg: Reduction operation argument
//   nSrcs: Number of sources (1 or 2)
//   srcs: Array of source pointers
//   nDsts: Number of destinations (typically 1)
//   dsts: Array of destination pointers
//   nElts: Number of elements to process
//   src0IsAccumType: true if srcs[0] uses AccumType (high precision)
//   src1IsAccumType: true if srcs[1] uses AccumType (only used if nSrcs > 1)
//   dst0IsAccumType: true if dsts[0] uses AccumType
//   randomSeed: Seed for Philox RNG (used for stochastic rounding when dst is
//   TransportType) randomBaseOffset: Base offset for Philox RNG (typically
//   prevStep * bufferSize)
////////////////////////////////////////////////////////////////////////////////
template <
    int Unroll,
    typename RedFn,
    typename AccumType,
    typename TransportType,
    typename IntElts>
__device__ __forceinline__ void reduceCopyMixed(
    int thread,
    int nThreads,
    uint64_t redArg,
    int nSrcs,
    void** srcs,
    int nDsts,
    void** dsts,
    IntElts nElts,
    bool src0IsAccumType,
    bool src1IsAccumType,
    bool dst0IsAccumType,
    uint64_t randomSeed,
    uint64_t randomBaseOffset) {
  // Dispatch based on runtime type flags
  // We encode the 3 booleans as a 3-bit integer for cleaner dispatch
  int typeConfig = (src0IsAccumType ? 4 : 0) | (src1IsAccumType ? 2 : 0) |
      (dst0IsAccumType ? 1 : 0);

  switch (typeConfig) {
    case 0: // src0=Transport, src1=Transport, dst0=Transport
      reduceCopyMixedImpl<
          Unroll,
          RedFn,
          AccumType,
          TransportType,
          false,
          false,
          false>(
          thread,
          nThreads,
          redArg,
          nSrcs,
          srcs,
          nDsts,
          dsts,
          nElts,
          randomSeed,
          randomBaseOffset);
      break;
    case 1: // src0=Transport, src1=Transport, dst0=Accum
      reduceCopyMixedImpl<
          Unroll,
          RedFn,
          AccumType,
          TransportType,
          false,
          false,
          true>(
          thread,
          nThreads,
          redArg,
          nSrcs,
          srcs,
          nDsts,
          dsts,
          nElts,
          randomSeed,
          randomBaseOffset);
      break;
    case 2: // src0=Transport, src1=Accum, dst0=Transport
      reduceCopyMixedImpl<
          Unroll,
          RedFn,
          AccumType,
          TransportType,
          false,
          true,
          false>(
          thread,
          nThreads,
          redArg,
          nSrcs,
          srcs,
          nDsts,
          dsts,
          nElts,
          randomSeed,
          randomBaseOffset);
      break;
    case 3: // src0=Transport, src1=Accum, dst0=Accum
      reduceCopyMixedImpl<
          Unroll,
          RedFn,
          AccumType,
          TransportType,
          false,
          true,
          true>(
          thread,
          nThreads,
          redArg,
          nSrcs,
          srcs,
          nDsts,
          dsts,
          nElts,
          randomSeed,
          randomBaseOffset);
      break;
    case 4: // src0=Accum, src1=Transport, dst0=Transport
      reduceCopyMixedImpl<
          Unroll,
          RedFn,
          AccumType,
          TransportType,
          true,
          false,
          false>(
          thread,
          nThreads,
          redArg,
          nSrcs,
          srcs,
          nDsts,
          dsts,
          nElts,
          randomSeed,
          randomBaseOffset);
      break;
    case 5: // src0=Accum, src1=Transport, dst0=Accum
      reduceCopyMixedImpl<
          Unroll,
          RedFn,
          AccumType,
          TransportType,
          true,
          false,
          true>(
          thread,
          nThreads,
          redArg,
          nSrcs,
          srcs,
          nDsts,
          dsts,
          nElts,
          randomSeed,
          randomBaseOffset);
      break;
    case 6: // src0=Accum, src1=Accum, dst0=Transport
      reduceCopyMixedImpl<
          Unroll,
          RedFn,
          AccumType,
          TransportType,
          true,
          true,
          false>(
          thread,
          nThreads,
          redArg,
          nSrcs,
          srcs,
          nDsts,
          dsts,
          nElts,
          randomSeed,
          randomBaseOffset);
      break;
    case 7: // src0=Accum, src1=Accum, dst0=Accum
      reduceCopyMixedImpl<
          Unroll,
          RedFn,
          AccumType,
          TransportType,
          true,
          true,
          true>(
          thread,
          nThreads,
          redArg,
          nSrcs,
          srcs,
          nDsts,
          dsts,
          nElts,
          randomSeed,
          randomBaseOffset);
      break;
  }
}

#endif // NCCL_COMMON_KERNEL_QUANTIZE_H_
