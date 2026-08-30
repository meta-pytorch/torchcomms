// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "device.h"
#include "op128.h"

#include "comms/utils/kernels/rng/philox_rng.cuh"
#include "comms/utils/kernels/stochastic_rounding/stochastic_rounding.cuh"

// ==============================================================================
// Apply_StochasticCast Templates
// All take seed and offset EXPLICITLY (no state class)
// ==============================================================================

template <typename SrcType, typename DstType, int EltPerPack>
struct Apply_StochasticCast;

// ============================================================================
// FP32 -> BF16 Specializations (Primary Use Case)
// ============================================================================

// Specialization for float -> __nv_bfloat16, 1 element
template <>
struct Apply_StochasticCast<float, __nv_bfloat16, 1> {
  __device__ __forceinline__ static BytePack<2> // sizeof(bf16) = 2 bytes
  cast(
      BytePack<4> a,
      uint64_t seed,
      uint64_t offset) { // sizeof(float) = 4 bytes
    float val = fromPack<float>(a);
    uint32_t r0, r1, r2, r3;
    philox_randint4x(seed, offset, r0, r1, r2, r3);
    __nv_bfloat16 result = stochastic_round_bf16_software(val, r0);
    return toPack(result);
  }
};

// Specialization for float -> __nv_bfloat16, 2 elements (optimal vectorized
// path)
template <>
struct Apply_StochasticCast<float, __nv_bfloat16, 2> {
  __device__ __forceinline__ static BytePack<4> // 2 * sizeof(bf16) = 4 bytes
  cast(
      BytePack<8> a,
      uint64_t seed,
      uint64_t offset) { // 2 * sizeof(float) = 8 bytes
    float2 vals = fromPack<float2>(a);
    uint32_t r0, r1, r2, r3;
    philox_randint4x(seed, offset, r0, r1, r2, r3);

#if __CUDA_ARCH__ >= 1000
    // Blackwell: use native hardware instruction
    // Combine random bits: use XOR for entropy mixing
    uint32_t rand_bits = r0 ^ (r1 << 16);
    __nv_bfloat162 result = stochastic_round_bf16x2_blackwell(vals, rand_bits);
#else
    // Pre-Blackwell: software fallback
    __nv_bfloat162 result = stochastic_round_bf16x2_software(vals, r0, r1);
#endif
    return toPack(result);
  }
};

// Specialization for float -> __nv_bfloat16, 4 elements
template <>
struct Apply_StochasticCast<float, __nv_bfloat16, 4> {
  __device__ __forceinline__ static BytePack<8> // 4 * sizeof(bf16) = 8 bytes
  cast(
      BytePack<16> a,
      uint64_t seed,
      uint64_t offset) { // 4 * sizeof(float) = 16 bytes
    float4 vals = fromPack<float4>(a);
    uint32_t r0, r1, r2, r3;
    philox_randint4x(seed, offset, r0, r1, r2, r3);

    __nv_bfloat162 lo, hi;
#if __CUDA_ARCH__ >= 1000
    // Blackwell: use native hardware instruction for each pair
    uint32_t rand_lo = r0 ^ (r1 << 16);
    uint32_t rand_hi = r2 ^ (r3 << 16);
    lo =
        stochastic_round_bf16x2_blackwell(make_float2(vals.x, vals.y), rand_lo);
    hi =
        stochastic_round_bf16x2_blackwell(make_float2(vals.z, vals.w), rand_hi);
#else
    // Pre-Blackwell: software fallback
    lo = stochastic_round_bf16x2_software(make_float2(vals.x, vals.y), r0, r1);
    hi = stochastic_round_bf16x2_software(make_float2(vals.z, vals.w), r2, r3);
#endif
    // Pack into BytePack<8>
    BytePack<8> result;
    result.half[0] = toPack(lo);
    result.half[1] = toPack(hi);
    return result;
  }
};

// ============================================================================
// Recursive/General Case
// ============================================================================

// General recursive case: split pack in half
template <typename SrcType, typename DstType, int EltPerPack>
struct Apply_StochasticCast {
  __device__ __forceinline__ static BytePack<EltPerPack * sizeof(DstType)> cast(
      BytePack<EltPerPack * sizeof(SrcType)> a,
      uint64_t seed,
      uint64_t offset) {
    BytePack<EltPerPack * sizeof(DstType)> result;
    result.half[0] =
        Apply_StochasticCast<SrcType, DstType, EltPerPack / 2>::cast(
            a.half[0], seed, offset);
    result.half[1] =
        Apply_StochasticCast<SrcType, DstType, EltPerPack / 2>::cast(
            a.half[1], seed, offset + EltPerPack / 2);
    return result;
  }
};

// ============================================================================
// Public API
// ============================================================================

// Apply stochastic cast from SrcType to DstType with explicit seed and offset
// Usage: applyStochasticCast<SrcType, DstType>(pack, seed, offset)
template <typename SrcType, typename DstType, typename Pack>
__device__ __forceinline__
    BytePack<BytePackOf<Pack>::Size * sizeof(DstType) / sizeof(SrcType)>
    applyStochasticCast(Pack a, uint64_t seed, uint64_t offset) {
  return Apply_StochasticCast<
      SrcType,
      DstType,
      BytePackOf<Pack>::Size / sizeof(SrcType)>::cast(toPack(a), seed, offset);
}
