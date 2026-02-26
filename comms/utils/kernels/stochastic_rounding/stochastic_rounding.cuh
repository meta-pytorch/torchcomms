// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// ==============================================================================
// Stochastic Rounding Primitives for FP32 → BF16
// Reference: https://fburl.com/code/w36pwfzh
// ==============================================================================
// Uses Philox RNG with explicit seed and offset parameters (no state class).
// Offset formula: Philox(seed, baseOffset + prevStep * inputBufferSize +
// elementOffset)

#pragma once

#include <cuda_bf16.h>

// Simple check for whether the number is finite
__device__ __forceinline__ bool isFinite(float val) {
  // Check for special values (NaN or Infinity).
  // In IEEE 754 FP32, the exponent field (bits 23-30) is all 1s for NaN/Inf.
  // The bit pattern 0x7F800000 isolates the exponent field.
  // If (x & 0x7F800000) == 0x7F800000, the value is NaN or Infinity.
  // We should propagate these values without stochastic rounding.
  constexpr uint32_t kExpMask = 0x7f800000;

  uint32_t bits = __float_as_uint(val);
  return (bits & kExpMask) != kExpMask;
}

// ==============================================================================
// Hardware-Accelerated Stochastic Rounding (Blackwell, SM >= 100)
// ==============================================================================
// Uses inline PTX assembly with the native cvt.rs.satfinite.bf16x2.f32
// instruction for hardware-accelerated stochastic rounding.
// Takes 2 FP32 values + random bits and produces 2 BF16 values packed in
// uint32.
#if __CUDA_ARCH__ >= 1000
__device__ __forceinline__ __nv_bfloat162
stochastic_round_bf16x2_blackwell(float2 vals, uint32_t rand_bits) {
  // Use cvt.rs.bf16x2.f32 for vectorized stochastic rounding
  // PTX instruction: cvt.rs.bf16x2.f32 dst, src_hi, src_lo, rand;
  // - dst: 32-bit register containing 2 packed BF16 values
  // - src_hi: higher FP32 value (will be in upper 16 bits of dst)
  // - src_lo: lower FP32 value (will be in lower 16 bits of dst)
  // - rand: 32-bit random value for entropy
  //
  // With pack=2:
  // - $1, $2 receive 2 consecutive FP32 elements from x
  // - $3, $4 receive 2 consecutive uint32 elements from entropy
  // - $0 is the output (2 packed BF16 values in uint32)
  //
  // Since arguments are in reverse order for inline_asm_elementwise
  // (lower indices in lower bits), we use {$2, $1} for the two FP32 values.
  // We use $3 for entropy (same entropy for both elements in pack).
  //
  // Constraints: =r for output, r for each of the 4 inputs (2 from x, 2 from
  // entropy)
  uint32_t out;
  asm volatile("{ cvt.rs.bf16x2.f32 %0, %2, %1, %3; }"
               : "=r"(out)
               : "f"(vals.x), "f"(vals.y), "r"(rand_bits));
  return *reinterpret_cast<__nv_bfloat162*>(&out);
}
#endif

// ==============================================================================
// Software Stochastic Rounding (Pre-Blackwell Fallback)
// ==============================================================================
// BF16 has same exponent as FP32 (8 bits) but only 7 mantissa bits vs 23.
// We add random bits to the lower 16 bits before truncating.
__device__ __forceinline__ __nv_bfloat16
stochastic_round_bf16_software(float val, uint32_t rand_bits) {
  // For non-finite values, we cast to BF16 directly (no rounding)
  if (!isFinite(val)) {
    return __float2bfloat16(val);
  }

  uint16_t out;
  asm("{\n\t"
      "  .reg .b32 r;\n\t"
      "  .reg .b16 a, b;\n\t"
      "  mov.b32 {a, b}, %2;\n\t" // Split rand number: a=low 16 bits, b=high 16
                                  // bits
      "  mov.b32 r, {a, 0};\n\t" // r = low 16 bits in lower half, zeros in
                                 // upper
      "  add.s32 r, %1, r;\n\t" // Add random bits to val, carry propagates up
      "  mov.b32 {a, b}, r;\n\t" // Split result: a=low, b=high (our BF16)
      "  mov.b16 %0, b;\n\t" // Output the high 16 bits as BF16
      "}\n\t"
      : "=h"(out)
      : "r"(__float_as_uint(val)), "r"(rand_bits));
  return __ushort_as_bfloat16(out);
}

// Vectorized software version for 2 elements (matches BF16x2)
__device__ __forceinline__ __nv_bfloat162
stochastic_round_bf16x2_software(float2 vals, uint32_t r0, uint32_t r1) {
  __nv_bfloat16 lo = stochastic_round_bf16_software(vals.x, r0);
  __nv_bfloat16 hi = stochastic_round_bf16_software(vals.y, r1);
  return __nv_bfloat162(lo, hi);
}
