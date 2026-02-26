#ifndef NCCL_DEVICE_PHILOX_RNG_H_
#define NCCL_DEVICE_PHILOX_RNG_H_

#include <cstdint>

// The code below was copied from: https://fburl.com/vcg6rcfk (lines 35-109)
// (in github, access required)

// ==============================================================================
// Philox 4x32 Counter-Based RNG (7 rounds - minimum to pass BigCrush)
// ==============================================================================
// Based on Triton's implementation:
// https://github.com/triton-lang/triton/blob/main/python/triton/language/random.py
// Philox is a counter-based RNG that produces statistically high-quality
// random numbers. 7 rounds is the minimum proven to pass BigCrush tests.

// Philox constants for 32-bit version
constexpr uint32_t PHILOX_KEY_A = 0x9E3779B9;
constexpr uint32_t PHILOX_KEY_B = 0xBB67AE85;
constexpr uint32_t PHILOX_ROUND_A = 0xD2511F53;
constexpr uint32_t PHILOX_ROUND_B = 0xCD9E8D57;

// Helper: unsigned 32-bit multiply high (upper 32 bits of 64-bit product)
__device__ __forceinline__ uint32_t umulhi32(uint32_t a, uint32_t b) {
  return __umulhi(a, b);
}

// Philox 4x32 implementation with n_rounds
// Takes 4 counters (c0, c1, c2, c3) and 2 keys (k0, k1)
// Returns 4 random uint32 values
template <int N_ROUNDS = 7>
__device__ __forceinline__ void philox4x32(
    uint32_t& c0,
    uint32_t& c1,
    uint32_t& c2,
    uint32_t& c3,
    uint32_t k0,
    uint32_t k1) {
#pragma unroll
  for (int round = 0; round < N_ROUNDS; round++) {
    // Save old values
    uint32_t _c0 = c0;
    uint32_t _c2 = c2;

    // Update random state using Philox round function
    c0 = umulhi32(PHILOX_ROUND_B, _c2) ^ c1 ^ k0;
    c2 = umulhi32(PHILOX_ROUND_A, _c0) ^ c3 ^ k1;
    c1 = PHILOX_ROUND_B * _c2;
    c3 = PHILOX_ROUND_A * _c0;

    // Raise key (Weyl sequence)
    k0 += PHILOX_KEY_A;
    k1 += PHILOX_KEY_B;
  }
}

// Generate 4 random uint32 values from seed and offset
// This matches Triton's randint4x function
__device__ __forceinline__ void philox_randint4x(
    uint64_t seed,
    uint64_t offset,
    uint32_t& r0,
    uint32_t& r1,
    uint32_t& r2,
    uint32_t& r3) {
  // Split seed into key (k0, k1)
  uint32_t k0 = static_cast<uint32_t>(seed);
  uint32_t k1 = static_cast<uint32_t>(seed >> 32);

  // Split offset into counters (c0, c1), c2 and c3 are 0
  uint32_t c0 = static_cast<uint32_t>(offset);
  uint32_t c1 = static_cast<uint32_t>(offset >> 32);
  uint32_t c2 = 0;
  uint32_t c3 = 0;

  // Run Philox 7 rounds
  philox4x32<7>(c0, c1, c2, c3, k0, k1);

  r0 = c0;
  r1 = c1;
  r2 = c2;
  r3 = c3;
}

#endif // NCCL_DEVICE_PHILOX_RNG_H_
