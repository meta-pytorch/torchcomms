// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::bitops {

/**
 * divUp - Ceiling division (divide and round up)
 *
 * Computes ceil(x / y) without using floating point arithmetic.
 * Useful for calculating the number of chunks/blocks needed to cover x items
 * when each chunk holds y items.
 *
 * @param x Dividend (numerator)
 * @param y Divisor (denominator), must be > 0
 * @return ceil(x / y)
 *
 * Example: divUp(10, 3) = 4, divUp(9, 3) = 3
 */
template <typename X, typename Y, typename Z = decltype(X() + Y())>
static __host__ __device__ constexpr Z divUp(X x, Y y) {
  return (x + y - 1) / y;
}

/**
 * roundUp - Round up to the nearest multiple
 *
 * Rounds x up to the nearest multiple of y.
 * Works for any positive y (not required to be a power of 2).
 *
 * @param x Value to round up
 * @param y Multiple to round to, must be > 0
 * @return Smallest multiple of y that is >= x
 *
 * Example: roundUp(10, 4) = 12, roundUp(12, 4) = 12
 */
template <typename X, typename Y, typename Z = decltype(X() + Y())>
static __host__ __device__ constexpr Z roundUp(X x, Y y) {
  return (x + y - 1) - (x + y - 1) % y;
}

/**
 * roundDown - Round down to the nearest multiple
 *
 * Rounds x down to the nearest multiple of y.
 * Works for any positive y (not required to be a power of 2).
 *
 * @param x Value to round down
 * @param y Multiple to round to, must be > 0
 * @return Largest multiple of y that is <= x
 *
 * Example: roundDown(10, 4) = 8, roundDown(12, 4) = 12
 */
template <typename X, typename Y, typename Z = decltype(X() + Y())>
static __host__ __device__ constexpr Z roundDown(X x, Y y) {
  return x - x % y;
}

/**
 * alignUp - Align integer up to power-of-2 boundary
 *
 * Rounds x up to the nearest multiple of a using bitwise operations.
 * More efficient than roundUp() but requires a to be a power of 2.
 *
 * @param x Value to align
 * @param a Alignment boundary, MUST be a power of 2
 * @return Smallest multiple of a that is >= x
 *
 * Example: alignUp(10, 16) = 16, alignUp(16, 16) = 16, alignUp(17, 16) = 32
 */
template <typename X, typename Y, typename Z = decltype(X() + Y())>
static __host__ __device__ constexpr Z alignUp(X x, Y a) {
  return (x + a - 1) & -Z(a);
}

/**
 * alignUp - Align pointer up to power-of-2 boundary
 *
 * Rounds pointer address up to the nearest multiple of a.
 * Only works with single-byte types (char*, unsigned char*, etc.).
 *
 * @param x Pointer to align
 * @param a Alignment boundary in bytes, MUST be a power of 2
 * @return Aligned pointer >= x
 *
 * Example: alignUp((char*)0x1003, 16) = (char*)0x1010
 */
template <typename T>
static __host__ __device__ T* alignUp(T* x, size_t a) {
  static_assert(sizeof(T) == 1, "Only single byte types allowed.");
  return reinterpret_cast<T*>(
      (reinterpret_cast<uintptr_t>(x) + a - 1) & -uintptr_t(a));
}

/**
 * alignDown - Align integer down to power-of-2 boundary
 *
 * Rounds x down to the nearest multiple of a using bitwise operations.
 * More efficient than roundDown() but requires a to be a power of 2.
 *
 * @param x Value to align
 * @param a Alignment boundary, MUST be a power of 2
 * @return Largest multiple of a that is <= x
 *
 * Example: alignDown(17, 16) = 16, alignDown(16, 16) = 16, alignDown(15, 16) =
 * 0
 */
template <typename X, typename Y, typename Z = decltype(X() + Y())>
static __host__ __device__ constexpr Z alignDown(X x, Y a) {
  return x & -Z(a);
}

/**
 * alignDown - Align pointer down to power-of-2 boundary
 *
 * Rounds pointer address down to the nearest multiple of a.
 * Only works with single-byte types (char*, unsigned char*, etc.).
 *
 * @param x Pointer to align
 * @param a Alignment boundary in bytes, MUST be a power of 2
 * @return Aligned pointer <= x
 *
 * Example: alignDown((char*)0x1017, 16) = (char*)0x1010
 */
template <typename T>
static __host__ __device__ T* alignDown(T* x, size_t a) {
  static_assert(sizeof(T) == 1, "Only single byte types allowed.");
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(x) & -uintptr_t(a));
}

/**
 * isPow2 - Check if a value is a power of 2
 *
 * Uses the bit manipulation trick: a power of 2 has exactly one bit set,
 * so x & (x-1) clears that bit, resulting in 0.
 *
 * @param x Value to check
 * @return true if x is a power of 2, false otherwise
 *
 * Note: Returns true for x=0 (edge case), which may not be desired.
 * Example: isPow2(16) = true, isPow2(15) = false, isPow2(1) = true
 */
template <typename Int>
constexpr __host__ __device__ bool isPow2(Int x) {
  return (x & (x - 1)) == 0;
}

} // namespace comms::bitops
