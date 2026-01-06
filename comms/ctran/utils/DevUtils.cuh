// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include "comms/common/DevUtils.cuh"

namespace ctran::utils {

// Convenience functions for compile-time (and run-time) parameter sizing
template <typename T>
constexpr __host__ __device__ __forceinline__ bool isPowerOf2(T v) {
  return (v & (v - 1)) == 0;
}

template <int N>
constexpr __host__ __device__ __forceinline__ bool isAlignedPointer(
    const void* const p) {
  static_assert(isPowerOf2(N));
  return (reinterpret_cast<uintptr_t>(p) & (N - 1)) == 0;
}

// assumes second argument is a power of 2
template <typename X, typename Z = decltype(X() + int())>
constexpr __host__ __device__ __forceinline__ Z alignUp(X x, int a) {
  return (x + a - 1) & Z(-a);
}

template <typename U, typename V>
constexpr __host__ __device__ __forceinline__ auto divDown(U a, V b)
    -> decltype(a / b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value);
  return (a / b);
}

template <typename U, typename V>
constexpr __host__ __device__ __forceinline__ auto divUp(U a, V b)
    -> decltype(a / b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value);
  return (a + b - 1) / b;
}

template <typename U, typename V>
constexpr __host__ __device__ __forceinline__ auto roundDown(U a, V b)
    -> decltype(a / b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value);
  return divDown(a, b) * b;
}

template <typename U, typename V>
constexpr __host__ __device__ __forceinline__ auto roundUp(U a, V b)
    -> decltype(a / b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value);
  return divUp(a, b) * b;
}

template <typename U, typename V>
constexpr __host__ __device__ __forceinline__ bool isEvenDivisor(U a, V b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value);
  return (a % V(b) == 0) && ((a / V(b)) >= 1);
}

template <typename Int>
inline __host__ __device__ int log2Down(Int x) {
  int w, n;
#if __CUDA_ARCH__
  if (sizeof(Int) <= sizeof(int)) {
    w = 8 * sizeof(int);
    n = __clz((int)x);
  } else if (sizeof(Int) <= sizeof(long long)) {
    w = 8 * sizeof(long long);
    n = __clzll((long long)x);
  } else {
    static_assert(
        sizeof(Int) <= sizeof(long long), "Unsupported integer size.");
  }
#else
  if (x == 0) {
    return -1;
  } else if (sizeof(Int) <= sizeof(unsigned int)) {
    w = 8 * sizeof(unsigned int);
    n = __builtin_clz((unsigned int)x);
  } else if (sizeof(Int) <= sizeof(unsigned long)) {
    w = 8 * sizeof(unsigned long);
    n = __builtin_clzl((unsigned long)x);
  } else if (sizeof(Int) <= sizeof(unsigned long long)) {
    w = 8 * sizeof(unsigned long long);
    n = __builtin_clzll((unsigned long long)x);
  } else {
    static_assert(
        sizeof(Int) <= sizeof(unsigned long long), "Unsupported integer size.");
  }
#endif
  return (w - 1) - n;
}

inline __host__ __device__ long log2i(long n) {
  return log2Down(n);
}

template <typename INPUT, typename OUTPUT>
__device__ __forceinline__ OUTPUT castTo(INPUT input) {
  return OUTPUT(input);
}

#ifdef __CUDA_BF16_TYPES_EXIST__
template <>
__device__ __forceinline__ __nv_bfloat16 castTo<float, __nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}
template <>
__device__ __forceinline__ float castTo<__nv_bfloat16, float>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
#endif

template <typename T>
__device__ __forceinline__ T atomicRead(T* a) {
  return atomicAdd(a, 0);
}

} // namespace ctran::utils
