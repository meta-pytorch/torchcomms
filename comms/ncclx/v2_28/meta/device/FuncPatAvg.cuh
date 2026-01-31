// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

////////////////////////////////////////////////////////////////////////////////
// FuncPatAvg - Native AVG support for PAT algorithm
// Unlike FuncSumPostDiv which only supports unsigned integers, FuncPatAvg
// supports all data types including float, half, bfloat16, and fp8.
// Reduction is pure sum, division is applied as postOp on final write.

template <typename T>
struct RedOpArg<FuncPatAvg<T>> {
  static constexpr bool ArgUsed = true;
  __device__ __forceinline__ static uint64_t loadArg(void* ptr) {
    return *(uint64_t*)ptr;
  }
};

// General FuncPatAvg definition for all types
template <typename T>
struct FuncPatAvg {
  using EltType = T;
  int nRanks;

  __device__ __forceinline__ FuncPatAvg(uint64_t opArg = 0) {
    nRanks = static_cast<int>(opArg);
  }

  // Division helper - different implementations for different types
  __device__ __forceinline__ T divide(T x) const {
    return x / static_cast<T>(nRanks);
  }
};

// Specialization for half (fp16)
template <>
struct FuncPatAvg<half> {
  using EltType = half;
  int nRanks;

  __device__ __forceinline__ FuncPatAvg(uint64_t opArg = 0) {
    nRanks = static_cast<int>(opArg);
  }

  __device__ __forceinline__ half divide(half x) const {
#if __CUDA_ARCH__ >= 530
    return __hdiv(x, __int2half_rn(nRanks));
#else
    return __float2half(__half2float(x) / static_cast<float>(nRanks));
#endif
  }
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
// Specialization for bfloat16
template <>
struct FuncPatAvg<__nv_bfloat16> {
  using EltType = __nv_bfloat16;
  int nRanks;

  __device__ __forceinline__ FuncPatAvg(uint64_t opArg = 0) {
    nRanks = static_cast<int>(opArg);
  }

  __device__ __forceinline__ __nv_bfloat16 divide(__nv_bfloat16 x) const {
#if __CUDA_ARCH__ >= 800
    return __hdiv(x, __int2bfloat16_rn(nRanks));
#else
    return __float2bfloat16(__bfloat162float(x) / static_cast<float>(nRanks));
#endif
  }
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
// Specialization for FP8 E4M3
template <>
struct FuncPatAvg<__nv_fp8_e4m3> {
  using EltType = __nv_fp8_e4m3;
  int nRanks;

  __device__ __forceinline__ FuncPatAvg(uint64_t opArg = 0) {
    nRanks = static_cast<int>(opArg);
  }

  __device__ __forceinline__ __nv_fp8_e4m3 divide(__nv_fp8_e4m3 x) const {
    return __nv_fp8_e4m3(static_cast<float>(x) / static_cast<float>(nRanks));
  }
};

// Specialization for FP8 E5M2
template <>
struct FuncPatAvg<__nv_fp8_e5m2> {
  using EltType = __nv_fp8_e5m2;
  int nRanks;

  __device__ __forceinline__ FuncPatAvg(uint64_t opArg = 0) {
    nRanks = static_cast<int>(opArg);
  }

  __device__ __forceinline__ __nv_fp8_e5m2 divide(__nv_fp8_e5m2 x) const {
    return __nv_fp8_e5m2(static_cast<float>(x) / static_cast<float>(nRanks));
  }
};
#endif

// Apply_Reduce for FuncPatAvg - dispatches to FuncSum (reduction is pure sum)
template <typename T, int EltPerPack>
struct Apply_Reduce<FuncPatAvg<T>, EltPerPack>
    : Apply_Reduce<FuncSum<T>, EltPerPack> {
  __device__ __forceinline__ static BytePack<EltPerPack * sizeof(T)> reduce(
      FuncPatAvg<T> fn,
      BytePack<EltPerPack * sizeof(T)> a,
      BytePack<EltPerPack * sizeof(T)> b) {
    // FuncPatAvg reduce dispatches to FuncSum - division only at postOp
    return Apply_Reduce<FuncSum<T>, EltPerPack>::reduce(FuncSum<T>(), a, b);
  }
};

// Apply_PostOp for FuncPatAvg - applies division
template <typename T>
struct Apply_PostOp<FuncPatAvg<T>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(T)> postOp(
      FuncPatAvg<T> fn,
      BytePack<sizeof(T)> a) {
    return toPack<T>(fn.divide(fromPack<T>(a)));
  }
};
