// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <type_traits>

////////////////////////////////////////////////////////////////////////////////
// FuncPatSumPostDiv - Native AVG support for PAT algorithm
// Unlike FuncSumPostDiv which only supports unsigned integers,
// FuncPatSumPostDiv supports float, double, bfloat16, and integer types
// (both signed and unsigned). Reduction is pure sum, division is applied as
// postOp on final write. The opArg encodes (divisor << 1) | isSigned.

template <typename T>
struct RedOpArg<FuncPatSumPostDiv<T>> {
  static constexpr bool ArgUsed = true;
  __device__ __forceinline__ static uint64_t loadArg(void* ptr) {
    return *(uint64_t*)ptr;
  }
};

// General FuncPatSumPostDiv definition for all types.
// opArg encoding: (divisor << 1) | isSigned, matching FuncSumPostDiv
// convention.
template <typename T>
struct FuncPatSumPostDiv {
  using EltType = T;
  uint32_t divisor;
  uint32_t isSigned;

  __device__ __forceinline__ FuncPatSumPostDiv(uint64_t opArg = 0) {
    isSigned = opArg & 1;
    divisor = opArg >> 1;
  }

  // Division helper - uses signed division when the original dtype is signed
  __device__ __forceinline__ T divide(T x) const {
    if constexpr (std::is_integral_v<T>) {
      if (isSigned) {
        // Use at least 32-bit signed type to avoid overflow for small T (e.g.,
        // uint8_t with nRanks > 127)
        using WideSignedT = typename std::conditional<
            sizeof(T) < 4,
            int32_t,
            typename std::make_signed<T>::type>::type;
        return T(WideSignedT(x) / WideSignedT(divisor));
      }
    }
    return x / T(divisor);
  }
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
// Specialization for bfloat16.
// opArg encoding: (divisor << 1) | isSigned, same as generic template.
// divide() uses float-precision arithmetic, so sign handling is not needed.
template <>
struct FuncPatSumPostDiv<__nv_bfloat16> {
  using EltType = __nv_bfloat16;
  uint32_t divisor;
  uint32_t isSigned;

  __device__ __forceinline__ FuncPatSumPostDiv(uint64_t opArg = 0) {
    isSigned = opArg & 1;
    divisor = opArg >> 1;
  }

  __device__ __forceinline__ __nv_bfloat16 divide(__nv_bfloat16 x) const {
    return __float2bfloat16(__bfloat162float(x) / static_cast<float>(divisor));
  }
};
#endif

// Apply_Reduce for FuncPatSumPostDiv - dispatches to FuncSum (reduction is pure
// sum)
template <typename T, int EltPerPack>
struct Apply_Reduce<FuncPatSumPostDiv<T>, EltPerPack>
    : Apply_Reduce<FuncSum<T>, EltPerPack> {
  __device__ __forceinline__ static BytePack<EltPerPack * sizeof(T)> reduce(
      FuncPatSumPostDiv<T> fn,
      BytePack<EltPerPack * sizeof(T)> a,
      BytePack<EltPerPack * sizeof(T)> b) {
    // FuncPatSumPostDiv reduce dispatches to FuncSum - division only at postOp
    return Apply_Reduce<FuncSum<T>, EltPerPack>::reduce(FuncSum<T>(), a, b);
  }
};

// Apply_PostOp for FuncPatSumPostDiv - applies division
template <typename T>
struct Apply_PostOp<FuncPatSumPostDiv<T>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(T)> postOp(
      FuncPatSumPostDiv<T> fn,
      BytePack<sizeof(T)> a) {
    return toPack<T>(fn.divide(fromPack<T>(a)));
  }
};
