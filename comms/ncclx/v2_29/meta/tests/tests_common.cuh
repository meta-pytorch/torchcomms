// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef TESTS_COMMON_CUH_
#define TESTS_COMMON_CUH_

/**
 * Defines common utilities for tests that depends on CUDA or NCCL
 */

#include <unistd.h>
#include <tuple>
#include <vector>
#include "cuda.h"
#include "mpi.h"
#include "nccl.h"

// Typed helper functions
template <typename T>
__device__ __forceinline__ T floatToType(float val) {
  return (T)val;
}

template <typename T>
__device__ __forceinline__ float toFloat(T val) {
  return (T)val;
}

template <>
__device__ __forceinline__ half floatToType<half>(float val) {
  return __float2half(val);
}

template <>
__device__ __forceinline__ float toFloat<half>(half val) {
  return __half2float(val);
}

#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
__device__ __forceinline__ __nv_bfloat16 floatToType<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <>
__device__ __forceinline__ float toFloat<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
__host__ __device__ __forceinline__ bool operator==(
    const __nv_fp8_e4m3& lh,
    const __nv_fp8_e4m3& rh) {
  return float(lh) == float(rh);
}
__host__ __device__ __forceinline__ bool operator==(
    const __nv_fp8_e5m2& lh,
    const __nv_fp8_e5m2& rh) {
  return float(lh) == float(rh);
}
#endif

#define DECL_TYPED_KERNS(T)                        \
  template __device__ T floatToType<T>(float val); \
  template __device__ float toFloat<T>(T val);

DECL_TYPED_KERNS(int8_t);
DECL_TYPED_KERNS(uint8_t);
DECL_TYPED_KERNS(int32_t);
DECL_TYPED_KERNS(uint32_t);
DECL_TYPED_KERNS(int64_t);
DECL_TYPED_KERNS(uint64_t);
DECL_TYPED_KERNS(float);
DECL_TYPED_KERNS(double);
// Skip half and __nv_bfloat16 since already declared with specific type

constexpr std::string_view kNcclUtCommDesc{"nccl_ut"};

#define MPICHECK_TEST(cmd)                                             \
  do {                                                                 \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define CUDACHECK_TEST(cmd)                  \
  do {                                       \
    cudaError_t e = cmd;                     \
    if (e != cudaSuccess) {                  \
      printf(                                \
          "Failed: Cuda error %s:%d '%s'\n", \
          __FILE__,                          \
          __LINE__,                          \
          cudaGetErrorString(e));            \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)

#define NCCLCHECK_TEST(cmd)                  \
  do {                                       \
    ncclResult_t r = cmd;                    \
    if (r != ncclSuccess) {                  \
      printf(                                \
          "Failed, NCCL error %s:%d '%s'\n", \
          __FILE__,                          \
          __LINE__,                          \
          ncclGetErrorString(r));            \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)

#define NCCLCHECKTHROW_TEST(cmd)                                        \
  do {                                                                  \
    ncclResult_t r = cmd;                                               \
    if (r != ncclSuccess) {                                             \
      throw std::runtime_error(                                         \
          std::string("Failed, NCCL error: ") + ncclGetErrorString(r)); \
    }                                                                   \
  } while (0)

// Assign a new value for environment variable during one test, and change the
// value back at the end of the test.
template <typename T>
class EnvRAII {
 public:
  EnvRAII(T& envVar, T newValue) : oldValue_(envVar), envVar_(envVar) {
    envVar_ = newValue;
  }

  ~EnvRAII() {
    envVar_ = oldValue_;
  }

 private:
  T oldValue_;
  T& envVar_;
};

#endif
