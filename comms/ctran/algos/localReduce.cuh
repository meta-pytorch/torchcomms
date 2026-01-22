// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <assert.h>
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif
#endif

#include "comms/common/DeviceConstants.cuh"
#include "comms/ctran/utils/DevUtils.cuh"

/* FIXME: We are not currently using vectorized arithmetic.  We only
 * use vector loads.  For reduction operations, most of the time
 * should be spent in the loads and stores, so this should be OK for
 * now. */

#if defined(__HIP_PLATFORM_AMD__) &&       \
    (defined(__HIP_NO_HALF_OPERATORS__) || \
     defined(__HIP_NO_HALF_CONVERSIONS__))
__device__ __forceinline__ __half operator+(const __half& a, const __half& b) {
  return __hadd(a, b);
}

__device__ __forceinline__ __half operator*(const __half& a, const __half& b) {
  return __hmul(a, b);
}

__device__ __forceinline__ __half operator/(const __half& a, const int& b) {
  return __float2half(__half2float(a) / b);
}

__device__ __forceinline__ bool operator>(const __half& a, const __half& b) {
  return __half2float(a) > __half2float(b);
}

#else // NVIDIA
__device__ __forceinline__ __half operator/(const __half& a, const int& b) {
  return __half(__half2float(a) / b);
}
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)
__device__ __forceinline__ __nv_bfloat16
operator/(const __nv_bfloat16& a, const int& b) {
  return __nv_bfloat16(__bfloat162float(a) / b);
}
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
__device__ __forceinline__ __nv_fp8_e4m3
operator+(const __nv_fp8_e4m3& a, const __nv_fp8_e4m3& b) {
  return __nv_fp8_e4m3(__hadd(__half(a), __half(b)));
}
__device__ __forceinline__ __nv_fp8_e5m2
operator+(const __nv_fp8_e5m2& a, const __nv_fp8_e5m2& b) {
  return __nv_fp8_e5m2(__hadd(__half(a), __half(b)));
}

__device__ __forceinline__ __nv_fp8_e4m3
operator*(const __nv_fp8_e4m3& a, const __nv_fp8_e4m3& b) {
  return __nv_fp8_e4m3(__half(a) * __half(b));
}
__device__ __forceinline__ __nv_fp8_e5m2
operator*(const __nv_fp8_e5m2& a, const __nv_fp8_e5m2& b) {
  return __nv_fp8_e5m2(__half(a) * __half(b));
}

__device__ __forceinline__ __nv_fp8_e4m3
operator/(const __nv_fp8_e4m3& a, const int& b) {
  return __nv_fp8_e4m3(__half2float(__half(a)) / b);
}
__device__ __forceinline__ __nv_fp8_e5m2
operator/(const __nv_fp8_e5m2& a, const int& b) {
  return __nv_fp8_e5m2(__half2float(__half(a)) / b);
}

__device__ __forceinline__ bool operator>(
    const __nv_fp8_e4m3& a,
    const __nv_fp8_e4m3& b) {
  return __half(a) > __half(b);
}
__device__ __forceinline__ bool operator>(
    const __nv_fp8_e5m2& a,
    const __nv_fp8_e5m2& b) {
  return __half(a) > __half(b);
}
#endif

template <typename T, commRedOp_t RedOp>
__device__ __forceinline__ T seqReduce(const T* vals, size_t nvectors) {
  T dst = vals[0];
  for (int i = 1; i < nvectors; i++) {
    if constexpr (RedOp == commSum || RedOp == commAvg) {
      dst = dst + vals[i];
    } else if constexpr (RedOp == commProd) {
      dst = dst * vals[i];
    } else if constexpr (RedOp == commMax) {
      dst = dst > vals[i] ? dst : vals[i];
    } else if constexpr (RedOp == commMin) {
      dst = dst > vals[i] ? vals[i] : dst;
    }
  }
  return dst;
}

template <typename T, commRedOp_t RedOp>
__device__ __forceinline__ T reduceNcclOp1(const T& a, const T& b) {
  if constexpr (RedOp == commSum || RedOp == commAvg) {
    return a + b;
  } else if constexpr (RedOp == commProd) {
    return a * b;
  } else if constexpr (RedOp == commMax) {
    return (a > b) ? a : b;
  } else if constexpr (RedOp == commMin) {
    return (a > b) ? b : a;
  }
}

template <typename T, int N = 16>
struct __align__(16) T_NBytes {
  static constexpr int kWords = N / sizeof(T);
  static_assert(ctran::utils::isEvenDivisor(N, sizeof(T)));
  T v[kWords];

  template <typename U>
  __device__ __forceinline__ T_NBytes<T, N> operator/(U divisor) const {
    auto out = *this;

    for (int i = 0; i < kWords; ++i) {
      out.v[i] = out.v[i] / divisor;
    }

    return out;
  }

  template <commRedOp_t RedOp>
  __device__ __forceinline__ T reduceHorizontal() const {
    auto out = v[0];
#pragma unroll
    for (int i = 1; i < kWords; ++i) {
      out = reduceNcclOp1<T, RedOp>(out, v[i]);
    }

    return out;
  }

  template <commRedOp_t RedOp>
  __device__ __forceinline__ T_NBytes<T, N> reduceVertical(
      const T_NBytes<T, N>& in) const {
    auto out = *this;
#pragma unroll
    for (int i = 0; i < kWords; ++i) {
      out.v[i] = reduceNcclOp1<T, RedOp>(out.v[i], in.v[i]);
    }

    return out;
  }
};

// Specialization for a fixed nvector count
template <typename T, commRedOp_t RedOp, int NSrcs, int NDsts>
__device__ __forceinline__ void localReduceVectorized(
    const T** srcs,
    T** dsts,
    size_t count,
    int workerId,
    int numWorkers,
    size_t nRanks = 1) {
  using TVec = T_NBytes<T, 16>;
  constexpr uint32_t kWordsPerVectorLoad = TVec::kWords;
  constexpr uint32_t kUnroll = 4;

  // Each warp will handle kUnroll values (warp contiguous and sequential)
  // at a time, resulting in each warp processing this many T-word per iteration
  constexpr uint32_t kWordsPerWarp =
      comms::device::kWarpSize * kWordsPerVectorLoad * kUnroll;

  const uint32_t linearThreadId = blockDim.x * workerId + threadIdx.x;
  const uint32_t warpId = linearThreadId / comms::device::kWarpSize;
  const uint32_t laneId = threadIdx.x % comms::device::kWarpSize;
  const uint32_t numWarps = numWorkers * blockDim.x / comms::device::kWarpSize;
  const uint32_t reduceStride = numWorkers * blockDim.x;

  const uint32_t u32Count = uint32_t(count);
  const uint32_t limitCount = ctran::utils::roundDown(u32Count, kWordsPerWarp);

  TVec s[kUnroll][NSrcs];

  uint32_t i = warpId * kWordsPerWarp + laneId * kWordsPerVectorLoad;

  // avoid the additional increment
  T* localDsts[NDsts];
#pragma unroll
  for (int d = 0; d < NDsts; ++d) {
    localDsts[d] = dsts[d] + i;
  }

  // Handle the portion of `count` that can be handled with the fully unrolled
  // vectorized loop
  for (; i < limitCount; i += numWarps * kWordsPerWarp) {
    // Having the NVectors loop be outermost here is 2-5% faster than innermost
#pragma unroll
    for (int j = 0; j < NSrcs; ++j) {
#pragma unroll
      for (int k = 0; k < kUnroll; ++k) {
        s[k][j] = *reinterpret_cast<const TVec*>(
            &srcs[j][i + k * comms::device::kWarpSize * kWordsPerVectorLoad]);
      }
    }

    // Reduce vertically along the vectors
    // However, here, the original kUnroll then NVectors loop order is faster
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
#pragma unroll
      for (int k = 1; k < NSrcs; ++k) {
        s[j][0] = s[j][0].template reduceVertical<RedOp>(s[j][k]);
      }

      if constexpr (RedOp == commAvg) {
        s[j][0] = s[j][0] / int(nRanks);
      }

#pragma unroll
      for (int d = 0; d < NDsts; ++d) {
        *reinterpret_cast<TVec*>(
            &localDsts[d][j * comms::device::kWarpSize * kWordsPerVectorLoad]) =
            s[j][0];
      }
    }
#pragma unroll
    for (int d = 0; d < NDsts; ++d) {
      localDsts[d] += numWarps * kWordsPerWarp;
    }
  }

  // Loop epilogue to handle remainder with a simple grid-stride loop
  for (uint32_t i = limitCount + linearThreadId; i < count; i += reduceStride) {
    T s = srcs[0][i];
#pragma unroll
    for (int j = 1; j < NSrcs; ++j) {
      s = reduceNcclOp1<T, RedOp>(s, srcs[j][i]);
    }

    if constexpr (RedOp == commAvg) {
      s = s / int(nRanks);
    }

#pragma unroll
    for (int d = 0; d < NDsts; ++d) {
      dsts[d][i] = s;
    }
  }

  __syncthreads();
}

template <typename T, commRedOp_t RedOp, int NSrcs, int NDsts>
__device__ __forceinline__ void localReduceVectorized(
    const T** srcs,
    T** dsts,
    size_t count,
    size_t nRanks = 1) {
  localReduceVectorized<T, RedOp, NSrcs, NDsts>(
      srcs, dsts, count, blockIdx.x, gridDim.x, nRanks);
}

// Specialization for a fixed nvector count
template <typename T, commRedOp_t RedOp>
__device__ __forceinline__ void localReduceFallback(
    uint32_t nsrcs,
    const T** srcs,
    uint32_t ndsts,
    T** dsts,
    size_t count,
    int workerId,
    int numWorkers,
    size_t nRanks = 1) {
  constexpr int kVectors = CTRAN_MAX_NVL_PEERS;
  assert(nsrcs <= kVectors);

  // FIXME: adjust unroll factor for this un-vectorized version
  constexpr uint32_t kUnroll = 8;

  // Each warp will handle kUnroll values (warp contiguous and sequential)
  // at a time
  constexpr uint32_t kWordsPerWarp = comms::device::kWarpSize * kUnroll;

  const uint32_t linearThreadId = blockDim.x * workerId + threadIdx.x;
  const uint32_t globalWarpId = linearThreadId / comms::device::kWarpSize;
  const uint32_t laneId = threadIdx.x % comms::device::kWarpSize;
  const uint32_t numGlobalWarps =
      numWorkers * blockDim.x / comms::device::kWarpSize;

  const size_t limitCount = ctran::utils::roundDown(count, kWordsPerWarp);

  T s[kUnroll][kVectors];

  for (size_t i = globalWarpId * kWordsPerWarp + laneId; i < limitCount;
       i += numGlobalWarps * kWordsPerWarp) {
    for (uint32_t j = 0; j < nsrcs; ++j) {
#pragma unroll
      for (int k = 0; k < kUnroll; ++k) {
        s[k][j] = *(&srcs[j][i + k * comms::device::kWarpSize]);
      }
    }

    // reduce vertically along the vectors
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      for (uint32_t k = 1; k < nsrcs; ++k) {
        s[j][0] = reduceNcclOp1<T, RedOp>(s[j][0], s[j][k]);
      }

      if constexpr (RedOp == commAvg) {
        s[j][0] = s[j][0] / int(nRanks);
      }

      for (int d = 0; d < ndsts; ++d) {
        dsts[d][i + j * comms::device::kWarpSize] = s[j][0];
      }
    }
  }

  // Loop epilogue to handle remainder with a simple grid-stride loop
  for (uint32_t i = limitCount + linearThreadId; i < count;
       i += numWorkers * blockDim.x) {
    T s = srcs[0][i];
#pragma unroll
    for (int j = 1; j < nsrcs; ++j) {
      s = reduceNcclOp1<T, RedOp>(s, srcs[j][i]);
    }

    if constexpr (RedOp == commAvg) {
      s = s / int(nRanks);
    }

    for (int d = 0; d < ndsts; ++d) {
      dsts[d][i] = s;
    }
  }

  __syncthreads();
}

// FIXME: enable --expt-relaxed-constexpr
constexpr uint32_t kMax_uint32_t = std::numeric_limits<uint32_t>::max();

// NOTE for commAvg: upper layer should set RedOp=commAvg when performing the
// last reduce for a given data segment. For earlier steps, just pass commSum.
// If commAvg is passed, it will apply element-wise average with nRanks
// following the same partition as the reduce computation. It avoids expensive
// cross-thread-block sync.
template <typename T, commRedOp_t RedOp>
__device__ __forceinline__ void localReduce(
    size_t nsrcs,
    const T** srcs,
    size_t ndsts,
    T** dsts,
    size_t count,
    int workerId,
    int numWorkers,
    size_t nRanks = 1) {
  // In order to use the optimized implementation:
  // -the src and dst pointers must be aligned to 16 bytes
  // -count must be sufficiently large, but under max uint32_t
  // nvectors must be 2/4/8
  bool isAligned = true;
  for (int i = 0; i < nsrcs; i++) {
    isAligned = isAligned && ctran::utils::isAlignedPointer<16>(srcs[i]);
  }
  for (int i = 0; i < ndsts; i++) {
    isAligned = isAligned && ctran::utils::isAlignedPointer<16>(dsts[i]);
  }

  bool isCountInBounds = count <= size_t(kMax_uint32_t);

  if (isAligned && isCountInBounds) {
    if (nsrcs == 8 && ndsts == 1) {
      localReduceVectorized<T, RedOp, /*NSrcs=*/8, /*NDsts=*/1>(
          srcs, dsts, count, workerId, numWorkers, nRanks);
      return;
    } else if (nsrcs == 4 && ndsts == 1) {
      localReduceVectorized<T, RedOp, /*NSrcs=*/4, /*NDsts=*/1>(
          srcs, dsts, count, workerId, numWorkers, nRanks);
      return;
    } else if (nsrcs == 2 && ndsts == 1) {
      localReduceVectorized<T, RedOp, /*NSrcs=*/2, /*NDsts=*/1>(
          srcs, dsts, count, workerId, numWorkers, nRanks);
      return;
    } else if (nsrcs == 8 && ndsts == 2) {
      localReduceVectorized<T, RedOp, /*NSrcs=*/8, /*NDsts=*/2>(
          srcs, dsts, count, workerId, numWorkers, nRanks);
      return;
    } else if (nsrcs == 2 && ndsts == 2) {
      localReduceVectorized<T, RedOp, /*NSrcs=*/2, /*NDsts=*/2>(
          srcs, dsts, count, workerId, numWorkers, nRanks);
      return;
    } else if (nsrcs == 1 && ndsts == 8) {
      localReduceVectorized<T, RedOp, /*NSrcs=*/1, /*NDsts=*/8>(
          srcs, dsts, count, workerId, numWorkers, nRanks);
      return;
    } else if (nsrcs == 8 && ndsts == 8) {
      localReduceVectorized<T, RedOp, /*NSrcs=*/8, /*NDsts=*/8>(
          srcs, dsts, count, workerId, numWorkers, nRanks);
      return;
    } else if (nsrcs == 1 && ndsts == 1) {
      localReduceVectorized<T, RedOp, /*NSrcs=*/1, /*NDsts=*/1>(
          srcs, dsts, count, workerId, numWorkers, nRanks);
      return;
    }
  }

  // Fallback slow implementation
  localReduceFallback<T, RedOp>(
      nsrcs, srcs, ndsts, dsts, count, workerId, numWorkers, nRanks);
}

template <typename T, commRedOp_t RedOp>
__device__ __forceinline__ void localReduce(
    size_t nvectors,
    const T** srcs,
    T* dst,
    size_t count,
    size_t nRanks = 1) {
  localReduce<T, RedOp>(
      nvectors, srcs, 1, &dst, count, blockIdx.x, gridDim.x, nRanks);
}

template <typename T, commRedOp_t RedOp>
__device__ __forceinline__ void localReduce(
    size_t nvectors,
    const T** srcs,
    T* dst,
    size_t count,
    int workerId,
    int numWorkers,
    size_t nRanks = 1) {
  localReduce<T, RedOp>(
      nvectors, srcs, 1, &dst, count, workerId, numWorkers, nRanks);
}

template <typename T, commRedOp_t RedOp>
__device__ __forceinline__ void localReduce(
    size_t nsrcs,
    const T** srcs,
    size_t ndsts,
    T** dsts,
    size_t count,
    size_t nRanks = 1) {
  localReduce<T, RedOp>(
      nsrcs, srcs, ndsts, dsts, count, blockIdx.x, gridDim.x, nRanks);
}

// Specialization for a alltoall based local reduce
// src:
// rank 0: A0, B0, C0, D0
// rank 1: A1, B1, C1, D1
// rank 2: A2, B2, C2, D3
// rank 3: A3, B3, C3, D3
// dst:
// rank 0: sum(chunk 0)
// rank 1: sum(chunk 1)
// rank 2: sum(chunk 2)
// rank 3: sum(chunk 3)
template <typename T, typename RedT, commRedOp_t RedOp>
__device__ __forceinline__ void localReduceForDequantAllToAll(
    const T* src,
    T* dst,
    size_t count,
    int myRank,
    size_t nRanks = 1) {
  constexpr auto kVectors = 8;
  // FIXME: adjust unroll factor for this un-vectorized version
  constexpr auto kUnroll = 8;

  // Each warp will handle kUnroll values (warp contiguous and sequential)
  // at a time
  constexpr auto kWordsPerWarp = comms::device::kWarpSize * kUnroll;

  const auto linearThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  const auto globalWarpId = linearThreadId / comms::device::kWarpSize;
  const auto laneId = threadIdx.x % comms::device::kWarpSize;
  const auto numGlobalWarps = gridDim.x * blockDim.x / comms::device::kWarpSize;

  const auto limitCount = ctran::utils::roundDown(count, kWordsPerWarp);

  RedT res[kUnroll][kVectors];
  // Need multiple rounds of local reduce if total num of ranks is more than
  // kVectors - 1.
  int vecIdx = 0;
  int step = 0;

  for (auto i = globalWarpId * kWordsPerWarp + laneId; i < limitCount;
       i += numGlobalWarps * kWordsPerWarp) {
    vecIdx = 0;
    step = 0;
    while (step < nRanks) {
      // step 0:
      // rank 0: src 0, rank 1: src 1, rank 2: src 2, rank 3: src 3
      // step 1:
      // rank 0: src 1, rank 1: src 2, rank 2: src 3, rank 3: src 0
      // step 2:
      // rank 0: src 2, rank 1: src 3, rank 2: src 0, rank 3: src 1
      // step 3:
      // rank 0: src 3, rank 1: src 0, rank 2: src 1, rank 3: src 2
      auto peer = (myRank + step) % nRanks;
      auto offset = count * peer;

#pragma unroll
      for (auto k = 0; k < kUnroll; ++k) {
        RedT curr = ctran::utils::castTo<T, RedT>(
            src[i + k * comms::device::kWarpSize + offset]);
        res[k][vecIdx] = curr;
      }

      vecIdx++;
      step++;

      if (vecIdx == kVectors || step == nRanks) {
#pragma unroll
        for (auto k = 0; k < kUnroll; ++k) {
          for (auto j = 1; j < vecIdx; j++) {
            // store the value into res[k][0]
            res[k][0] = reduceNcclOp1<RedT, RedOp>(res[k][0], res[k][j]);
          }
        }

        // Reset for next round; reserve res[k][0] for intermediate result
        // from the finished round
        vecIdx = 1;
      }
    }

    // cast to the orignal data type the final result
#pragma unroll
    for (auto j = 0; j < kUnroll; ++j) {
      if constexpr (RedOp == commAvg) {
        res[j][0] = res[j][0] / int(nRanks);
      }
      dst[i + j * comms::device::kWarpSize] =
          ctran::utils::castTo<RedT, T>(res[j][0]);
    }
  }

  // Loop epilogue to handle remainder with a simple grid-stride loop
  for (auto i = limitCount + linearThreadId; i < count;
       i += gridDim.x * blockDim.x) {
    vecIdx = 0;
    step = 0;
    while (step < nRanks) {
      int peer = (myRank + step) % nRanks;
      size_t offset = count * peer;
      res[0][vecIdx] = ctran::utils::castTo<T, RedT>(src[i + offset]);
      vecIdx++;
      step++;

      if (vecIdx == kVectors || step == nRanks) {
        for (auto j = 1; j < vecIdx; j++) {
          res[0][0] = reduceNcclOp1<RedT, RedOp>(res[0][0], res[0][j]);
        }

        // Reset for next round; reserve res[0][0] for intermediate result
        // from the finished round
        vecIdx = 1;
      }
    }

    if constexpr (RedOp == commAvg) {
      res[0][0] = res[0][0] / int(nRanks);
    }
    dst[i] = ctran::utils::castTo<RedT, T>(res[0][0]);
  }

  __syncthreads();
}
