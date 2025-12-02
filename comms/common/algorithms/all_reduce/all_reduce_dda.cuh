// Copyright (c) Meta Platforms, Inc. and affiliates.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/amd_detail/amd_hip_bf16.h>
// Use a backend-agnostic BF16 alias so hipify does not rewrite it into the
// host-side hip_bfloat16 struct on ROCm.
using bf16 = __hip_bfloat16;
using bf162 = __hip_bfloat162;
#else
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;
#endif
#include "comms/common/IpcGpuBarrier.cuh"
#include "comms/common/algorithms/CollCommon.cuh"

namespace meta::comms {

template <typename T>
concept SupportedTypes =
    (std::same_as<T, half> || std::same_as<T, bf16>);

template <SupportedTypes T>
static inline __device__ uint32_t
vecElementAdd(const uint32_t& a, const uint32_t& b) {
  if constexpr (std::is_same<T, half>::value) {
    const __half* x = reinterpret_cast<const __half*>(&a);
    const __half* y = reinterpret_cast<const __half*>(&b);
    __half2 p = __halves2half2(x[0], x[1]);
    __half2 q = __halves2half2(y[0], y[1]);
    __half2 z = __hadd2(p, q);
    return (reinterpret_cast<uint32_t*>(&z))[0];
  } else if constexpr (std::is_same<T, bf16>::value) {
    const bf16* x = reinterpret_cast<const bf16*>(&a);
    const bf16* y = reinterpret_cast<const bf16*>(&b);
    bf162 p = {x[0], x[1]};
    bf162 q = {y[0], y[1]};
    bf162 z = __hadd2(p, q);
    return (reinterpret_cast<uint32_t*>(&z))[0];
  }
  return 0;
}

template <SupportedTypes T>
static inline __device__ uint4 vecElementAdd(const uint4& a, const uint4& b) {
  uint4 res{0, 0, 0, 0};
  res.x = vecElementAdd<T>(a.x, b.x);
  res.y = vecElementAdd<T>(a.y, b.y);
  res.z = vecElementAdd<T>(a.z, b.z);
  res.w = vecElementAdd<T>(a.w, b.w);
  return res;
}

template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
    __global__ void ddaAllReduceFlatIpc(
        T* const* __restrict__ ipcbuffs,
        T* __restrict__ recvbuff,
        size_t count,
        const T* __restrict__ sendbuff,
        int selfRank,
        IpcGpuBarrier barrier,
        const T* __restrict__ acc) {
  // use uint4 to do 16-byte loads to maximize memory efficiency
  // We assume that count % countPerThread == 0. This assumption is enforced
  // before kernel launch
  // TODO: we should be able to deal with left over as well
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // It is expensive to launch hipMemcpyAsync on ROCm
  // Move data copy here. Each block copies part of sendbuff data
  copyFromSrcToDest<T>(
      sendbuff, ipcbuffs[selfRank], idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  reduceScatter<T, NRANKS, hasAcc>(
      ipcbuffs, recvbuff, acc, selfRank, idxStart, idxEnd, idxStride, 2);

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
    __global__ void ddaAllReduceTreeIpc(
        T* const* __restrict__ ipcbuffs,
        T* __restrict__ recvbuff,
        size_t count,
        const T* __restrict__ sendbuff,
        int selfRank,
        IpcGpuBarrier barrier,
        const T* __restrict__ acc) {
  barrier.syncOnSameBlockIdx<
      false /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  // use uint4 to do 16-byte loads to maximize memory efficiency
  // We assume that count % countPerThread == 0. This assumption is enforced
  // before kernel launch
  // TODO: we should be able to deal with left over as well
  const size_t countPerRank = count / NRANKS;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  // stage 1: reduce-scatter
  reduceScatter<T, NRANKS, hasAcc>(
      ipcbuffs,
      ipcbuffs[selfRank],
      acc,
      selfRank,
      idxStart,
      idxEnd,
      idxStride,
      1);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  // stage 2: all-gather
  allGather<T, NRANKS>(
      ipcbuffs, recvbuff, selfRank, idxStart, idxEnd, idxStride, true);

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
