// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "comms/common/IpcGpuBarrier.cuh"

namespace meta::comms {

template <typename T>
concept SupportedTypes =
    (std::same_as<T, half> || std::same_as<T, __nv_bfloat16>);

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
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(&a);
    const __nv_bfloat16* y = reinterpret_cast<const __nv_bfloat16*>(&b);
    __nv_bfloat162 p = {x[0], x[1]};
    __nv_bfloat162 q = {y[0], y[1]};
    __nv_bfloat162 z = __hadd2(p, q);
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
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;

  // It is expensive to launch hipMemcpyAsync on ROCm
  // Move data copy here. Each block copies part of sendbuff data
  T* ipcbuff = ipcbuffs[selfRank];
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    *reinterpret_cast<uint4*>(&ipcbuff[idx]) =
        reinterpret_cast<const uint4*>(&sendbuff[idx])[0];
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    uint4 sum{0, 0, 0, 0};
    if constexpr (hasAcc) {
      sum = reinterpret_cast<const uint4*>(&acc[idx])[0];
    }
    // Pipelining read val from other ranks and accumulation
    uint4 srcVals[2];
    // Prologue: read data from first rank
    *reinterpret_cast<uint4*>(&srcVals[0]) =
        reinterpret_cast<const uint4*>(&ipcbuffs[0][idx])[0];

#pragma unroll NRANKS - 1
    for (int r = 0; r < NRANKS - 1; ++r) {
      // NOTE: the reduction order is same on each rank to achieve deterministic
      // result on every rank. We may change the order order to optimize memory
      // access if reduction order doesn't matter. Kick-off read data from next
      // rank
      *reinterpret_cast<uint4*>(&srcVals[(r + 1) & 1]) =
          reinterpret_cast<const uint4*>(&ipcbuffs[(r + 1) % NRANKS][idx])[0];
      // Do accumulation for current rank
      sum = vecElementAdd<T>(sum, srcVals[r & 1]);
    }
    // Epilogue: accumulation for last rank
    sum = vecElementAdd<T>(sum, srcVals[(NRANKS - 1) & 1]);

    // Store to the destination buffer.
    *reinterpret_cast<uint4*>(&recvbuff[idx]) =
        *reinterpret_cast<const uint4*>(&sum);
  }

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

  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  // use uint4 to do 16-byte loads to maximize memory efficiency
  // We assume that count % cntPerThread == 0. This assumption is enforced
  // before kernel launch
  // TODO: we should be able to deal with left over as well
  constexpr auto cntPerThread = sizeof(uint4) / sizeof(T);
  const size_t cntPerRank = count / NRANKS;
  const size_t rankStart = selfRank * cntPerRank;
  const size_t idxStride = gridDim.x * blockDim.x * cntPerThread;

  // stage 1: reduce-scatter
  for (size_t idx = rankStart + gtIdx * cntPerThread;
       idx < rankStart + cntPerRank;
       idx += idxStride) {
    uint4 sum{0, 0, 0, 0};
    // TODO: The bias accumulation needs to be moved to stage 2 if the bias
    // vector can be different on each rank. Currently we assume the bias vector
    // is the same across ranks.
    if constexpr (hasAcc) {
      sum = reinterpret_cast<const uint4*>(&acc[idx])[0];
    }
    // Pipelining read val from other ranks and accumulation
    uint4 srcVals[2];
    // Prologue: read data from first rank
    *reinterpret_cast<uint4*>(&srcVals[0]) =
        reinterpret_cast<const uint4*>(&ipcbuffs[0][idx])[0];
#pragma unroll NRANKS - 1
    for (int r = 0; r < NRANKS - 1; ++r) {
      // Kick-off reading data from next rank
      *reinterpret_cast<uint4*>(&srcVals[(r + 1) & 1]) =
          reinterpret_cast<const uint4*>(&ipcbuffs[(r + 1) % NRANKS][idx])[0];
      // Do accumulation for current rank
      sum = vecElementAdd<T>(sum, srcVals[r & 1]);
    }
    // Epilogue: accumulation for last rank
    sum = vecElementAdd<T>(sum, srcVals[(NRANKS - 1) & 1]);

    // Store to the local buffer
    *reinterpret_cast<uint4*>(&ipcbuffs[selfRank][idx]) =
        *reinterpret_cast<const uint4*>(&sum);
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  // stage 2: all-gather
  for (size_t idx = gtIdx * cntPerThread; idx < cntPerRank; idx += idxStride) {
#pragma unroll NRANKS
    for (int r = 0; r < NRANKS; ++r) {
      int srcRank = (selfRank + r) % NRANKS;
      int srcIdx = idx + srcRank * cntPerRank;
      *reinterpret_cast<uint4*>(&recvbuff[srcIdx]) =
          reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    }
  }

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
