// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#else
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/utils/DevUtils.cuh"

static constexpr int UnrollSize = 8;

template <int Unroll16b, typename T>
__device__ __forceinline__ void bcastUnroll(
    const int nvectors,
    const T* src,
    T** dsts,
    const int count,
    const int groupIdx,
    const int nGroups) {
  static_assert(
      sizeof(T) <= sizeof(uint4),
      "Type T must be smaller than or equal to uint4");
  constexpr int kTper16Bytes = sizeof(uint4) / sizeof(T);
  constexpr int kUnroll = kTper16Bytes * Unroll16b;

  const auto numPerBlock = blockDim.x * kUnroll;
  const auto limitUnroll = ctran::utils::roundDown(count, numPerBlock);

  for (auto i = groupIdx * numPerBlock + threadIdx.x; i < limitUnroll;
       i += nGroups * numPerBlock) {
    T regv[kUnroll];

#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      regv[j] = src[i + j * blockDim.x];
    }
    for (int v = 0; v < nvectors; v++) { // add unroll here
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        dsts[v][i + j * blockDim.x] = regv[j];
      }
    }
  }
  for (auto offset = limitUnroll + blockDim.x * groupIdx + threadIdx.x;
       offset < count;
       offset += nGroups * blockDim.x) {
    for (int i = 0; i < nvectors; i++) {
      dsts[i][offset] = src[offset];
    }
  }

  __syncthreads();
}

template <typename T>
__device__ __forceinline__ void bcast(
    const int nvectors,
    const T* src,
    T** dsts,
    const int count,
    const int groupIdx,
    const int nGroups) {
  bool canCopy16Byte = canCopy16(src, count);
  for (int i = 0; i < nvectors; i++) {
    canCopy16Byte = canCopy16Byte && canCopy16(dsts[i]);
  }

  if (canCopy16Byte) {
    constexpr int kTperUint4 = sizeof(uint4) / sizeof(T);
    bcastUnroll<UnrollSize>(
        nvectors,
        reinterpret_cast<const uint4*>(src),
        reinterpret_cast<uint4**>(dsts),
        count / kTperUint4,
        groupIdx,
        nGroups);
  } else {
    bcastUnroll<UnrollSize>(nvectors, src, dsts, count, groupIdx, nGroups);
  }
}

// Unlike bcast, multiPutBcast copies the full data to a single peer and
// switch to the next peer after complete previous peer's data transfer.
// Each rank shifts the peer by 1 to avoid congestion. However, if each rank
// doesn't start at the same time, congestion can still happen. Algorithm layer
// needs to ensure all ranks start at the same time to avoid such risk.
template <typename T>
__device__ __forceinline__ void multiPutBcast(
    size_t nvectors,
    const T* src,
    T** dsts,
    size_t count,
    const int groupIdx,
    const int nGroups) {
  const int localRank = statex->localRank();
  const size_t nbytes = count * sizeof(T);

  for (int i = 0; i < nvectors; i++) {
    const int peer = (localRank + i) % nvectors;
    // Copy - P2P from local buffer to remote buffer
    if (src != dsts[peer] && count > 0) {
      if (canCopy16(src, dsts[peer], count)) {
        copy<uint4>(
            reinterpret_cast<uint4*>(dsts[peer]),
            reinterpret_cast<const uint4*>(src),
            nbytes / sizeof(uint4),
            groupIdx,
            nGroups);
      } else {
        copy<T>(dsts[peer], src, count, groupIdx, nGroups);
      }
    }
  }
}
