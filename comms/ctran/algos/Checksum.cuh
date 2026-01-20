// Copyright (c) Meta Platforms, Inc. and affiliates.

/*
The code is larged based on fbcode/dietgpu/ans/GpuChecksum.cuh.
The original code uses cub library which is not available in ncclx,
we replace related functions with custom implementations.
*/

#pragma once

#include "comms/ctran/algos/DevCommon.cuh"

// Returns the increment needed to aligned the pointer to the next highest
// aligned address
template <int Align>
__device__ __forceinline__ uint32_t getAlignmentRoundUp(const void* p) {
  static_assert(ctran::utils::isPowerOf2(Align));
  uint32_t diff = uint32_t(uintptr_t(p) & uintptr_t(Align - 1));
  return diff == 0 ? 0 : uint32_t(Align) - diff;
}

__device__ __forceinline__ uint32_t warpReduceXor(uint32_t val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
#if defined(__HIP_PLATFORM_AMD__)
    val ^= __shfl_xor(0xffffffffffffffff, val, offset);
#else
    val ^= __shfl_xor_sync(0xffffffff, val, offset);
#endif
  }
  return val;
}

__device__ __forceinline__ uint32_t blockReduceXor(uint32_t val) {
  static __shared__ uint32_t shared[32];
  auto lane = threadIdx.x % warpSize;
  auto warpId = threadIdx.x / warpSize;
  val = warpReduceXor(val);
  if (lane == 0) {
    shared[warpId] = val;
  }
  __syncthreads();
  if (warpId == 0) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    val = warpReduceXor(val);
  }
  return val;
}

template <int Threads>
__global__ void checksumKernel(
    const uint8_t* __restrict__ in,
    const uint32_t size,
    uint32_t* __restrict__ out) {
  // FIXME: general presumption in dietgpu that input data for ANS is only byte
  // aligned, while float data is only float word aligned, whereas ideally we
  // would like a 32 bit checksum. Since there is ultimately no guarantee of
  // anything but byte alignment and we wish to compute the same checksum
  // regardless of memory placement, the only checksum that makes sense to
  // produce is uint8.
  // We can fix this to compute a full 32-bit checksum by keeping track of
  // initial alignment and shuffling data around I think.
  uint32_t checksum32 = 0;

  // If the size of batch is smaller than the increment for alignment, we only
  // handle the batch
  auto roundUp4 = min(size, getAlignmentRoundUp<sizeof(uint4)>(in));

  // The size of data that remains after alignment
  auto remaining = size - roundUp4;

  // The size of data (in uint4 words) that we can process with alignment
  uint32_t numU4 = ctran::utils::divDown(remaining, sizeof(uint4));

  auto inAligned = in + roundUp4;
  auto inAligned4 = (const uint4*)inAligned;

  // Handle the non-aligned portion that we have to load as single bytes, if any
  if (blockIdx.x == 0 && threadIdx.x < roundUp4) {
    static_assert(sizeof(uint4) <= Threads, "");
    checksum32 ^= in[threadIdx.x];
  }

  // Handle the portion that is aligned and uint4 vectorizable
  // 37.60 us / 80.76% gmem / 51.29% smem for uint4 on A100
  for (uint32_t i = blockIdx.x * Threads + threadIdx.x; i < numU4;
       i += gridDim.x * Threads) {
    uint4 v = inAligned4[i];

    checksum32 ^= v.x;
    checksum32 ^= v.y;
    checksum32 ^= v.z;
    checksum32 ^= v.w;
  }

  if (blockIdx.x == 0) {
    // Handle the remainder portion that doesn't comprise full words
    int i = numU4 * sizeof(uint4) + threadIdx.x;
    if (i < remaining) {
      checksum32 ^= inAligned[i];
    }
  }

  // Fold the bytes of checksum32
  checksum32 = (checksum32 & 0xffU) ^ ((checksum32 >> 8) & 0xffU) ^
      ((checksum32 >> 16) & 0xffU) ^ ((checksum32 >> 24) & 0xffU);

  checksum32 = blockReduceXor(checksum32);

  // The first thread of the block performs the atomic XOR operation
  if (threadIdx.x == 0) {
    atomicXor(out, checksum32);
  }
}
