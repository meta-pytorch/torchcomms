// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

/**
 * memcpy_vectorized_aligned - High-performance vectorized memory copy
 *
 * Cooperative memory copy optimized for GPU-to-GPU transfers with:
 *   - Configurable unrolling via template parameter (default 4x)
 *   - Vectorized loads/stores (16-byte uint4 operations)
 *   - Coalesced memory access pattern
 *   - Requires aligned memory (aligned with vector load/store size)
 *
 * @tparam VecType Vector type for loads/stores (typically uint4 = 16 bytes)
 * @tparam kUnroll Unroll factor (default 4, optimal for most transfers)
 * @param dst_base Base pointer to destination buffer
 * @param src_base Base pointer to source buffer
 * @param nelems Number of elements of VecType to copy
 * @param group ThreadGroup for cooperative copy (all threads participate)
 *
 * STRIDING PATTERN
 * =====================================================
 * Each thread accesses elements strided by group_size, not consecutive.
 * This ensures coalesced memory transactions across the thread group.
 *
 * Example with kUnroll=4, group_size=128:
 *   Thread 0: [0, 128, 256, 384]
 *   Thread 1: [1, 129, 257, 385]
 *   ...
 *   Thread 127: [127, 255, 383, 511]
 *
 * This gives perfect 128-thread-wide coalesced accesses per unroll iteration.
 *
 * UNROLL FACTOR GUIDELINES:
 * =========================
 * - kUnroll=8 (default): Optimal with coalesced striding pattern
 * - kUnroll=4: Slightly lower ILP but less register pressure
 * - kUnroll=2: For very small messages or high register pressure scenarios
 */
template <typename VecType, int kUnroll = 8>
__device__ __forceinline__ void memcpy_vectorized_aligned(
    VecType* dst_p,
    const VecType* src_p,
    std::size_t nelems,
    const ThreadGroup& group) {
#ifdef __CUDA_ARCH__
  // Loop stride: group_size threads × kUnroll elements each
  const std::size_t kLoopStride = group.group_size * kUnroll;
  const std::size_t numVecsAligned = (nelems / kLoopStride) * kLoopStride;
  VecType* __restrict__ dst = dst_p;
  const VecType* __restrict__ src = src_p;

  // Main loop: coalesced strided access pattern (deep_ep style)
  // Each thread loads kUnroll elements, strided by group_size
  for (std::size_t i = group.thread_id_in_group; i < numVecsAligned;
       i += kLoopStride) {
    VecType v[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      v[j] = src[i + j * group.group_size];
    }
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      dst[i + j * group.group_size] = v[j];
    }
  }

  // Handle remaining vectors (not fitting in kLoopStride groups)
  for (std::size_t i = numVecsAligned + group.thread_id_in_group; i < nelems;
       i += group.group_size) {
    dst[i] = src[i];
  }
#endif // __CUDA_ARCH__
}

template <int kUnroll = 8>
__device__ __forceinline__ void memcpy_vectorized(
    char* dst,
    const char* src,
    std::size_t len,
    const ThreadGroup& group) {
#ifdef __CUDA_ARCH__
  constexpr std::size_t kAlignment = sizeof(uint4);
  if ((uintptr_t)dst % kAlignment == 0 && (uintptr_t)src % kAlignment == 0) {
    const std::size_t nelems = len / kAlignment;
    uint4* __restrict__ dst_p = reinterpret_cast<uint4*>(dst);
    const uint4* __restrict__ src_p = reinterpret_cast<const uint4*>(src);
    memcpy_vectorized_aligned<uint4, kUnroll>(dst_p, src_p, nelems, group);
    len -= nelems * kAlignment;
    if (len == 0) {
      return;
    }
    dst = reinterpret_cast<char*>(dst_p + nelems);
    src = reinterpret_cast<const char*>(src_p + nelems);
  }

  memcpy_vectorized_aligned<char, kUnroll>(dst, src, len, group);
#endif // __CUDA_ARCH__
}

/**
 * memcpy_vectorized_dual_dest_aligned - Dual-destination vectorized memory copy
 *
 * Reads each element from src once, then stores to both dst1 and dst2.
 * Eliminates the extra HBM read that occurs when doing two sequential copies
 * (src->dst1 then dst1->dst2). Same striding pattern as
 * memcpy_vectorized_aligned.
 *
 * @tparam VecType Vector type for loads/stores (typically uint4 = 16 bytes)
 * @tparam kUnroll Unroll factor (default 8)
 * @param dst1_p First destination buffer pointer
 * @param dst2_p Second destination buffer pointer
 * @param src_p Source buffer pointer
 * @param nelems Number of elements of VecType to copy
 * @param group ThreadGroup for cooperative copy (all threads participate)
 */
template <typename VecType, int kUnroll = 8>
__device__ __forceinline__ void memcpy_vectorized_dual_dest_aligned(
    VecType* dst1_p,
    VecType* dst2_p,
    const VecType* src_p,
    std::size_t nelems,
    const ThreadGroup& group) {
#ifdef __CUDA_ARCH__
  const std::size_t kLoopStride = group.group_size * kUnroll;
  const std::size_t numVecsAligned = (nelems / kLoopStride) * kLoopStride;
  VecType* __restrict__ dst1 = dst1_p;
  VecType* __restrict__ dst2 = dst2_p;
  const VecType* __restrict__ src = src_p;

  for (std::size_t i = group.thread_id_in_group; i < numVecsAligned;
       i += kLoopStride) {
    VecType v[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      v[j] = src[i + j * group.group_size];
    }
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      dst1[i + j * group.group_size] = v[j];
    }
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      dst2[i + j * group.group_size] = v[j];
    }
  }

  for (std::size_t i = numVecsAligned + group.thread_id_in_group; i < nelems;
       i += group.group_size) {
    VecType v = src[i];
    dst1[i] = v;
    dst2[i] = v;
  }
#endif // __CUDA_ARCH__
}

/**
 * memcpy_vectorized_dual_dest - Byte-level dual-destination vectorized copy
 *
 * Copies len bytes from src to both dst1 and dst2 with a single source read.
 * Checks alignment of all three pointers to select vectorized (uint4) or
 * byte-level path.
 *
 * @tparam kUnroll Unroll factor (default 8)
 * @param dst1 First destination buffer
 * @param dst2 Second destination buffer
 * @param src Source buffer
 * @param len Number of bytes to copy
 * @param group ThreadGroup for cooperative copy
 */
template <int kUnroll = 8>
__device__ __forceinline__ void memcpy_vectorized_dual_dest(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t len,
    const ThreadGroup& group) {
#ifdef __CUDA_ARCH__
  constexpr std::size_t kAlignment = sizeof(uint4);
  if ((uintptr_t)dst1 % kAlignment == 0 && (uintptr_t)dst2 % kAlignment == 0 &&
      (uintptr_t)src % kAlignment == 0) {
    const std::size_t nelems = len / kAlignment;
    uint4* __restrict__ dst1_p = reinterpret_cast<uint4*>(dst1);
    uint4* __restrict__ dst2_p = reinterpret_cast<uint4*>(dst2);
    const uint4* __restrict__ src_p = reinterpret_cast<const uint4*>(src);
    memcpy_vectorized_dual_dest_aligned<uint4, kUnroll>(
        dst1_p, dst2_p, src_p, nelems, group);
    len -= nelems * kAlignment;
    if (len == 0) {
      return;
    }
    dst1 = reinterpret_cast<char*>(dst1_p + nelems);
    dst2 = reinterpret_cast<char*>(dst2_p + nelems);
    src = reinterpret_cast<const char*>(src_p + nelems);
  }

  memcpy_vectorized_dual_dest_aligned<char, kUnroll>(
      dst1, dst2, src, len, group);
#endif // __CUDA_ARCH__
}

/**
 * assert_buffer_non_overlap - Assert that source and destination buffers do not
 * overlap
 *
 * Checks that the memory regions [src_d, src_d + nbytes) and
 * [dst_d, dst_d + nbytes) are disjoint (non-overlapping). If they overlap,
 * the kernel is aborted via __trap().
 *
 * This is a safety check for memory copy operations that assume non-overlapping
 * buffers. Overlapping buffers with memcpy-style operations lead to undefined
 * behavior.
 *
 * @param dst_d Destination buffer pointer
 * @param src_d Source buffer pointer
 * @param nbytes Size of both buffers in bytes
 *
 * Note: Only active on device (__CUDA_ARCH__). No-op on host.
 */
__device__ __forceinline__ void
assert_buffer_non_overlap(char* dst_d, const char* src_d, std::size_t nbytes) {
#ifdef __CUDA_ARCH__
  if (!(src_d + nbytes <= dst_d || dst_d + nbytes <= src_d)) {
    __trap(); // Abort kernel if buffers overlap
  }
#endif // __CUDA_ARCH__
}

} // namespace comms::pipes
