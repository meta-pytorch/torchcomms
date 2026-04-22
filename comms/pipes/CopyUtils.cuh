// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include "comms/pipes/HipCompat.cuh"

#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

// =============================================================================
// AMD system-coherent store for P2P writes over XGMI
// =============================================================================
// On AMD GPUs, regular stores to remote GPU memory go through L1/L2 cache
// and may not be visible to the remote GPU until a cache flush. For P2P
// transfers, we need system-coherent stores that bypass/flush caches.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)

/**
 * Requires: dst and src must be 8-byte aligned (guaranteed when called via
 * uint4* from memcpy_vectorized_aligned_sys). Misaligned pointers cause
 * undefined behavior on AMD flat_store_dwordx2.
 */
__device__ __forceinline__ void store_sys_u128(uint4* dst, const uint4* src) {
  // Note: 128-bit store is split into two 64-bit stores. Atomicity is not
  // required — callers synchronize the full transfer via signal/barrier
  // primitives before the consumer reads.
#if defined(__gfx942__) || defined(__gfx950__)
  // 16-byte system-coherent store via 2x dwordx2 with sc0 sc1
  const uint64_t* s = reinterpret_cast<const uint64_t*>(src);
  uint64_t* d = reinterpret_cast<uint64_t*>(dst);
  uint64_t v0 = s[0];
  uint64_t v1 = s[1];
  asm volatile("flat_store_dwordx2 %0, %1 sc0 sc1" : : "v"(d), "v"(v0));
  asm volatile("flat_store_dwordx2 %0, %1 sc0 sc1" : : "v"(d + 1), "v"(v1));
#elif defined(__gfx90a__)
  const uint64_t* s = reinterpret_cast<const uint64_t*>(src);
  uint64_t* d = reinterpret_cast<uint64_t*>(dst);
  uint64_t v0 = s[0];
  uint64_t v1 = s[1];
  asm volatile("flat_store_dwordx2 %0, %1 glc slc" : : "v"(d), "v"(v0));
  asm volatile("flat_store_dwordx2 %0, %1 glc slc" : : "v"(d + 1), "v"(v1));
#else
  // Unsupported AMD architecture — plain store lacks system coherence and
  // would silently break P2P correctness. Fail at compile time so new
  // architectures get an explicit implementation.
#error \
    "store_sys_u128: no system-coherent store implementation for this AMD GPU architecture"
#endif
}

#endif // __HIP_DEVICE_COMPILE__

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
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
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
#endif // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
}

// AMD-optimized uint4 copy with system-coherent stores for P2P over XGMI.
// NVIDIA NVLink provides hardware cache coherence for P2P writes, so the
// standard memcpy_vectorized_aligned() is sufficient. AMD XGMI does not
// guarantee coherence — remote stores may remain in local L1/L2 caches —
// so this variant uses explicit cache-bypassing stores (sc0 sc1 / glc slc).
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
template <int kUnroll = 8>
__device__ __forceinline__ void memcpy_vectorized_aligned_sys(
    uint4* dst,
    const uint4* src,
    std::size_t nelems,
    const ThreadGroup& group) {
  const std::size_t kLoopStride = group.group_size * kUnroll;
  const std::size_t numVecsAligned = (nelems / kLoopStride) * kLoopStride;

  for (std::size_t i = group.thread_id_in_group; i < numVecsAligned;
       i += kLoopStride) {
    uint4 v[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      v[j] = src[i + j * group.group_size];
    }
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      store_sys_u128(&dst[i + j * group.group_size], &v[j]);
    }
  }

  for (std::size_t i = numVecsAligned + group.thread_id_in_group; i < nelems;
       i += group.group_size) {
    store_sys_u128(&dst[i], &src[i]);
  }
}
#endif

template <int kUnroll = 8>
__device__ __forceinline__ void memcpy_vectorized(
    char* dst,
    const char* src,
    std::size_t len,
    const ThreadGroup& group) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
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
#endif // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
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
 * Note: Only active on device (__CUDA_ARCH__ / __HIP_DEVICE_COMPILE__). No-op
 * on host.
 */
__device__ __forceinline__ void
assert_buffer_non_overlap(char* dst_d, const char* src_d, std::size_t nbytes) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if (!(src_d + nbytes <= dst_d || dst_d + nbytes <= src_d)) {
    __trap(); // Abort kernel if buffers overlap
  }
#endif // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
}

} // namespace comms::pipes
