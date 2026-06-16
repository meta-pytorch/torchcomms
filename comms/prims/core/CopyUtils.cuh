// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include "comms/prims/transport/amd/HipHostCompat.h"

#include "comms/prims/core/ThreadGroup.cuh"

namespace comms::prims {

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

/**
 * memcpy_vectorized_aligned (dual-destination) - Fused recv-and-forward copy
 *
 * Reads source data once and writes to two destinations simultaneously.
 * This halves the read bandwidth vs two separate memcpy_vectorized_aligned
 * calls, which is critical for ring algorithm forwarding where an
 * intermediate rank needs to copy to both its local user buffer and the
 * successor's remote staging buffer.
 *
 * Same striding pattern as the single-dst version above.
 */
template <typename VecType, int kUnroll = 8>
__device__ __forceinline__ void memcpy_vectorized_aligned(
    VecType* dst1_p,
    VecType* dst2_p,
    const VecType* src_p,
    std::size_t nelems,
    const ThreadGroup& group) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
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
    VecType val = src[i];
    dst1[i] = val;
    dst2[i] = val;
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

// AMD dual-destination copy: one destination gets system-coherent stores
// (for NVLink remote write to successor), the other gets plain stores
// (for local user buffer).
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
template <int kUnroll = 8>
__device__ __forceinline__ void memcpy_vectorized_aligned_sys(
    uint4* dst_local,
    uint4* dst_remote,
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
      dst_local[i + j * group.group_size] = v[j];
    }
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      store_sys_u128(&dst_remote[i + j * group.group_size], &v[j]);
    }
  }

  for (std::size_t i = numVecsAligned + group.thread_id_in_group; i < nelems;
       i += group.group_size) {
    dst_local[i] = src[i];
    store_sys_u128(&dst_remote[i], &src[i]);
  }
}
#endif

/**
 * assert_buffer_non_overlap - Assert that source and destination buffers do not
 * overlap.
 *
 * memcpy_vectorized uses memcpy-style, __restrict__ vectorized loads/stores and
 * a cooperative striding order, so overlapping source/destination ranges are
 * not safe. Callers that support in-place operation must bypass the copy.
 */
__device__ __forceinline__ void assert_buffer_non_overlap(
    char* dst_d,
    const char* src_d,
    std::size_t nbytes,
    const ThreadGroup& group) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const uintptr_t dst_begin = reinterpret_cast<uintptr_t>(dst_d);
  const uintptr_t src_begin = reinterpret_cast<uintptr_t>(src_d);
  const uintptr_t dst_end = dst_begin + nbytes;
  const uintptr_t src_end = src_begin + nbytes;
  if (nbytes > 0 && dst_begin != src_begin && dst_begin < src_end &&
      src_begin < dst_end) {
    if (group.is_leader()) {
      printf(
          "memcpy_vectorized partial overlap: dst=[0x%llx,0x%llx) src=[0x%llx,0x%llx) nbytes=%llu block=(%u,%u,%u) thread=(%u,%u,%u)\n",
          static_cast<unsigned long long>(dst_begin),
          static_cast<unsigned long long>(dst_end),
          static_cast<unsigned long long>(src_begin),
          static_cast<unsigned long long>(src_end),
          static_cast<unsigned long long>(nbytes),
          blockIdx.x,
          blockIdx.y,
          blockIdx.z,
          threadIdx.x,
          threadIdx.y,
          threadIdx.z);
    }
    __trap();
  }
#endif // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
}

template <int kUnroll = 8>
__device__ __forceinline__ void memcpy_vectorized(
    char* dst,
    const char* src,
    std::size_t len,
    const ThreadGroup& group) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if (len == 0 || dst == src) {
    return;
  }
  assert_buffer_non_overlap(dst, src, len, group);

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
 * memcpy_vectorized (dual-destination) - Fused copy to two destinations.
 *
 * Handles alignment and dispatches to the appropriate dual-dst
 * memcpy_vectorized_aligned variant. Reads source once, writes to both
 * destinations.
 */
template <int kUnroll = 8>
__device__ __forceinline__ void memcpy_vectorized(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t len,
    const ThreadGroup& group) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if (len == 0 || (dst1 == src && dst2 == src)) {
    return;
  }
  if (dst1 == src) {
    memcpy_vectorized<kUnroll>(dst2, src, len, group);
    return;
  }
  if (dst2 == src || dst1 == dst2) {
    memcpy_vectorized<kUnroll>(dst1, src, len, group);
    return;
  }
  assert_buffer_non_overlap(dst1, src, len, group);
  assert_buffer_non_overlap(dst2, src, len, group);
  assert_buffer_non_overlap(dst1, dst2, len, group);

  constexpr std::size_t kAlignment = sizeof(uint4);
  if ((uintptr_t)dst1 % kAlignment == 0 && (uintptr_t)dst2 % kAlignment == 0 &&
      (uintptr_t)src % kAlignment == 0) {
    const std::size_t nelems = len / kAlignment;
    uint4* __restrict__ dst1_p = reinterpret_cast<uint4*>(dst1);
    uint4* __restrict__ dst2_p = reinterpret_cast<uint4*>(dst2);
    const uint4* __restrict__ src_p = reinterpret_cast<const uint4*>(src);
    memcpy_vectorized_aligned<uint4, kUnroll>(
        dst1_p, dst2_p, src_p, nelems, group);
    len -= nelems * kAlignment;
    if (len == 0) {
      return;
    }
    dst1 = reinterpret_cast<char*>(dst1_p + nelems);
    dst2 = reinterpret_cast<char*>(dst2_p + nelems);
    src = reinterpret_cast<const char*>(src_p + nelems);
  }

  memcpy_vectorized_aligned<char, kUnroll>(dst1, dst2, src, len, group);
#endif // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
}

} // namespace comms::prims
