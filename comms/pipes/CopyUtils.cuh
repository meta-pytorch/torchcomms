// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <array>
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
 * memcpy_vectorized_multi_dest_aligned - Multi-destination vectorized memory
 * copy
 *
 * Reads each element from src once, then stores to all N destination buffers.
 * Eliminates the (N-1) extra HBM reads that occur when doing N sequential
 * copies. Same striding pattern as memcpy_vectorized_aligned.
 *
 * REQUIREMENTS:
 * =============
 * - All destination pointers and the source pointer must be aligned to
 *   sizeof(VecType)
 * - Destination buffers must not alias each other or the source buffer
 * - N must be >= 1 (enforced by static_assert)
 *
 * PERFORMANCE NOTES:
 * ==================
 * - For N=1, prefer memcpy_vectorized_aligned which has full __restrict__
 *   qualifier coverage
 * - __restrict__ is applied to src_p only; the load-store separation pattern
 *   (all loads complete before any stores) provides the key optimization
 *   regardless
 *
 * @tparam N Number of destination buffers (must be >= 1)
 * @tparam VecType Vector type for loads/stores (typically uint4 = 16 bytes)
 * @tparam kUnroll Unroll factor (default 8)
 * @param dst_ps Array of N destination buffer pointers
 * @param src_p Source buffer pointer
 * @param nelems Number of elements of VecType to copy
 * @param group ThreadGroup for cooperative copy (all threads participate)
 *
 * N BOUNDS:
 * =========
 * - N is limited to 8 maximum to avoid excessive register pressure (each
 *   destination requires kUnroll additional store instructions per iteration)
 */
template <std::size_t N, typename VecType, int kUnroll = 8>
__device__ __forceinline__ void memcpy_vectorized_multi_dest_aligned(
    const std::array<VecType*, N>& dst_ps,
    const VecType* __restrict__ src_p,
    std::size_t nelems,
    const ThreadGroup& group) {
  static_assert(
      N > 0 && N <= 8, "N must be between 1 and 8 (register pressure)");
#ifdef __CUDA_ARCH__
  // std::array<T,N> is layout-compatible with T[N]; cast to raw pointer for
  // CUDA device code (std::array member functions are not __device__-qualified)
  VecType* const* dst = reinterpret_cast<VecType* const*>(&dst_ps);

  const std::size_t kLoopStride = group.group_size * kUnroll;
  const std::size_t numVecsAligned = (nelems / kLoopStride) * kLoopStride;

  // Main loop: coalesced strided access pattern
  for (std::size_t i = group.thread_id_in_group; i < numVecsAligned;
       i += kLoopStride) {
    // Phase 1: Load kUnroll elements from src into registers
    VecType v[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      v[j] = src_p[i + j * group.group_size];
    }
    // Phase 2: Store to each of N destinations
#pragma unroll
    for (std::size_t d = 0; d < N; ++d) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        dst[d][i + j * group.group_size] = v[j];
      }
    }
  }

  // Remainder: elements not fitting in full kLoopStride groups
  for (std::size_t i = numVecsAligned + group.thread_id_in_group; i < nelems;
       i += group.group_size) {
    VecType v = src_p[i];
#pragma unroll
    for (std::size_t d = 0; d < N; ++d) {
      dst[d][i] = v;
    }
  }
#endif // __CUDA_ARCH__
}

/**
 * memcpy_vectorized_multi_dest - Byte-level multi-destination vectorized copy
 *
 * Copies len bytes from src to all N destination buffers with a single source
 * read. Checks alignment of all (N+1) pointers to select vectorized (uint4) or
 * byte-level path.
 *
 * @tparam N Number of destination buffers (must be >= 1)
 * @tparam kUnroll Unroll factor (default 8)
 * @param dsts Array of N destination buffer pointers
 * @param src Source buffer
 * @param len Number of bytes to copy
 * @param group ThreadGroup for cooperative copy
 */
template <std::size_t N, int kUnroll = 8>
__device__ __forceinline__ void memcpy_vectorized_multi_dest(
    const std::array<char*, N>& dsts,
    const char* src,
    std::size_t len,
    const ThreadGroup& group) {
  static_assert(
      N > 0 && N <= 8, "N must be between 1 and 8 (register pressure)");
#ifdef __CUDA_ARCH__
  constexpr std::size_t kAlignment = sizeof(uint4);

  // std::array<T,N> is layout-compatible with T[N]; cast to raw pointer for
  // CUDA device code (std::array member functions are not __device__-qualified)
  char* const* dsts_raw = reinterpret_cast<char* const*>(&dsts);

  // Check alignment of all N destinations + source
  // Use bitwise AND to avoid branch divergence in the unrolled loop
  bool all_aligned = ((uintptr_t)src % kAlignment == 0);
#pragma unroll
  for (std::size_t d = 0; d < N; ++d) {
    all_aligned = all_aligned & ((uintptr_t)dsts_raw[d] % kAlignment == 0);
  }

  // Local copies for pointer adjustment after aligned section
  char* local_dsts[N];
#pragma unroll
  for (std::size_t d = 0; d < N; ++d) {
    local_dsts[d] = dsts_raw[d];
  }
  const char* local_src = src;

  if (all_aligned) {
    const std::size_t nelems = len / kAlignment;
    uint4* uint4_dsts[N];
#pragma unroll
    for (std::size_t d = 0; d < N; ++d) {
      uint4_dsts[d] = reinterpret_cast<uint4*>(local_dsts[d]);
    }
    const uint4* __restrict__ src_p = reinterpret_cast<const uint4*>(local_src);

    memcpy_vectorized_multi_dest_aligned<N, uint4, kUnroll>(
        reinterpret_cast<const std::array<uint4*, N>&>(uint4_dsts),
        src_p,
        nelems,
        group);

    len -= nelems * kAlignment;
    if (len == 0) {
      return;
    }
    // Adjust pointers for remainder bytes
#pragma unroll
    for (std::size_t d = 0; d < N; ++d) {
      local_dsts[d] = reinterpret_cast<char*>(uint4_dsts[d] + nelems);
    }
    local_src = reinterpret_cast<const char*>(src_p + nelems);
  }

  memcpy_vectorized_multi_dest_aligned<N, char, kUnroll>(
      reinterpret_cast<const std::array<char*, N>&>(local_dsts),
      local_src,
      len,
      group);
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
