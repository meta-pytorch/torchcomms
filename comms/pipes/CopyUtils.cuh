// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

/**
 * copy_chunk_vectorized - High-performance vectorized memory copy
 *
 * Cooperative memory copy optimized for GPU-to-GPU transfers with:
 *   - Configurable unrolling via template parameter (default 4x)
 *   - Vectorized loads/stores (16-byte uint4 operations)
 *   - Coalesced memory access pattern
 *   - Remainder handling for non-aligned sizes
 *
 * @tparam VecType Vector type for loads/stores (typically uint4 = 16 bytes)
 * @tparam kUnroll Unroll factor (default 4, optimal for most transfers)
 * @param dst_base Base pointer to destination buffer
 * @param src_base Base pointer to source buffer
 * @param chunk_bytes Number of bytes to copy
 * @param dst_offset Offset into destination buffer (in bytes)
 * @param src_offset Offset into source buffer (in bytes)
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
__device__ __forceinline__ void copy_chunk_vectorized(
    char* dst_base,
    const char* src_base,
    std::size_t chunk_bytes,
    std::size_t dst_offset,
    std::size_t src_offset,
    const ThreadGroup& group) {
#ifdef __CUDA_ARCH__
  constexpr std::size_t kVecSize = sizeof(VecType);

  // Loop stride: group_size threads × kUnroll elements each
  const std::size_t kLoopStride = group.group_size * kUnroll;
  const std::size_t numVecs = chunk_bytes / kVecSize;
  const std::size_t numVecsAligned = (numVecs / kLoopStride) * kLoopStride;

  const VecType* srcVec =
      reinterpret_cast<const VecType*>(src_base + src_offset);
  VecType* dstVec = reinterpret_cast<VecType*>(dst_base + dst_offset);

  // Main loop: coalesced strided access pattern (deep_ep style)
  // Each thread loads kUnroll elements, strided by group_size
  for (std::size_t i = group.thread_id_in_group; i < numVecsAligned;
       i += kLoopStride) {
    VecType v[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      v[j] = srcVec[i + j * group.group_size];
    }
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      dstVec[i + j * group.group_size] = v[j];
    }
  }

  // Handle remaining vectors (not fitting in kLoopStride groups)
  for (std::size_t i = numVecsAligned + group.thread_id_in_group; i < numVecs;
       i += group.group_size) {
    dstVec[i] = srcVec[i];
  }

  // Handle remainder bytes (non-vector-aligned tail)
  const std::size_t vec_aligned_bytes = numVecs * kVecSize;
  const std::size_t remainder = chunk_bytes - vec_aligned_bytes;
  if (remainder > 0) {
    const char* src_remainder = src_base + src_offset + vec_aligned_bytes;
    char* dst_remainder = dst_base + dst_offset + vec_aligned_bytes;
    for (std::size_t i = group.thread_id_in_group; i < remainder;
         i += group.group_size) {
      dst_remainder[i] = src_remainder[i];
    }
  }
#endif // __CUDA_ARCH__
}

} // namespace comms::pipes
