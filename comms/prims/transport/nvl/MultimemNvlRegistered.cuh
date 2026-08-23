// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace comms::prims::detail {

template <typename T>
__host__ __device__ constexpr bool is_reduce_broadcast_valid(
    uintptr_t destination,
    uintptr_t source,
    std::size_t elements) {
  if (elements == 0) {
    return true;
  }
  if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
    return ((destination | source) & 15) == 0 && (elements & 7) == 0;
  }
  return ((destination | source) & 3) == 0;
}

} // namespace comms::prims::detail

#if defined(ENABLE_PRIMS)

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/transport/nvl/MultimemNvlReduce.cuh"
#include "comms/prims/transport/nvl/MultimemNvlStore.cuh"

namespace comms::prims::detail {

template <int kUnroll, typename Load>
__device__ __forceinline__ void reduce_broadcast_vectors(
    ThreadGroup& group,
    uint4* destination,
    const uint4* source,
    std::size_t count,
    Load load) {
  static_assert(kUnroll > 0);
  const std::size_t groupSize = group.group_size;
  const std::size_t first = group.thread_id_in_group;
  for (std::size_t base = first; base < count; base += groupSize * kUnroll) {
    uint4 values[kUnroll];
#pragma unroll
    for (int index = 0; index < kUnroll; ++index) {
      const std::size_t offset =
          base + static_cast<std::size_t>(index) * groupSize;
      if (offset < count) {
        values[index] = load(source + offset);
      }
    }
#pragma unroll
    for (int index = 0; index < kUnroll; ++index) {
      const std::size_t offset =
          base + static_cast<std::size_t>(index) * groupSize;
      if (offset < count) {
        comms::prims::detail::multimem_store_v4_u32(
            destination + offset, values[index]);
      }
    }
  }
}

} // namespace comms::prims::detail

namespace comms::prims::multimem {

/**
 * Reduce from one multicast address and broadcast to another without scratch.
 *
 * The reduced value remains in registers between `multimem.ld_reduce` and
 * `multimem.st`.
 * The primitive performs no cross-rank synchronization.
 * In-place callers partition the range into disjoint rank-owned shards and
 * bracket the operation with team barriers.
 * This primitive exposes fp16 and bf16 only as complete 16-byte reduction
 * packs, so those types require aligned source and destination ranges whose
 * element counts are multiples of eight.
 */
template <typename T, int kUnroll = 1, bool kAccF32 = true>
__device__ __forceinline__ void reduce_broadcast_at(
    ThreadGroup& group,
    T* destination,
    const T* source,
    std::size_t elements) {
  static_assert(kUnroll > 0);
  if (!comms::prims::detail::is_reduce_broadcast_valid<T>(
          reinterpret_cast<uintptr_t>(destination),
          reinterpret_cast<uintptr_t>(source),
          elements)) {
    __trap();
    return;
  }
  if (elements == 0) {
    return;
  }

  if constexpr (std::is_same_v<T, float>) {
    const auto addresses = reinterpret_cast<uintptr_t>(destination) |
        reinterpret_cast<uintptr_t>(source);
    const std::size_t vectorCount = (addresses & 15) == 0 ? elements / 4 : 0;
    if (vectorCount != 0) {
      comms::prims::detail::reduce_broadcast_vectors<kUnroll>(
          group,
          reinterpret_cast<uint4*>(destination),
          reinterpret_cast<const uint4*>(source),
          vectorCount,
          [](const uint4* address) {
            const float4 reduced =
                comms::prims::detail::multimem_ld_reduce_v4_f32(
                    reinterpret_cast<const float4*>(address));
            uint4 bits;
            __builtin_memcpy(&bits, &reduced, sizeof(bits));
            return bits;
          });
    }
    const std::size_t stride = group.group_size;
    for (std::size_t index = vectorCount * 4 + group.thread_id_in_group;
         index < elements;
         index += stride) {
      const float reduced =
          comms::prims::detail::multimem_ld_reduce_f32(source + index);
      uint32_t bits;
      __builtin_memcpy(&bits, &reduced, sizeof(bits));
      comms::prims::detail::multimem_store_u32(
          reinterpret_cast<uint32_t*>(destination + index), bits);
    }
  } else if constexpr (std::is_same_v<T, __half>) {
    comms::prims::detail::reduce_broadcast_vectors<kUnroll>(
        group,
        reinterpret_cast<uint4*>(destination),
        reinterpret_cast<const uint4*>(source),
        elements / 8,
        [](const uint4* address) {
          return comms::prims::detail::multimem_ld_reduce_v4_f16x2<kAccF32>(
              address);
        });
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    comms::prims::detail::reduce_broadcast_vectors<kUnroll>(
        group,
        reinterpret_cast<uint4*>(destination),
        reinterpret_cast<const uint4*>(source),
        elements / 8,
        [](const uint4* address) {
          return comms::prims::detail::multimem_ld_reduce_v4_bf16x2<kAccF32>(
              address);
        });
  } else if constexpr (std::is_same_v<T, int32_t>) {
    const std::size_t groupSize = group.group_size;
    const std::size_t first = group.thread_id_in_group;
    for (std::size_t base = first; base < elements;
         base += groupSize * kUnroll) {
      int32_t values[kUnroll];
#pragma unroll
      for (int index = 0; index < kUnroll; ++index) {
        const std::size_t offset =
            base + static_cast<std::size_t>(index) * groupSize;
        if (offset < elements) {
          values[index] =
              comms::prims::detail::multimem_ld_reduce_s32(source + offset);
        }
      }
#pragma unroll
      for (int index = 0; index < kUnroll; ++index) {
        const std::size_t offset =
            base + static_cast<std::size_t>(index) * groupSize;
        if (offset < elements) {
          comms::prims::detail::multimem_store_u32(
              reinterpret_cast<uint32_t*>(destination + offset),
              static_cast<uint32_t>(values[index]));
        }
      }
    }
  } else {
    static_assert(sizeof(T) == 0, "unsupported reduce-broadcast datatype");
  }
}

} // namespace comms::prims::multimem

#endif // ENABLE_PRIMS
