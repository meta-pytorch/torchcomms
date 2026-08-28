// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#if defined(ENABLE_PRIMS)
#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#endif

namespace comms::prims::detail {

__host__ __device__ constexpr bool is_multimem_store_valid(
    uintptr_t destination,
    std::size_t bytes) {
  return bytes == 0 || ((destination & 3) == 0 && (bytes & 3) == 0);
}

} // namespace comms::prims::detail

#if defined(ENABLE_PRIMS)

namespace comms::prims::detail {

__device__ __forceinline__ uint32_t load_u32_unaligned(const char* source) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(source);
  return static_cast<uint32_t>(bytes[0]) |
      (static_cast<uint32_t>(bytes[1]) << 8) |
      (static_cast<uint32_t>(bytes[2]) << 16) |
      (static_cast<uint32_t>(bytes[3]) << 24);
}

__device__ __forceinline__ void multimem_store_u32(
    uint32_t* destination,
    uint32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.relaxed.sys.global.b32 [%0], %1;"
               :
               : "l"(__cvta_generic_to_global(destination)), "r"(value)
               : "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

__device__ __forceinline__ void multimem_store_v4_u32(
    uint4* destination,
    uint4 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(__cvta_generic_to_global(destination)),
                 "r"(value.x),
                 "r"(value.y),
                 "r"(value.z),
                 "r"(value.w)
               : "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

template <int kUnroll>
__device__ __forceinline__ void store_vectors(
    ThreadGroup& group,
    uint4* destination,
    const uint4* source,
    std::size_t count) {
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
        values[index] = source[offset];
      }
    }
#pragma unroll
    for (int index = 0; index < kUnroll; ++index) {
      const std::size_t offset =
          base + static_cast<std::size_t>(index) * groupSize;
      if (offset < count) {
        multimem_store_v4_u32(destination + offset, values[index]);
      }
    }
  }
}

template <int kUnroll>
__device__ __forceinline__ void store_words(
    ThreadGroup& group,
    uint32_t* destination,
    const char* source,
    std::size_t count) {
  static_assert(kUnroll > 0);
  const std::size_t groupSize = group.group_size;
  const std::size_t first = group.thread_id_in_group;
  for (std::size_t base = first; base < count; base += groupSize * kUnroll) {
    uint32_t values[kUnroll];
#pragma unroll
    for (int index = 0; index < kUnroll; ++index) {
      const std::size_t offset =
          base + static_cast<std::size_t>(index) * groupSize;
      if (offset < count) {
        values[index] = load_u32_unaligned(source + offset * sizeof(uint32_t));
      }
    }
#pragma unroll
    for (int index = 0; index < kUnroll; ++index) {
      const std::size_t offset =
          base + static_cast<std::size_t>(index) * groupSize;
      if (offset < count) {
        multimem_store_u32(destination + offset, values[index]);
      }
    }
  }
}

} // namespace comms::prims::detail

namespace comms::prims::multimem {

/**
 * Broadcast bytes from a local source into an arbitrary multicast address.
 *
 * Every member of `group` must call with identical arguments.
 * Source and destination may be identical. Otherwise their ranges must not
 * overlap; like memcpy, this primitive does not provide memmove semantics.
 * Host launchers must call `is_multimem_store_valid` before launch; the device
 * check below is only a fail-fast backstop for contract violations.
 * A nonempty destination must be four-byte aligned and `bytes` must be a
 * multiple of four, matching the smallest NVLS multicast store width.
 * The aligned path uses 16-byte vector stores, while the general path accepts
 * any source alignment and issues four-byte stores.
 * Stores use relaxed system ordering, so callers publish completion with an
 * ordered signal before consumers read their local backing.
 */
template <int kUnroll = 1>
__device__ __forceinline__ void store(
    ThreadGroup& group,
    char* destination,
    const void* source,
    std::size_t bytes) {
  static_assert(kUnroll > 0);
  if (bytes == 0) {
    return;
  }

  const auto destinationAddress = reinterpret_cast<uintptr_t>(destination);
  const bool valid =
      comms::prims::detail::is_multimem_store_valid(destinationAddress, bytes);
  PIPES_DEVICE_CHECK_MSG(
      valid,
      "multimem::store requires a four-byte-aligned destination and extent");
  if (!valid) {
    return;
  }

  const auto sourceAddress = reinterpret_cast<uintptr_t>(source);
  if (((sourceAddress | destinationAddress) & 15) == 0) {
    const std::size_t vectorCount = bytes / sizeof(uint4);
    comms::prims::detail::store_vectors<kUnroll>(
        group,
        reinterpret_cast<uint4*>(destination),
        static_cast<const uint4*>(source),
        vectorCount);
    const std::size_t vectorBytes = vectorCount * sizeof(uint4);
    comms::prims::detail::store_words<kUnroll>(
        group,
        reinterpret_cast<uint32_t*>(destination + vectorBytes),
        static_cast<const char*>(source) + vectorBytes,
        (bytes - vectorBytes) / sizeof(uint32_t));
    return;
  }

  comms::prims::detail::store_words<kUnroll>(
      group,
      reinterpret_cast<uint32_t*>(destination),
      static_cast<const char*>(source),
      bytes / sizeof(uint32_t));
}

} // namespace comms::prims::multimem

#endif // ENABLE_PRIMS
