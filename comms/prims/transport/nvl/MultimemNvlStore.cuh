// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// multimem.st broadcast primitives for single-NVL-domain collectives.
//
// Holds the store side of the NVLS staging model: the raw `multimem.st` PTX
// emitters (`comms::prims::detail`) and the public bulk-broadcast entry point
// `multimem::store<>` that broadcasts a local buffer into a multicast VA (every
// store through the VA is replicated by NVSwitch into all ranks' backings). The
// reduce side lives in `MultimemNvlReduce.cuh`; the staging orchestration that
// composes both lives in `MultimemNvlStaging.cuh`.

// clang-tidy analyzes this .cuh as a standalone main file and misflags the
// pragma; it is a genuine include-once header. False positive, so suppress it.
// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#if defined(ENABLE_PRIMS)

#include <cstddef>
#include <cstdint>

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/transport/nvl/MultimemNvlTransportDevice.cuh"

// PTX helpers extend the transport header's existing `comms::prims::detail`
// namespace (which already holds the release.sys signal-path emitters). The
// public free-function entry point (`store<>`) lives in
// `comms::prims::multimem` further below and delegates into `detail::` for the
// raw PTX.
namespace comms::prims::detail {

// ----------------------------------------------------------------------------
// multimem.st store helpers (relaxed.sys, data-path).
// Signal-path emitters live in MultimemNvlTransportDevice.cuh as
// `detail::multimem_store_release_sys_u64` / `multimem_red_release_sys_add_u64`
// (release.sys) and are separate from these because they carry the
// producer/consumer handshake.
// ----------------------------------------------------------------------------

__device__ __forceinline__ uint8_t load_u8_unaligned(const char* src) {
  return *reinterpret_cast<const uint8_t*>(src);
}

__device__ __forceinline__ uint16_t load_u16_unaligned(const char* src) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(src);
  return static_cast<uint16_t>(bytes[0]) |
      (static_cast<uint16_t>(bytes[1]) << 8);
}

__device__ __forceinline__ uint32_t load_u32_unaligned(const char* src) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(src);
  return static_cast<uint32_t>(bytes[0]) |
      (static_cast<uint32_t>(bytes[1]) << 8) |
      (static_cast<uint32_t>(bytes[2]) << 16) |
      (static_cast<uint32_t>(bytes[3]) << 24);
}

__device__ __forceinline__ uint64_t load_u64_unaligned(const char* src) {
  return static_cast<uint64_t>(load_u32_unaligned(src)) |
      (static_cast<uint64_t>(load_u32_unaligned(src + 4)) << 32);
}

__device__ __forceinline__ uint4 load_v4_u32_unaligned(const char* src) {
  return uint4{
      load_u32_unaligned(src),
      load_u32_unaligned(src + 4),
      load_u32_unaligned(src + 8),
      load_u32_unaligned(src + 12)};
}

// NOTE: multimem_store_u8 and multimem_store_u16 emit regular `st.global.bN`
// rather than `multimem.st.*`. PTX `multimem.st.*` only exists for 4/8/16-byte
// widths; for 1- and 2-byte sub-tail stores the multicast semantics still
// apply because `dst` is a multimem virtual address - every store through that
// VA is replicated by NVSwitch regardless of the underlying PTX opcode. The
// `multimem_` prefix here denotes "store into the multimem VA", not the
// `multimem.*` PTX family.
__device__ __forceinline__ void multimem_store_u8(uint8_t* dst, uint8_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("st.global.b8 [%0], %1;"
               :
               : "l"(dst), "r"(static_cast<uint32_t>(v))
               : "memory");
#else
  *dst = v;
#endif
}

__device__ __forceinline__ void multimem_store_u16(uint16_t* dst, uint16_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("st.global.b16 [%0], %1;" : : "l"(dst), "h"(v) : "memory");
#else
  *dst = v;
#endif
}

__device__ __forceinline__ void multimem_store_u32(uint32_t* dst, uint32_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.global.b32 [%0], %1;"
               :
               : "l"(dst), "r"(v)
               : "memory");
#else
  *dst = v;
#endif
}

__device__ __forceinline__ void multimem_store_u64(uint64_t* dst, uint64_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.global.b64 [%0], %1;"
               :
               : "l"(dst), "l"(v)
               : "memory");
#else
  *dst = v;
#endif
}

__device__ __forceinline__ void multimem_store_v4_u32(uint4* dst, uint4 v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
#else
  *dst = v;
#endif
}

template <int kUnroll>
__device__ __forceinline__ void strided_multimem_store_aligned(
    comms::prims::ThreadGroup& group,
    uint4* dstVec,
    const uint4* srcVec,
    std::size_t vecCount) {
  static_assert(kUnroll > 0);
  constexpr std::size_t unroll = static_cast<std::size_t>(kUnroll);
  const std::size_t loopStride =
      static_cast<std::size_t>(group.group_size) * unroll;
  const std::size_t alignedVecCount = (vecCount / loopStride) * loopStride;

  for (std::size_t i = group.thread_id_in_group; i < alignedVecCount;
       i += loopStride) {
    uint4 vals[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      vals[j] = srcVec[i + static_cast<std::size_t>(j) * group.group_size];
    }
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      const std::size_t offset =
          i + static_cast<std::size_t>(j) * group.group_size;
      multimem_store_v4_u32(dstVec + offset, vals[j]);
    }
  }

  for (std::size_t i = alignedVecCount + group.thread_id_in_group; i < vecCount;
       i += group.group_size) {
    multimem_store_v4_u32(dstVec + i, srcVec[i]);
  }
}

__device__ __forceinline__ void
multimem_store_bytes(char* dst, const char* src, std::size_t bytes) {
  while (bytes > 0) {
    const auto dstAddr = reinterpret_cast<uintptr_t>(dst);
    if (bytes >= sizeof(uint4) && (dstAddr & 0xF) == 0) {
      multimem_store_v4_u32(
          reinterpret_cast<uint4*>(dst), load_v4_u32_unaligned(src));
      dst += sizeof(uint4);
      src += sizeof(uint4);
      bytes -= sizeof(uint4);
    } else if (bytes >= sizeof(uint64_t) && (dstAddr & 0x7) == 0) {
      multimem_store_u64(
          reinterpret_cast<uint64_t*>(dst), load_u64_unaligned(src));
      dst += sizeof(uint64_t);
      src += sizeof(uint64_t);
      bytes -= sizeof(uint64_t);
    } else if (bytes >= sizeof(uint32_t) && (dstAddr & 0x3) == 0) {
      multimem_store_u32(
          reinterpret_cast<uint32_t*>(dst), load_u32_unaligned(src));
      dst += sizeof(uint32_t);
      src += sizeof(uint32_t);
      bytes -= sizeof(uint32_t);
    } else if (bytes >= sizeof(uint16_t) && (dstAddr & 0x1) == 0) {
      multimem_store_u16(
          reinterpret_cast<uint16_t*>(dst), load_u16_unaligned(src));
      dst += sizeof(uint16_t);
      src += sizeof(uint16_t);
      bytes -= sizeof(uint16_t);
    } else {
      multimem_store_u8(
          reinterpret_cast<uint8_t*>(dst), load_u8_unaligned(src));
      ++dst;
      ++src;
      --bytes;
    }
  }
}

__device__ __forceinline__ void strided_multimem_store_unaligned(
    comms::prims::ThreadGroup& group,
    char* dst,
    const char* src,
    std::size_t bytes) {
  constexpr std::size_t kChunkBytes = sizeof(uint4);
  for (std::size_t offset =
           static_cast<std::size_t>(group.thread_id_in_group) * kChunkBytes;
       offset < bytes;
       offset += static_cast<std::size_t>(group.group_size) * kChunkBytes) {
    const std::size_t remaining = bytes - offset;
    multimem_store_bytes(
        dst + offset,
        src + offset,
        remaining < kChunkBytes ? remaining : kChunkBytes);
  }
}

} // namespace comms::prims::detail

namespace comms::prims::multimem {

/**
 * Bulk broadcast into a multimem VA. When both `src` and `dst` are 16-byte
 * aligned, the bulk goes through the unrolled vectorized v4.f32 fast path
 * (`detail::strided_multimem_store_aligned<kUnroll>`) plus a tail.
 *
 * Any other alignment (including 16B-aligned `dst` with unaligned `src`) uses
 * `detail::strided_multimem_store_unaligned`, which still issues a v4
 * `multimem.st` per 16-byte chunk whenever `dst` is 16B-aligned -- gathering
 * the unaligned `src` via `load_v4_u32_unaligned` -- and only steps down to
 * 8/4/2/1-byte multimem stores for sub-16B `dst` alignment or the final tail.
 * It forgoes the fully-aligned path's `kUnroll` batching; a dedicated unrolled
 * aligned-dst/unaligned-src path is intentionally omitted, since the staging
 * buffers are 16B-aligned on both ends so that shape is not the hot path and
 * the extra specialization is not worth the complexity.
 *
 * `dst` must point into a multimem VA (typically
 * `transport.multimem_data_ptr(offset)`); `src` is a local buffer. All ranks
 * in the NVL team must call with the same `bytes`.
 */
template <int kUnroll>
__device__ __forceinline__ void store(
    comms::prims::ThreadGroup& group,
    char* dst,
    const void* src,
    std::size_t bytes) {
  static_assert(kUnroll > 0);
  const auto srcAddr = reinterpret_cast<uintptr_t>(src);
  const auto dstAddr = reinterpret_cast<uintptr_t>(dst);
  if (((srcAddr | dstAddr) & 0xF) == 0) {
    const std::size_t alignedBytes = bytes & ~static_cast<std::size_t>(0xF);
    auto* srcVec = reinterpret_cast<const uint4*>(src);
    auto* dstVec = reinterpret_cast<uint4*>(dst);
    const std::size_t vecCount = alignedBytes / sizeof(uint4);
    comms::prims::detail::strided_multimem_store_aligned<kUnroll>(
        group, dstVec, srcVec, vecCount);
    if (alignedBytes != bytes) {
      comms::prims::detail::strided_multimem_store_unaligned(
          group,
          dst + alignedBytes,
          static_cast<const char*>(src) + alignedBytes,
          bytes - alignedBytes);
    }
  } else {
    comms::prims::detail::strided_multimem_store_unaligned(
        group, dst, static_cast<const char*>(src), bytes);
  }
}

} // namespace comms::prims::multimem

#endif // ENABLE_PRIMS
