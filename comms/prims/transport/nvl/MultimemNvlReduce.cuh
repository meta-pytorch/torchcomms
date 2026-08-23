// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// multimem.ld_reduce load-reduce primitives for single-NVL-domain collectives.
//
// Holds the reduce side of the NVLS staging model: the raw `multimem.ld_reduce`
// PTX emitters (`comms::prims::detail`) and the public entry point
// `multimem::load_reduce_at<>` that reads the cross-rank reduction of a
// multicast VA into a local buffer. The store side (`multimem::store()`) lives
// in `MultimemNvlStore.cuh`.

// clang-tidy analyzes this .cuh as a standalone main file and misflags the
// pragma; it is a genuine include-once header. False positive, so suppress it.
// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#if defined(ENABLE_PRIMS)

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/transport/nvl/MultimemNvlTransportDevice.cuh"

// PTX helpers extend the transport header's existing `comms::prims::detail`
// namespace. The public free-function entry point (`load_reduce_at<>`) lives in
// `comms::prims::multimem` further below and delegates into `detail::` for the
// raw PTX.
namespace comms::prims::detail {

// ----------------------------------------------------------------------------
// multimem.ld_reduce load-reduce helpers (relaxed.sys, data-path).
// ----------------------------------------------------------------------------

// Typed multimem load-reduce ops (relaxed.sys). These are the `.ld_reduce`
// data peers of the `.st` store helpers alongside the signal-path emitters
// (`multimem_store_release_sys_u64` / `multimem_red_release_sys_add_u64` in
// MultimemNvlTransportDevice.cuh); those stay release.sys because they carry
// the producer/consumer handshake, while these are relaxed.sys since for bulk
// data the ordering is established by the signal that follows the data op.
// Only the add reduction is implemented today; widen with a template/op
// parameter later.

__device__ __forceinline__ float multimem_ld_reduce_f32(const float* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  float v;
  asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
               : "=f"(v)
               : "l"(__cvta_generic_to_global(src))
               : "memory");
  return v;
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  return {};
}

// Integer add reduction. NVLS has no 128-bit (v4) integer ld_reduce, so this is
// scalar-only (one .add.s32 per element) -- there is no v4 int fast path.
__device__ __forceinline__ int32_t multimem_ld_reduce_s32(const int32_t* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  int32_t v;
  asm volatile("multimem.ld_reduce.relaxed.sys.global.add.s32 %0, [%1];"
               : "=r"(v)
               : "l"(__cvta_generic_to_global(src))
               : "memory");
  return v;
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  return {};
}

__device__ __forceinline__ float4 multimem_ld_reduce_v4_f32(const float4* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  float4 v;
  asm volatile(
      "multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
      : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
      : "l"(__cvta_generic_to_global(src))
      : "memory");
  return v;
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  return {};
}

// 128-bit (v4) half reduction: one multimem.ld_reduce returns the cross-rank
// sum of 8 halves (4 packed f16x2). 16 B/op vs the 4 B/op of the scalar-pair
// f16x2 path, matching the float v4.f32 width. kAccF32 selects f32 accumulation
// (`.acc::f32`, the default) vs native 2-byte accumulation.
template <bool kAccF32 = true>
__device__ __forceinline__ uint4 multimem_ld_reduce_v4_f16x2(const uint4* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12020
  uint4 v;
  if constexpr (kAccF32) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.f16x2 "
        "{%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(__cvta_generic_to_global(src))
        : "memory");
  } else {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.f16x2 "
        "{%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(__cvta_generic_to_global(src))
        : "memory");
  }
  return v;
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  return {};
}

// Reduce-load a single half: `multimem.ld_reduce.f16x2` needs a 4-byte aligned
// address, so read the aligned pair containing `src` and pick the requested
// lane. Safe for any alignment of `src`.
template <bool kAccF32 = true>
__device__ __forceinline__ __half multimem_ld_reduce_f16(const __half* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12020
  const uintptr_t genericAddr = reinterpret_cast<uintptr_t>(src);
  const uintptr_t globalAddr =
      static_cast<uintptr_t>(__cvta_generic_to_global(src));
  const uintptr_t alignedGlobalAddr = globalAddr & ~uintptr_t{0x3};
  uint32_t raw;
  if constexpr (kAccF32) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.f16x2 %0, [%1];"
        : "=r"(raw)
        : "l"(alignedGlobalAddr)
        : "memory");
  } else {
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f16x2 %0, [%1];"
                 : "=r"(raw)
                 : "l"(alignedGlobalAddr)
                 : "memory");
  }
  union {
    uint32_t r;
    __half h[2];
  } u{raw};
  return u.h[(genericAddr / sizeof(__half)) & 0x1];
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  return {};
}

// bf16 mirrors the f16 emitters above: same widths and .acc::f32 accumulation,
// with the `.bf16x2` packed-type variant of each multimem verb. bf16 is the
// dominant ML training dtype.

// 128-bit (v4) bf16 reduction: one multimem.ld_reduce returns the cross-rank
// sum of 8 bf16 (4 packed bf16x2) with f32 accumulation. 16 B/op, matching
// v4.f16x2.
template <bool kAccF32 = true>
__device__ __forceinline__ uint4
multimem_ld_reduce_v4_bf16x2(const uint4* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12020
  uint4 v;
  if constexpr (kAccF32) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
        "{%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(__cvta_generic_to_global(src))
        : "memory");
  } else {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 "
        "{%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(__cvta_generic_to_global(src))
        : "memory");
  }
  return v;
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  return {};
}

// Reduce-load a single bf16: `multimem.ld_reduce.bf16x2` needs a 4-byte aligned
// address, so read the aligned pair containing `src` and pick the requested
// lane. Safe for any alignment of `src`.
template <bool kAccF32 = true>
__device__ __forceinline__ __nv_bfloat16
multimem_ld_reduce_bf16(const __nv_bfloat16* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12020
  const uintptr_t genericAddr = reinterpret_cast<uintptr_t>(src);
  const uintptr_t globalAddr =
      static_cast<uintptr_t>(__cvta_generic_to_global(src));
  const uintptr_t alignedGlobalAddr = globalAddr & ~uintptr_t{0x3};
  uint32_t raw;
  if constexpr (kAccF32) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.bf16x2 %0, [%1];"
        : "=r"(raw)
        : "l"(alignedGlobalAddr)
        : "memory");
  } else {
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.bf16x2 %0, [%1];"
                 : "=r"(raw)
                 : "l"(alignedGlobalAddr)
                 : "memory");
  }
  union {
    uint32_t r;
    __nv_bfloat16 b[2];
  } u{raw};
  return u.b[(genericAddr / sizeof(__nv_bfloat16)) & 0x1];
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  return {};
}

} // namespace comms::prims::detail

namespace comms::prims::multimem {

/** Reduction operator for the multimem data reduce verbs. Only Add today. */
enum class MultimemRedOp { Add };

/**
 * Reduces one aligned 16-byte fp16 or bf16 block into registers.
 *
 * Keeping the reduced block separate from the eventual stores lets callers
 * place a team barrier between the read and write phases when adjacent ranks
 * own different lanes of the same 16-byte block.
 */
template <typename T, bool kAccF32 = true>
__device__ __forceinline__ uint4 load_reduce_block16(const T* multicastBlock) {
  static_assert(
      std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>,
      "load_reduce_block16 supports fp16 and bf16");
  const auto* source = reinterpret_cast<const uint4*>(multicastBlock);
  if constexpr (std::is_same_v<T, __half>) {
    return comms::prims::detail::multimem_ld_reduce_v4_f16x2<kAccF32>(source);
  } else {
    return comms::prims::detail::multimem_ld_reduce_v4_bf16x2<kAccF32>(source);
  }
}

/** Stores a contiguous lane range from a previously reduced 16-byte block. */
template <typename T>
__device__ __forceinline__ void store_reduced_block16_range(
    T* destination,
    const uint4& block,
    std::size_t firstLane,
    std::size_t count) {
  static_assert(
      sizeof(T) == 2, "store_reduced_block16_range requires a 2-byte type");
  const auto* lanes = reinterpret_cast<const T*>(&block);
  for (std::size_t index = 0; index < count; ++index) {
    destination[index] = lanes[firstLane + index];
  }
}

/**
 * multimem.ld_reduce from an ARBITRARY multicast base pointer into `dst`.
 *
 * `mc` must point into a multicast VA; `dst` is local. Staging callers combine
 * this with `transport.multimem_data_ptr(offset)`. Uses typed dispatch and a
 * 16-byte fast path when the pointers are jointly aligned.
 */
template <
    typename T,
    MultimemRedOp Op = MultimemRedOp::Add,
    bool kAccF32 = true>
__device__ __forceinline__ void load_reduce_at(
    comms::prims::ThreadGroup& group,
    T* dst,
    const T* mc,
    std::size_t elems) {
  static_assert(
      Op == MultimemRedOp::Add,
      "multimem load_reduce_at: only Add implemented");
  const std::size_t stride = group.group_size;
  const std::size_t t0 = group.thread_id_in_group;
  const uintptr_t addrOr =
      reinterpret_cast<uintptr_t>(mc) | reinterpret_cast<uintptr_t>(dst);
  auto vec_loop = [&](auto* dstVec, std::size_t vecCount, auto load_pack) {
    for (std::size_t i = t0; i < vecCount; i += stride) {
      dstVec[i] = load_pack(i);
    }
  };

  if constexpr (std::is_same_v<T, float>) {
    if ((addrOr & 0xF) == 0) {
      const std::size_t vec = elems / 4;
      const auto* mcVec = reinterpret_cast<const float4*>(mc);
      vec_loop(reinterpret_cast<float4*>(dst), vec, [&](std::size_t k) {
        return comms::prims::detail::multimem_ld_reduce_v4_f32(mcVec + k);
      });
      for (std::size_t j = vec * 4 + t0; j < elems; j += stride) {
        dst[j] = comms::prims::detail::multimem_ld_reduce_f32(mc + j);
      }
    } else {
      for (std::size_t i = t0; i < elems; i += stride) {
        dst[i] = comms::prims::detail::multimem_ld_reduce_f32(mc + i);
      }
    }
  } else if constexpr (std::is_same_v<T, __half>) {
    if ((addrOr & 0xF) == 0) {
      const std::size_t vec = elems / 8;
      const auto* mcVec = reinterpret_cast<const uint4*>(mc);
      vec_loop(reinterpret_cast<uint4*>(dst), vec, [&](std::size_t k) {
        return comms::prims::detail::multimem_ld_reduce_v4_f16x2<kAccF32>(
            mcVec + k);
      });
      for (std::size_t j = vec * 8 + t0; j < elems; j += stride) {
        dst[j] = comms::prims::detail::multimem_ld_reduce_f16<kAccF32>(mc + j);
      }
    } else {
      for (std::size_t i = t0; i < elems; i += stride) {
        dst[i] = comms::prims::detail::multimem_ld_reduce_f16<kAccF32>(mc + i);
      }
    }
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if ((addrOr & 0xF) == 0) {
      const std::size_t vec = elems / 8;
      const auto* mcVec = reinterpret_cast<const uint4*>(mc);
      vec_loop(reinterpret_cast<uint4*>(dst), vec, [&](std::size_t k) {
        return comms::prims::detail::multimem_ld_reduce_v4_bf16x2<kAccF32>(
            mcVec + k);
      });
      for (std::size_t j = vec * 8 + t0; j < elems; j += stride) {
        dst[j] = comms::prims::detail::multimem_ld_reduce_bf16<kAccF32>(mc + j);
      }
    } else {
      for (std::size_t i = t0; i < elems; i += stride) {
        dst[i] = comms::prims::detail::multimem_ld_reduce_bf16<kAccF32>(mc + i);
      }
    }
  } else if constexpr (std::is_same_v<T, int32_t>) {
    for (std::size_t i = t0; i < elems; i += stride) {
      dst[i] = comms::prims::detail::multimem_ld_reduce_s32(mc + i);
    }
  } else {
    static_assert(
        sizeof(T) == 0, "multimem load_reduce_at: unsupported element type");
  }
}

} // namespace comms::prims::multimem

#endif // ENABLE_PRIMS
