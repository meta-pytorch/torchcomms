// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// multimem.ld_reduce load-reduce primitives for single-NVL-domain collectives.
//
// Holds the reduce side of the NVLS staging model: the raw `multimem.ld_reduce`
// PTX emitters (`comms::prims::detail`) and the public entry point
// `multimem::load_reduce_at<>` that reads the cross-rank reduction of a
// multicast VA into a local buffer. The store side (`multimem::store()`) and
// the staging orchestration that composes both land later in the stack
// (`MultimemNvlStore.cuh`, `MultimemNvlStaging.cuh`).

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
#else
  return *src;
#endif
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
#else
  return *src;
#endif
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
#else
  return *src;
#endif
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
#else
  return *src;
#endif
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
#else
  return *src;
#endif
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
#else
  return *src;
#endif
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
#else
  return *src;
#endif
}

// Proxy fence between a multicast-alias read (multimem.ld_reduce) and a
// unicast-alias store to the same physical memory (in-place no-copy reduce).
// PTX ISA 8.6 requires a proxy fence to synchronize memory operations across
// different proxies. No-op when kEnabled is false (out-of-place: the read and
// store target distinct allocations, so there is no cross-proxy aliasing).
template <bool kEnabled>
__device__ __forceinline__ void proxy_alias_fence() {
#if defined(__CUDA_ARCH__)
  if constexpr (kEnabled) {
    asm volatile("fence.proxy.alias;" ::: "memory");
  }
#endif
}

} // namespace comms::prims::detail

namespace comms::prims::multimem {

// Legacy alias for callers that historically reached the PTX helpers via
// `comms::prims::detail`.
namespace nocopy_detail = comms::prims::detail;

// multimem.ld_reduce unroll depth on the bandwidth-critical reduce read: issue
// this many switch-reduce loads into registers before storing, to overlap their
// latencies (memory-level parallelism). It is the single knob for RS, direct
// AR, and composed AR reads. 16 (not the NVLS symmetric kernels' 8) is the
// measured sweet spot on GB300: a single CTA is memory-level-parallelism bound
// at large sizes (one 640-thread CTA saturates the switch only at ~4 CTAs), and
// doubling the in-flight loads lifts the low-CTA regime sharply -- 4GB no-copy
// AllReduce goes 119->201 GB/s at 1 CTA and 230->390 GB/s at 2 CTAs, while the
// multi-CTA plateau is unchanged (~458 GB/s at 4-8 CTAs). 32 regresses
// everything: uint4 tmp[32] = 128 registers/thread spills for a 640-thread CTA.
//
// Passed as the `kUnroll` template argument of load_reduce_at (below).
constexpr int kReduceUnroll = 16;

/**
 * multimem.ld_reduce from an ARBITRARY multicast base pointer into `dst`.
 *
 * `mc` must point into a multicast VA; `dst` is local. Callers that reduce
 * relative to a transport's staging window should combine this with
 * `transport.multimem_data_ptr(offset)`; callers that reduce the user's
 * registered multicast send VA (a distinct multicast object) pass that
 * pointer directly. Typed dispatch and 16-byte (v4) fast path when the
 * pointers are jointly 16-byte aligned.
 */
template <
    typename T,
    comms::prims::MultimemRedOp Op = comms::prims::MultimemRedOp::Add,
    // Unroll depth for the multimem-load reduce loop (see kReduceUnroll): batch
    // this many switch-reduce loads into registers before storing, for
    // memory-level parallelism.
    int kUnroll = 1,
    bool kAliased = false,
    bool kAccF32 = true>
__device__ __forceinline__ void load_reduce_at(
    comms::prims::ThreadGroup& group,
    T* dst,
    const T* mc,
    std::size_t elems) {
  static_assert(
      Op == comms::prims::MultimemRedOp::Add,
      "multimem load_reduce_at: only Add implemented");
  static_assert(kUnroll > 0);
  const std::size_t stride = group.group_size;
  const std::size_t t0 = group.thread_id_in_group;
  const uintptr_t addrOr =
      reinterpret_cast<uintptr_t>(mc) | reinterpret_cast<uintptr_t>(dst);
  // When kAliased (in-place no-copy: dst aliases the multicast-read physical),
  // the multicast read and unicast store cross proxies -> emit the PTX 8.6
  // proxy fence between this thread's reads and its stores (no-op otherwise).

  // kUnroll-wide grid-stride over 16B packs: issue kUnroll multimem.ld_reduce
  // loads into registers before storing any, so the per-op switch-reduce
  // latencies overlap (memory-level parallelism). Matches transport.load_reduce
  // and reduce_broadcast_at; without it this path is latency-bound (it was the
  // ~2-3x no-copy ReduceScatter slowdown vs AllReduce). dstVec/load_pack carry
  // the typed 16B pack; the scalar tail stays per-element.
  auto vec_loop = [&](auto* dstVec, std::size_t vecCount, auto load_pack) {
    std::size_t i = t0;
    for (; i + (kUnroll - 1) * stride < vecCount; i += kUnroll * stride) {
      decltype(load_pack(i)) tmp[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        tmp[u] = load_pack(i + static_cast<std::size_t>(u) * stride);
      }
      nocopy_detail::proxy_alias_fence<kAliased>();
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        dstVec[i + static_cast<std::size_t>(u) * stride] = tmp[u];
      }
    }
    // Batched remainder: issue this thread's last (< kUnroll) packs all into
    // registers before storing any, so a tile whose whole shard is below the
    // kUnroll*stride threshold that engages the loop above still gets memory-
    // level parallelism instead of collapsing to one in-flight switch-reduce
    // per thread (the serialized single-issue tail was the ~128KiB-shard
    // ReduceScatter/AllReduce small-message latency hump). The kAliased proxy
    // fence between the batched load and store is preserved for in-place
    // correctness (a no-op out-of-place).
    {
      decltype(load_pack(i)) tmp[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const std::size_t idx = i + static_cast<std::size_t>(u) * stride;
        if (idx < vecCount) {
          tmp[u] = load_pack(idx);
        }
      }
      nocopy_detail::proxy_alias_fence<kAliased>();
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const std::size_t idx = i + static_cast<std::size_t>(u) * stride;
        if (idx < vecCount) {
          dstVec[idx] = tmp[u];
        }
      }
    }
  };

  if constexpr (std::is_same_v<T, float>) {
    if ((addrOr & 0xF) == 0) {
      const std::size_t vec = elems / 4;
      const auto* mcVec = reinterpret_cast<const float4*>(mc);
      vec_loop(reinterpret_cast<float4*>(dst), vec, [&](std::size_t k) {
        return nocopy_detail::multimem_ld_reduce_v4_f32(mcVec + k);
      });
      for (std::size_t j = vec * 4 + t0; j < elems; j += stride) {
        const float v = nocopy_detail::multimem_ld_reduce_f32(mc + j);
        nocopy_detail::proxy_alias_fence<kAliased>();
        dst[j] = v;
      }
    } else {
      for (std::size_t i = t0; i < elems; i += stride) {
        const float v = nocopy_detail::multimem_ld_reduce_f32(mc + i);
        nocopy_detail::proxy_alias_fence<kAliased>();
        dst[i] = v;
      }
    }
  } else if constexpr (std::is_same_v<T, __half>) {
    if ((addrOr & 0xF) == 0) {
      const std::size_t vec = elems / 8;
      const auto* mcVec = reinterpret_cast<const uint4*>(mc);
      vec_loop(reinterpret_cast<uint4*>(dst), vec, [&](std::size_t k) {
        return nocopy_detail::multimem_ld_reduce_v4_f16x2<kAccF32>(mcVec + k);
      });
      for (std::size_t j = vec * 8 + t0; j < elems; j += stride) {
        const __half v = nocopy_detail::multimem_ld_reduce_f16<kAccF32>(mc + j);
        nocopy_detail::proxy_alias_fence<kAliased>();
        dst[j] = v;
      }
    } else {
      for (std::size_t i = t0; i < elems; i += stride) {
        const __half v = nocopy_detail::multimem_ld_reduce_f16<kAccF32>(mc + i);
        nocopy_detail::proxy_alias_fence<kAliased>();
        dst[i] = v;
      }
    }
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if ((addrOr & 0xF) == 0) {
      const std::size_t vec = elems / 8;
      const auto* mcVec = reinterpret_cast<const uint4*>(mc);
      vec_loop(reinterpret_cast<uint4*>(dst), vec, [&](std::size_t k) {
        return nocopy_detail::multimem_ld_reduce_v4_bf16x2<kAccF32>(mcVec + k);
      });
      for (std::size_t j = vec * 8 + t0; j < elems; j += stride) {
        const __nv_bfloat16 v =
            nocopy_detail::multimem_ld_reduce_bf16<kAccF32>(mc + j);
        nocopy_detail::proxy_alias_fence<kAliased>();
        dst[j] = v;
      }
    } else {
      for (std::size_t i = t0; i < elems; i += stride) {
        const __nv_bfloat16 v =
            nocopy_detail::multimem_ld_reduce_bf16<kAccF32>(mc + i);
        nocopy_detail::proxy_alias_fence<kAliased>();
        dst[i] = v;
      }
    }
  } else if constexpr (std::is_same_v<T, int32_t>) {
    // No 128-bit integer multimem.ld_reduce -> scalar, but still unroll the
    // issue for memory-level parallelism.
    std::size_t i = t0;
    for (; i + (kUnroll - 1) * stride < elems; i += kUnroll * stride) {
      int32_t tmp[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        tmp[u] = nocopy_detail::multimem_ld_reduce_s32(
            mc + (i + static_cast<std::size_t>(u) * stride));
      }
      nocopy_detail::proxy_alias_fence<kAliased>();
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        dst[i + static_cast<std::size_t>(u) * stride] = tmp[u];
      }
    }
    // Batched remainder (see vec_loop): keep MLP for small tiles; the kAliased
    // proxy fence is preserved for in-place correctness.
    {
      int32_t tmp[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const std::size_t idx = i + static_cast<std::size_t>(u) * stride;
        if (idx < elems) {
          tmp[u] = nocopy_detail::multimem_ld_reduce_s32(mc + idx);
        }
      }
      nocopy_detail::proxy_alias_fence<kAliased>();
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const std::size_t idx = i + static_cast<std::size_t>(u) * stride;
        if (idx < elems) {
          dst[idx] = tmp[u];
        }
      }
    }
  } else {
    static_assert(
        sizeof(T) == 0, "multimem load_reduce_at: unsupported element type");
  }
}

} // namespace comms::prims::multimem

#endif // ENABLE_PRIMS
