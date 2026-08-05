// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/utils/kernels/rng/philox_rng.cuh"
#include "comms/utils/kernels/stochastic_rounding/stochastic_rounding.cuh"

namespace comms::prims {

// Raw-PTX TMA / mbarrier wrappers. Deliberately not <cuda/ptx>: CCCL's
// `cuda/ptx` header is not on the fbcode CUDA include path for every target
// that pulls this header in, and the inline asm below is byte-identical to
// what `cuda::ptx::cp_async_bulk(space_shared, space_global, ...)` expands to.
// All of these are PTX ISA 8.6 / SM_90 family-agnostic; they assemble for the
// generic `sm_103` target that the b300 build uses (verified with
// `nvcc -arch=sm_103` + `cuobjdump -sass`: UBLKCP.S.G, SYNCS.ARRIVE.TRANS64,
// SYNCS.PHASECHK.TRANS64.TRYWAIT, FENCE.VIEW.ASYNC.{G,S}).
namespace quant_tma {

// Blackwell and newer only. On sm_90 the bulk global->shared destination must
// be .shared::cluster; ptxas rejects the .shared::cta form used here with
// "State space incorrect for instruction 'cp.async.bulk'". This must stay in
// step with the compute-capability gate in
// launch_direct_reduce_scatter_ib_quantized_impl.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#define COMMS_PRIMS_QUANT_TMA_AVAILABLE 1
#else
#define COMMS_PRIMS_QUANT_TMA_AVAILABLE 0
#endif

#if COMMS_PRIMS_QUANT_TMA_AVAILABLE

__device__ __forceinline__ std::uint32_t smem_addr(const void* p) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarrier_init(
    std::uint64_t* bar,
    std::uint32_t arrive_count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(smem_addr(bar)), "r"(arrive_count)
               : "memory");
}

__device__ __forceinline__ void mbarrier_inval(std::uint64_t* bar) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
               :
               : "r"(smem_addr(bar))
               : "memory");
}

// Orders this thread's prior generic-proxy shared accesses ahead of async-proxy
// (TMA) shared writes. Its load-bearing use is stage reuse: every consumer
// publishes its reads of a stage before the group barrier that releases that
// stage to the next tile's TMA. (It is NOT required after mbarrier.init:
// cp.async.bulk reaches its mbarrier operand through the generic proxy.)
__device__ __forceinline__ void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Orders this thread's prior generic-proxy global accesses (the DATA_READY flag
// acquire that the transport leader performed) ahead of async-proxy global
// reads issued by the same thread.
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

// cp.async.bulk.shared::cta.global — 1-D descriptorless TMA. Requires
// 16B-aligned src, 16B-aligned dst and a size that is a multiple of 16.
__device__ __forceinline__ void bulk_load(
    void* dst_smem,
    const void* src_gmem,
    std::uint32_t nbytes,
    std::uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];"
      :
      : "r"(smem_addr(dst_smem)),
        "l"(src_gmem),
        "r"(nbytes),
        "r"(smem_addr(bar))
      : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    std::uint64_t* bar,
    std::uint32_t tx_bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
      :
      : "r"(smem_addr(bar)), "r"(tx_bytes)
      : "memory");
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(
    std::uint64_t* bar,
    std::uint32_t parity) {
  std::uint32_t done;
  asm volatile(
      "{\n\t.reg .pred P;\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n\t"
      "selp.b32 %0, 1, 0, P;\n\t}"
      : "=r"(done)
      : "r"(smem_addr(bar)), "r"(parity)
      : "memory");
  return done != 0u;
}

#endif // COMMS_PRIMS_QUANT_TMA_AVAILABLE

} // namespace quant_tma

template <bool kUseTma>
struct QuantizedReduceScatterCopyOpT {
  // Whether the receive path stages its operands through shared memory with
  // bulk TMA. Selected at launch through MCCL_PRIMS_TMA.
  //
  // This must NOT depend on COMMS_PRIMS_QUANT_TMA_AVAILABLE: that macro keys
  // off
  // __CUDA_ARCH__, which is undefined on the host pass, so folding it in here
  // would make the launcher reserve zero dynamic shared memory for a kernel
  // whose device code uses it. __HIP_PLATFORM_AMD__ is safe to fold in because
  // it is defined identically on both passes, and ROCm has neither TMA nor
  // enough LDS (64 KB/workgroup) for the staging region.
#if defined(__HIP_PLATFORM_AMD__)
  static constexpr bool kTmaRecv = false;
#else
  static constexpr bool kTmaRecv = kUseTma;
#endif

  // ---------------------------------------------------------------------
  // Dynamic shared memory. The send and receive warp groups are inlined into
  // the SAME CTA and run concurrently, so the region is statically partitioned
  // and each group only ever touches its own half. This variant gives the send
  // group nothing.
  //
  //   [0, kRecvHeaderBytes)                mbarrier array + 128B align slack
  //   [.. + s*kRecvStageBytes ..)          stage s: bf16 wire tile | fp32 local
  // ---------------------------------------------------------------------
  static constexpr std::size_t kSendSmemBytes = 0;
  static constexpr std::uint32_t kRecvStages = 2;
  static constexpr std::uint32_t kRecvTileElems = 16384;
  // Independent float4 outputs each recv thread keeps in flight while draining
  // a staged tile. Covers LDS latency now that the loads are off the LSU.
  static constexpr std::uint32_t kRecvSmemUnroll = 8;

  static constexpr std::size_t kWireTileBytes =
      static_cast<std::size_t>(kRecvTileElems) * sizeof(__nv_bfloat16);
  static constexpr std::size_t kLocalTileBytes =
      static_cast<std::size_t>(kRecvTileElems) * sizeof(float);
  static constexpr std::size_t kRecvStageBytes =
      kWireTileBytes + kLocalTileBytes;
  static constexpr std::size_t kRecvHeaderBytes = 256;
  static constexpr std::size_t kRecvSmemBytes =
      kRecvHeaderBytes + kRecvStages * kRecvStageBytes;
  __host__ __device__ static constexpr std::size_t smem_bytes() {
    return kTmaRecv ? (kSendSmemBytes + kRecvSmemBytes) : 0;
  }

  static_assert(
      kRecvTileElems % 8u == 0u,
      "tile must keep both the bf16 (2B) and fp32 (4B) sub-tiles a multiple of "
      "16 bytes for cp.async.bulk");
  static_assert(
      kRecvStageBytes % 128u == 0u,
      "stage stride must preserve 128B alignment of every tile");
  static_assert(
      kRecvStageBytes < (1u << 20),
      "per-stage expect_tx must stay inside the mbarrier tx-count range");

  struct Args {
    const float* sender_input_base;
    const float* receiver_input_base;
    float* receiver_output_base;
    std::uint64_t seed;
    std::uint64_t logical_element_base;
  };

  static constexpr bool kVariableSize = false;
  static constexpr std::size_t kActivationThreshold = 0;

  __host__ __device__ __forceinline__ static constexpr std::size_t
  worst_case_chunk_stride(std::size_t chunkSize) {
    return chunkSize;
  }

  // Number of independent 8-element packets a send thread keeps in flight.
  // The packets are strided by the group size, so none of them depends on
  // another's loads.
  static constexpr std::size_t kSendUnrollNC = 2;

#if defined(__CUDA_ARCH__)
  struct SendPacket {
    float4 lo;
    float4 hi;
  };

  struct SendQuantized {
    __nv_bfloat162 r0;
    __nv_bfloat162 r1;
    __nv_bfloat162 r2;
    __nv_bfloat162 r3;
  };

  __device__ __forceinline__ static SendPacket load_send_packet(
      const float* src,
      bool vector_aligned) {
    SendPacket loaded;
    loaded.lo = vector_aligned ? reinterpret_cast<const float4*>(src)[0]
                               : make_float4(src[0], src[1], src[2], src[3]);
    loaded.hi = vector_aligned ? reinterpret_cast<const float4*>(src + 4)[0]
                               : make_float4(src[4], src[5], src[6], src[7]);
    return loaded;
  }

  __device__ __forceinline__ static SendQuantized quantize_send_packet(
      const SendPacket& loaded,
      const PhiloxResult& random) {
    SendQuantized quantized;
    quantized.r0 = stochastic_round_bf16x2<kHasHardwareSR>(
        make_float2(loaded.lo.x, loaded.lo.y), random.u32[0]);
    quantized.r1 = stochastic_round_bf16x2<kHasHardwareSR>(
        make_float2(loaded.lo.z, loaded.lo.w), random.u32[1]);
    quantized.r2 = stochastic_round_bf16x2<kHasHardwareSR>(
        make_float2(loaded.hi.x, loaded.hi.y), random.u32[2]);
    quantized.r3 = stochastic_round_bf16x2<kHasHardwareSR>(
        make_float2(loaded.hi.z, loaded.hi.w), random.u32[3]);
    return quantized;
  }

  __device__ __forceinline__ static void store_send_packet(
      __nv_bfloat16* dst,
      const SendQuantized& quantized,
      bool vector_aligned,
      bool quad_aligned) {
    if (quad_aligned) {
      reinterpret_cast<uint4*>(dst)[0] = make_uint4(
          *reinterpret_cast<const std::uint32_t*>(&quantized.r0),
          *reinterpret_cast<const std::uint32_t*>(&quantized.r1),
          *reinterpret_cast<const std::uint32_t*>(&quantized.r2),
          *reinterpret_cast<const std::uint32_t*>(&quantized.r3));
    } else if (vector_aligned) {
      reinterpret_cast<uint2*>(dst)[0] = make_uint2(
          *reinterpret_cast<const std::uint32_t*>(&quantized.r0),
          *reinterpret_cast<const std::uint32_t*>(&quantized.r1));
      reinterpret_cast<uint2*>(dst + 4)[0] = make_uint2(
          *reinterpret_cast<const std::uint32_t*>(&quantized.r2),
          *reinterpret_cast<const std::uint32_t*>(&quantized.r3));
    } else {
      dst[0] = __low2bfloat16(quantized.r0);
      dst[1] = __high2bfloat16(quantized.r0);
      dst[2] = __low2bfloat16(quantized.r1);
      dst[3] = __high2bfloat16(quantized.r1);
      dst[4] = __low2bfloat16(quantized.r2);
      dst[5] = __high2bfloat16(quantized.r2);
      dst[6] = __low2bfloat16(quantized.r3);
      dst[7] = __high2bfloat16(quantized.r3);
    }
  }
#endif

  __device__ __forceinline__ static std::size_t send(
      char* staging,
      const char* /*src*/,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t byte_offset,
      Args args) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    PIPES_DEVICE_CHECK(byte_offset % sizeof(__nv_bfloat16) == 0);
    PIPES_DEVICE_CHECK(nbytes % sizeof(__nv_bfloat16) == 0);
    auto* stagingBf16 = reinterpret_cast<__nv_bfloat16*>(staging);
    const std::size_t elementOffset = byte_offset / sizeof(__nv_bfloat16);
    const std::size_t elementCount = nbytes / sizeof(__nv_bfloat16);
    const std::uint64_t logicalBegin =
        args.logical_element_base + elementOffset;
    const std::size_t elementsToAlignment =
        static_cast<std::size_t>((8 - logicalBegin % 8) % 8);
    const std::size_t prefixCount =
        elementsToAlignment < elementCount ? elementsToAlignment : elementCount;

    for (std::size_t i = group.thread_id_in_group; i < prefixCount;
         i += group.group_size) {
      const std::uint64_t logicalElement = logicalBegin + i;
      const PhiloxResult random =
          philox_randint4x(args.seed, logicalElement / 8);
      stagingBf16[i] = stochastic_round_bf16<kHasHardwareSR>(
          args.sender_input_base[elementOffset + i],
          random.u16[logicalElement % 8]);
    }

    const std::size_t packetCount = (elementCount - prefixCount) / 8;
    const bool vectorAligned =
        (reinterpret_cast<std::uintptr_t>(
             args.sender_input_base + elementOffset + prefixCount) %
         alignof(float4)) == 0 &&
        (reinterpret_cast<std::uintptr_t>(stagingBf16 + prefixCount) %
         alignof(uint2)) == 0;
    const bool sendQuadAligned =
        (reinterpret_cast<std::uintptr_t>(stagingBf16 + prefixCount) %
         alignof(uint4)) == 0;
#if defined(__CUDA_ARCH__)
    // A Philox result supplies the entropy for one complete 8-element packet.
    const std::size_t sendStride = group.group_size;
    std::size_t packet = group.thread_id_in_group;
    for (; packet + (kSendUnrollNC - 1) * sendStride < packetCount;
         packet += kSendUnrollNC * sendStride) {
      SendPacket loaded[kSendUnrollNC];
#pragma unroll
      for (std::size_t u = 0; u < kSendUnrollNC; ++u) {
        loaded[u] = load_send_packet(
            args.sender_input_base + elementOffset + prefixCount +
                (packet + u * sendStride) * 8,
            vectorAligned);
      }
      SendQuantized quantized[kSendUnrollNC];
#pragma unroll
      for (std::size_t u = 0; u < kSendUnrollNC; ++u) {
        quantized[u] = quantize_send_packet(
            loaded[u],
            philox_randint4x(
                args.seed,
                (logicalBegin + prefixCount + (packet + u * sendStride) * 8) /
                    8));
      }
#pragma unroll
      for (std::size_t u = 0; u < kSendUnrollNC; ++u) {
        store_send_packet(
            stagingBf16 + prefixCount + (packet + u * sendStride) * 8,
            quantized[u],
            vectorAligned,
            sendQuadAligned);
      }
    }
    for (; packet < packetCount; packet += sendStride) {
      const std::size_t base = prefixCount + packet * 8;
      const SendPacket loaded = load_send_packet(
          args.sender_input_base + elementOffset + base, vectorAligned);
      const SendQuantized quantized = quantize_send_packet(
          loaded, philox_randint4x(args.seed, (logicalBegin + base) / 8));
      store_send_packet(
          stagingBf16 + base, quantized, vectorAligned, sendQuadAligned);
    }
#else
    for (std::size_t packet = group.thread_id_in_group; packet < packetCount;
         packet += group.group_size) {
      const std::size_t i = prefixCount + packet * 8;
      const PhiloxResult random =
          philox_randint4x(args.seed, (logicalBegin + i) / 8);
#pragma unroll
      for (std::size_t pair = 0; pair < 4; ++pair) {
        const std::size_t pairIndex = i + pair * 2;
        const float2 values = make_float2(
            args.sender_input_base[elementOffset + pairIndex],
            args.sender_input_base[elementOffset + pairIndex + 1]);
        const __nv_bfloat162 rounded =
            stochastic_round_bf16x2<false>(values, random.u32[pair]);
        stagingBf16[pairIndex] = __low2bfloat16(rounded);
        stagingBf16[pairIndex + 1] = __high2bfloat16(rounded);
      }
    }
#endif

    const std::size_t tailBegin = prefixCount + packetCount * 8;
    for (std::size_t i = tailBegin + group.thread_id_in_group; i < elementCount;
         i += group.group_size) {
      const std::uint64_t logicalElement = logicalBegin + i;
      const PhiloxResult random =
          philox_randint4x(args.seed, logicalElement / 8);
      stagingBf16[i] = stochastic_round_bf16<kHasHardwareSR>(
          args.sender_input_base[elementOffset + i],
          random.u16[logicalElement % 8]);
    }
#endif
    return nbytes;
  }

  __device__ __forceinline__ static std::size_t recv(
      char* /*dst*/,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t byte_offset,
      Args args) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    PIPES_DEVICE_CHECK(byte_offset % sizeof(__nv_bfloat16) == 0);
    PIPES_DEVICE_CHECK(nbytes % sizeof(__nv_bfloat16) == 0);
    const auto* stagingBf16 = reinterpret_cast<const __nv_bfloat16*>(staging);
    const std::size_t elementOffset = byte_offset / sizeof(__nv_bfloat16);
    const std::size_t elementCount = nbytes / sizeof(__nv_bfloat16);

    const bool vectorAligned =
        (reinterpret_cast<std::uintptr_t>(stagingBf16) % alignof(uint2)) == 0 &&
        (reinterpret_cast<std::uintptr_t>(
             args.receiver_input_base + elementOffset) %
         alignof(float4)) == 0 &&
        (reinterpret_cast<std::uintptr_t>(
             args.receiver_output_base + elementOffset) %
         alignof(float4)) == 0;
    const std::uint32_t rpackCount =
        vectorAligned ? static_cast<std::uint32_t>(elementCount / 4) : 0u;
    const uint2* stagingPacks = reinterpret_cast<const uint2*>(stagingBf16);
    const uint4* stagingQuads = reinterpret_cast<const uint4*>(stagingBf16);
    const float4* localPacks = reinterpret_cast<const float4*>(
        args.receiver_input_base + elementOffset);
    float4* outPacks =
        reinterpret_cast<float4*>(args.receiver_output_base + elementOffset);
    const std::uint32_t rtid =
        static_cast<std::uint32_t>(group.thread_id_in_group);
    const std::uint32_t rstride = static_cast<std::uint32_t>(group.group_size);
    // cp.async.bulk needs 16B-aligned global src/dst and 16B-multiple sizes;
    // this is exactly the existing quad-aligned predicate.
    const bool quadAligned =
        (reinterpret_cast<std::uintptr_t>(stagingBf16) % alignof(uint4)) == 0 &&
        (reinterpret_cast<std::uintptr_t>(
             args.receiver_input_base + elementOffset) %
         alignof(uint4)) == 0 &&
        (reinterpret_cast<std::uintptr_t>(
             args.receiver_output_base + elementOffset) %
         alignof(uint4)) == 0;

#if COMMS_PRIMS_QUANT_TMA_AVAILABLE
    if constexpr (kTmaRecv) {
      // Every tile boundary lands on a multiple of 8 elements, which keeps the
      // bf16 sub-tile a multiple of 16B and the fp32 sub-tile a multiple of
      // 32B.
      const std::size_t alignedElems = (elementCount / 8u) * 8u;
      if (quadAligned && alignedElems >= kRecvTileElems) {
        extern __shared__ __align__(128) char quant_dyn_smem[];
        char* recvBase = quant_dyn_smem + kSendSmemBytes;
        auto* bar = reinterpret_cast<std::uint64_t*>(recvBase);
        char* tileBase = recvBase + kRecvStages * sizeof(std::uint64_t);
        // Dynamic smem is only guaranteed 16B-aligned; TMA destinations want
        // 128.
        tileBase +=
            (128u -
             (static_cast<std::uintptr_t>(__cvta_generic_to_shared(tileBase)) %
              128u)) %
            128u;

        // Only the recv group participates. An mbarrier's arrive-count is a
        // property of the object, not of the CTA, so a proper subset of threads
        // may own it: the leader is the sole arriver (count 1) and every other
        // recv thread is a pure waiter (try_wait is not an arrival).
        if (group.is_leader()) {
#pragma unroll
          for (std::uint32_t s = 0; s < kRecvStages; ++s) {
            quant_tma::mbarrier_init(&bar[s], 1u);
          }
          quant_tma::fence_proxy_async_shared();
        }
        group.sync(); // named barrier for THIS group; never __syncthreads()

        const std::uint32_t tiles = static_cast<std::uint32_t>(
            (alignedElems + kRecvTileElems - 1u) / kRecvTileElems);

        auto tile_elems = [&](std::uint32_t t) -> std::uint32_t {
          const std::size_t off = static_cast<std::size_t>(t) * kRecvTileElems;
          return static_cast<std::uint32_t>(
              (off + kRecvTileElems <= alignedElems) ? kRecvTileElems
                                                     : (alignedElems - off));
        };

        // Leader-only. The leader is also the thread that acquired DATA_READY
        // in the transport, so fence.proxy.async.global here orders that
        // generic acquire ahead of the async-proxy reads of the same bytes.
        auto issue = [&](std::uint32_t t) {
          if (!group.is_leader()) {
            return;
          }
          const std::size_t off = static_cast<std::size_t>(t) * kRecvTileElems;
          const std::uint32_t elems = tile_elems(t);
          const std::uint32_t wireBytes =
              elems * static_cast<std::uint32_t>(sizeof(__nv_bfloat16));
          const std::uint32_t localBytes =
              elems * static_cast<std::uint32_t>(sizeof(float));
          char* dst = tileBase + (t % kRecvStages) * kRecvStageBytes;
          // The sole arrival is fused into expect_tx, so the phase cannot flip
          // before the transaction count is registered; either order is
          // correct. Kept ahead of the copies to match CUTLASS and FA3.
          quant_tma::mbarrier_arrive_expect_tx(
              &bar[t % kRecvStages], wireBytes + localBytes);
          quant_tma::bulk_load(
              dst, stagingBf16 + off, wireBytes, &bar[t % kRecvStages]);
          quant_tma::bulk_load(
              dst + kWireTileBytes,
              args.receiver_input_base + elementOffset + off,
              localBytes,
              &bar[t % kRecvStages]);
        };

        // Orders the leader's generic-proxy DATA_READY acquire, performed once
        // in the transport before this call, ahead of every async-proxy read it
        // issues below. One fence per call is sufficient: no further
        // generic-proxy read of the staged bytes happens in between.
        if (group.is_leader()) {
          quant_tma::fence_proxy_async_global();
        }

        constexpr std::uint32_t kPrefetch = kRecvStages - 1u;
        for (std::uint32_t u = 0; u < kPrefetch && u < tiles; ++u) {
          issue(u);
        }
        for (std::uint32_t t = 0; t < tiles; ++t) {
          // Refills stage (t + kRecvStages - 1) % kRecvStages == (t-1) %
          // kRecvStages, which was last read in iteration t-1 and released by
          // that iteration's trailing group.sync().
          if (t + kPrefetch < tiles) {
            issue(t + kPrefetch);
          }
          const std::uint32_t s = t % kRecvStages;
          const std::uint32_t phase = (t / kRecvStages) & 1u;
          // Every other wait in this transport is bounded. Without this a stuck
          // TMA hangs the CTA with no rank/channel/tile diagnostic and the job
          // only dies to an external watchdog.
          std::uint64_t spins = 0;
          while (!quant_tma::mbarrier_try_wait_parity(&bar[s], phase)) {
            if ((++spins & 0xFFFFFFULL) == 0) {
              printf(
                  "[PIPES] FATAL: quant TMA recv stalled: tile=%u/%u stage=%u "
                  "parity=%u\n",
                  t,
                  tiles,
                  s,
                  phase);
              PIPES_DEVICE_TRAP();
            }
          }

          const char* src = tileBase + s * kRecvStageBytes;
          const uint2* wireS = reinterpret_cast<const uint2*>(src);
          const float4* localS =
              reinterpret_cast<const float4*>(src + kWireTileBytes);
          const std::size_t off = static_cast<std::size_t>(t) * kRecvTileElems;
          float4* outT = outPacks + (off / 4u);
          const std::uint32_t vecs = tile_elems(t) / 4u;

          std::uint32_t j = rtid;
          for (; j + (kRecvSmemUnroll - 1u) * rstride < vecs;
               j += kRecvSmemUnroll * rstride) {
            uint2 sv[kRecvSmemUnroll];
            float4 lv[kRecvSmemUnroll];
#pragma unroll
            for (std::uint32_t u = 0; u < kRecvSmemUnroll; ++u) {
              sv[u] = wireS[j + u * rstride];
              lv[u] = localS[j + u * rstride];
            }
#pragma unroll
            for (std::uint32_t u = 0; u < kRecvSmemUnroll; ++u) {
              const float2 lo = __bfloat1622float2(
                  *reinterpret_cast<const __nv_bfloat162*>(&sv[u].x));
              const float2 hi = __bfloat1622float2(
                  *reinterpret_cast<const __nv_bfloat162*>(&sv[u].y));
              outT[j + u * rstride] = make_float4(
                  lo.x + lv[u].x,
                  lo.y + lv[u].y,
                  hi.x + lv[u].z,
                  hi.y + lv[u].w);
            }
          }
          for (; j < vecs; j += rstride) {
            const uint2 s0 = wireS[j];
            const float4 l0 = localS[j];
            const float2 lo = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&s0.x));
            const float2 hi = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&s0.y));
            outT[j] =
                make_float4(lo.x + l0.x, lo.y + l0.y, hi.x + l0.z, hi.y + l0.w);
          }
          // Fence first (per-thread: publishes THIS thread's reads of stage s),
          // sync second (cross-thread edge). The reverse order does not order
          // the other threads' reads against the TMA that overwrites this stage
          // on tile t+kRecvStages, and is a real bug.
          quant_tma::fence_proxy_async_shared();
          group.sync(); // releases stage s for reuse by tile t+kRecvStages
        }

        if (group.is_leader()) {
#pragma unroll
          for (std::uint32_t s = 0; s < kRecvStages; ++s) {
            quant_tma::mbarrier_inval(&bar[s]);
          }
        }

        for (std::size_t i = alignedElems + group.thread_id_in_group;
             i < elementCount;
             i += group.group_size) {
          const std::size_t element = elementOffset + i;
          args.receiver_output_base[element] =
              __bfloat162float(stagingBf16[i]) +
              args.receiver_input_base[element];
        }
        // receiver_input_base aliases receiver_output_base from peer step 1 on,
        // so every thread must publish its output stores to the async proxy
        // before the next step's TMA reads them back as the accumulator. Once
        // per call: within a call, tile t's stores and tile t+1's TMA reads
        // cover disjoint bytes.
        quant_tma::fence_proxy_async_global();
        return nbytes;
      }
    }
#endif // COMMS_PRIMS_QUANT_TMA_AVAILABLE

    // ---- fallback: v9_s2_r5 register path (unaligned or sub-tile chunks) ----
    // Load both adjacent packs before either store because input may alias
    // output after the first peer step.
    const std::uint32_t pairCount = quadAligned ? rpackCount / 2u : 0u;
    constexpr std::uint32_t kRecvUnroll = 5;
    const std::uint32_t unrollSpan = rstride * kRecvUnroll;
    std::uint32_t pr = rtid;
    for (; kRecvUnroll > 1u && pr + (kRecvUnroll - 1u) * rstride < pairCount;
         pr += unrollSpan) {
      uint4 sq[kRecvUnroll];
      float4 la[kRecvUnroll];
      float4 lb[kRecvUnroll];
#pragma unroll
      for (std::uint32_t u = 0; u < kRecvUnroll; ++u) {
        const std::uint32_t q = pr + u * rstride;
        sq[u] = stagingQuads[q];
        la[u] = localPacks[q * 2u];
        lb[u] = localPacks[q * 2u + 1u];
      }
#pragma unroll
      for (std::uint32_t u = 0; u < kRecvUnroll; ++u) {
        const std::uint32_t q = pr + u * rstride;
        const float2 rL0 = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(&sq[u].x));
        const float2 rH0 = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(&sq[u].y));
        const float2 rL1 = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(&sq[u].z));
        const float2 rH1 = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(&sq[u].w));
        outPacks[q * 2u] = make_float4(
            rL0.x + la[u].x, rL0.y + la[u].y, rH0.x + la[u].z, rH0.y + la[u].w);
        outPacks[q * 2u + 1u] = make_float4(
            rL1.x + lb[u].x, rL1.y + lb[u].y, rH1.x + lb[u].z, rH1.y + lb[u].w);
      }
    }
    for (; pr < pairCount; pr += rstride) {
      const std::uint32_t p0 = pr * 2u;
      const std::uint32_t p1 = p0 + 1u;
      const uint4 sq1 = stagingQuads[pr];
      const float4 l0 = localPacks[p0];
      const float4 l1 = localPacks[p1];
      const float2 rL0 =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sq1.x));
      const float2 rH0 =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sq1.y));
      const float2 rL1 =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sq1.z));
      const float2 rH1 =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sq1.w));
      outPacks[p0] =
          make_float4(rL0.x + l0.x, rL0.y + l0.y, rH0.x + l0.z, rH0.y + l0.w);
      outPacks[p1] =
          make_float4(rL1.x + l1.x, rL1.y + l1.y, rH1.x + l1.z, rH1.y + l1.w);
    }
    for (std::uint32_t pack = pairCount * 2u + rtid; pack < rpackCount;
         pack += rstride) {
      const uint2 s0 = stagingPacks[pack];
      const float4 l0 = localPacks[pack];
      const float2 rL0 =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&s0.x));
      const float2 rH0 =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&s0.y));
      outPacks[pack] =
          make_float4(rL0.x + l0.x, rL0.y + l0.y, rH0.x + l0.z, rH0.y + l0.w);
    }
    const std::size_t packCount = rpackCount;

    const std::size_t scalarBegin = packCount * 4;
    for (std::size_t i = scalarBegin + group.thread_id_in_group;
         i < elementCount;
         i += group.group_size) {
      const std::size_t element = elementOffset + i;
      args.receiver_output_base[element] =
          __bfloat162float(stagingBf16[i]) + args.receiver_input_base[element];
    }
#if COMMS_PRIMS_QUANT_TMA_AVAILABLE
    // A chunk that fell back to the register path still writes the accumulator
    // that a later, TMA-eligible chunk reads back through the async proxy.
    if constexpr (kTmaRecv) {
      quant_tma::fence_proxy_async_global();
    }
#endif
#endif
    return nbytes;
  }
};

using QuantizedReduceScatterCopyOp = QuantizedReduceScatterCopyOpT<true>;
using QuantizedReduceScatterCopyOpNoTma = QuantizedReduceScatterCopyOpT<false>;

} // namespace comms::prims

// File-scoped by definition; do not leak it to translation units that include
// this header.
#undef COMMS_PRIMS_QUANT_TMA_AVAILABLE
