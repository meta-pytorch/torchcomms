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

struct QuantizedReduceScatterCopyOp {
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
    for (std::size_t packet = group.thread_id_in_group; packet < packetCount;
         packet += group.group_size) {
      const std::size_t base = prefixCount + packet * 8;
      const PhiloxResult random = philox_randint4x(
          args.seed, (logicalBegin + prefixCount + packet * 8) / 8);
      const float* src = args.sender_input_base + elementOffset + base;
      const float4 v0 = vectorAligned
          ? reinterpret_cast<const float4*>(src)[0]
          : make_float4(src[0], src[1], src[2], src[3]);
      const float4 v1 = vectorAligned
          ? reinterpret_cast<const float4*>(src + 4)[0]
          : make_float4(src[4], src[5], src[6], src[7]);
      const __nv_bfloat162 r0 = stochastic_round_bf16x2<kHasHardwareSR>(
          make_float2(v0.x, v0.y), random.u32[0]);
      const __nv_bfloat162 r1 = stochastic_round_bf16x2<kHasHardwareSR>(
          make_float2(v0.z, v0.w), random.u32[1]);
      const __nv_bfloat162 r2 = stochastic_round_bf16x2<kHasHardwareSR>(
          make_float2(v1.x, v1.y), random.u32[2]);
      const __nv_bfloat162 r3 = stochastic_round_bf16x2<kHasHardwareSR>(
          make_float2(v1.z, v1.w), random.u32[3]);
      if (sendQuadAligned) {
        reinterpret_cast<uint4*>(stagingBf16 + base)[0] = make_uint4(
            *reinterpret_cast<const std::uint32_t*>(&r0),
            *reinterpret_cast<const std::uint32_t*>(&r1),
            *reinterpret_cast<const std::uint32_t*>(&r2),
            *reinterpret_cast<const std::uint32_t*>(&r3));
      } else if (vectorAligned) {
        reinterpret_cast<uint2*>(stagingBf16 + base)[0] = make_uint2(
            *reinterpret_cast<const std::uint32_t*>(&r0),
            *reinterpret_cast<const std::uint32_t*>(&r1));
        reinterpret_cast<uint2*>(stagingBf16 + base + 4)[0] = make_uint2(
            *reinterpret_cast<const std::uint32_t*>(&r2),
            *reinterpret_cast<const std::uint32_t*>(&r3));
      } else {
        stagingBf16[base] = __low2bfloat16(r0);
        stagingBf16[base + 1] = __high2bfloat16(r0);
        stagingBf16[base + 2] = __low2bfloat16(r1);
        stagingBf16[base + 3] = __high2bfloat16(r1);
        stagingBf16[base + 4] = __low2bfloat16(r2);
        stagingBf16[base + 5] = __high2bfloat16(r2);
        stagingBf16[base + 6] = __low2bfloat16(r3);
        stagingBf16[base + 7] = __high2bfloat16(r3);
      }
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
    // Load both adjacent packs before either store because input may alias
    // output after the first peer step.
    const bool quadAligned =
        (reinterpret_cast<std::uintptr_t>(stagingBf16) % alignof(uint4)) == 0 &&
        (reinterpret_cast<std::uintptr_t>(
             args.receiver_input_base + elementOffset) %
         alignof(uint4)) == 0 &&
        (reinterpret_cast<std::uintptr_t>(
             args.receiver_output_base + elementOffset) %
         alignof(uint4)) == 0;
    const std::uint32_t pairCount = quadAligned ? rpackCount / 2u : 0u;
    for (std::uint32_t pr = rtid; pr < pairCount; pr += rstride) {
      const std::uint32_t p0 = pr * 2u;
      const std::uint32_t p1 = p0 + 1u;
      const uint4 sq = stagingQuads[pr];
      const float4 l0 = localPacks[p0];
      const float4 l1 = localPacks[p1];
      const float2 rL0 =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sq.x));
      const float2 rH0 =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sq.y));
      const float2 rL1 =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sq.z));
      const float2 rH1 =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sq.w));
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
#endif
    return nbytes;
  }
};

struct QuantizedCopyOp : QuantizedReduceScatterCopyOp {
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
             args.receiver_output_base + elementOffset) %
         alignof(float4)) == 0;
    const std::size_t packCount = vectorAligned ? elementCount / 4 : 0;
    const auto* stagingPacks = reinterpret_cast<const uint2*>(stagingBf16);
    auto* outputPacks =
        reinterpret_cast<float4*>(args.receiver_output_base + elementOffset);
    for (std::size_t pack = group.thread_id_in_group; pack < packCount;
         pack += group.group_size) {
      const uint2 values = stagingPacks[pack];
      const float2 low = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(&values.x));
      const float2 high = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(&values.y));
      outputPacks[pack] = make_float4(low.x, low.y, high.x, high.y);
    }

    const std::size_t scalarBegin = packCount * 4;
    for (std::size_t i = scalarBegin + group.thread_id_in_group;
         i < elementCount;
         i += group.group_size) {
      args.receiver_output_base[elementOffset + i] =
          __bfloat162float(stagingBf16[i]);
    }
#endif
    return nbytes;
  }
};

} // namespace comms::prims
