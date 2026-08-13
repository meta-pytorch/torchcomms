// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/DeviceMacros.cuh"
#include "comms/prims/core/ThreadGroup.cuh"

namespace comms::prims {

struct Memcpy {
  // Fixed-size CopyOp policy: the transport reserves exactly `chunkSize`
  // per sub-chunk and emits exactly `nbytes`. See AnsCompress (CopyOp.cuh)
  // for the variable-size counterpart that overrides these.
  static constexpr bool kVariableSize = false;
  static constexpr std::size_t kActivationThreshold = 0;
  __host__ __device__ __forceinline__ static constexpr std::size_t
  worst_case_chunk_stride(std::size_t chunkSize) {
    return chunkSize;
  }

  template <typename... Args>
  __device__ __forceinline__ static std::size_t send(
      char* staging,
      const char* src,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*byte_offset*/,
      Args...) {
    memcpy_vectorized(staging, src, nbytes, group);
    return nbytes;
  }

  template <typename... Args>
  __device__ __forceinline__ static std::size_t recv(
      char* dst,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*byte_offset*/,
      Args...) {
    memcpy_vectorized(dst, staging, nbytes, group);
    return nbytes;
  }

  template <typename... Args>
  __device__ __forceinline__ static void forward(
      char* dst,
      char* fwd_staging,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*byte_offset*/,
      Args...) {
    if (dst) {
      memcpy_vectorized(dst, fwd_staging, staging, nbytes, group);
    } else {
      memcpy_vectorized(fwd_staging, staging, nbytes, group);
    }
  }

  template <typename P, typename... Args>
  __device__ __forceinline__ static void sendLL(
      ThreadGroup& group,
      char* staging,
      const char* src,
      std::size_t nbytes,
      std::size_t /*byte_offset*/,
      typename P::FlagType flagVal,
      Args...) {
    static_assert(P::kData == 4 && P::kFlag == 4);
#if PIPES_IS_DEVICE_COMPILE
    const std::size_t nPackets = P::packet_count(nbytes);
    for (std::size_t i = group.thread_id_in_group; i < nPackets;
         i += group.group_size) {
      uint32_t payload = 0;
      const std::size_t valid = P::valid_payload(i, nbytes);
      const std::size_t off = i * static_cast<std::size_t>(P::kData);
      auto* payloadBytes = reinterpret_cast<char*>(&payload);
#pragma unroll
      for (int b = 0; b < P::kData; ++b) {
        if (static_cast<std::size_t>(b) < valid) {
          payloadBytes[b] = src[off + b];
        }
      }
      const uint64_t packet = (static_cast<uint64_t>(flagVal) << 32) | payload;
      auto* dst = reinterpret_cast<volatile uint64_t*>(
          staging + i * static_cast<std::size_t>(P::kPacketBytes));
      *dst = packet;
    }
    group.sync();
#else
    (void)group;
    (void)staging;
    (void)src;
    (void)nbytes;
    (void)flagVal;
#endif
  }

  template <typename P, typename Timeout, typename... Args>
  __device__ __forceinline__ static void recvLL(
      ThreadGroup& group,
      char* dst,
      const char* staging,
      std::size_t nbytes,
      std::size_t /*byte_offset*/,
      typename P::FlagType flagVal,
      const Timeout& timeout,
      Args...) {
    static_assert(P::kData == 4 && P::kFlag == 4);
#if PIPES_IS_DEVICE_COMPILE
    constexpr uint32_t kTimeoutPollMask = 1023;
    const std::size_t nPackets = P::packet_count(nbytes);
    for (std::size_t i = group.thread_id_in_group; i < nPackets;
         i += group.group_size) {
      const auto* src = reinterpret_cast<const volatile uint64_t*>(
          staging + i * static_cast<std::size_t>(P::kPacketBytes));
      uint64_t packet = 0;
      uint32_t spins = 0;
      do {
        packet = *src;
        if (((++spins & kTimeoutPollMask) == 0) && timeout.checkExpired()) {
          PIPES_DEVICE_TRAP();
        }
      } while (static_cast<typename P::FlagType>(packet >> 32) != flagVal);

      const auto payload = static_cast<uint32_t>(packet);
      const auto* payloadBytes = reinterpret_cast<const char*>(&payload);
      const std::size_t valid = P::valid_payload(i, nbytes);
      const std::size_t off = i * static_cast<std::size_t>(P::kData);
#pragma unroll
      for (int b = 0; b < P::kData; ++b) {
        if (static_cast<std::size_t>(b) < valid) {
          dst[off + b] = payloadBytes[b];
        }
      }
    }
    group.sync();
#else
    (void)group;
    (void)dst;
    (void)staging;
    (void)nbytes;
    (void)flagVal;
    (void)timeout;
#endif
  }
};

} // namespace comms::prims
