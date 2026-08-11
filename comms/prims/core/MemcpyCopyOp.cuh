// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/LLImpl.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"

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

  // Low-latency (LL) protocol hooks: packet-aware encode/decode over the
  // data+flag interleaved staging of LlxPacket<P>. `byte_offset` carries the
  // chunk's offset within the logical transfer, matching send()/recv() -- a
  // plain copy ignores it, but offset-sensitive ops (quantization, reduction)
  // need it for alignment and addressing, so it must not be dropped here.
  // Plain-copy Memcpy delegates to the LLImpl<P> codec; a reduce/convert CopyOp
  // can override these with a packet-aware implementation. Presence is detected
  // by has_sendLL_v / has_recvLL_v.
  template <typename P, typename... Args>
  __device__ __forceinline__ static void sendLL(
      ThreadGroup& group,
      char* staging,
      const char* src,
      std::size_t nbytes,
      std::size_t /*byte_offset*/,
      typename P::FlagType flagVal,
      Args...) {
    LLImpl<P>::pack(group, staging, src, nbytes, flagVal);
  }

  template <typename P, typename... Args>
  // Takes a Timeout where recv()/sendLL() do not: LL's readiness wait happens
  // inside the codec (the flag is in the payload), so this is the one hook that
  // can block indefinitely and therefore the one that needs a deadline.
  __device__ __forceinline__ static void recvLL(
      ThreadGroup& group,
      char* dst,
      const char* staging,
      std::size_t nbytes,
      std::size_t /*byte_offset*/,
      typename P::FlagType flagVal,
      const Timeout& timeout,
      Args...) {
    LLImpl<P>::unpack(group, dst, staging, nbytes, flagVal, timeout);
  }
};

// Detection traits: does CopyOp `Op` provide packet-aware LL hooks for packet
// geometry `P`? A CopyOp opts into the LL protocol by defining sendLL<P> /
// recvLL<P> (see Memcpy). The IBGDA transport asserts these before dispatching
// the LL send/recv path, so an op that only supports contiguous copy fails with
// a clear message instead of a deep template error.
template <typename Op, typename P, typename = void>
inline constexpr bool has_sendLL_v = false;
template <typename Op, typename P>
inline constexpr bool has_sendLL_v<
    Op,
    P,
    std::void_t<decltype(Op::template sendLL<P>(
        std::declval<ThreadGroup&>(),
        std::declval<char*>(),
        std::declval<const char*>(),
        std::declval<std::size_t>(),
        std::declval<std::size_t>(),
        std::declval<typename P::FlagType>()))>> = true;

template <typename Op, typename P, typename = void>
inline constexpr bool has_recvLL_v = false;
template <typename Op, typename P>
inline constexpr bool has_recvLL_v<
    Op,
    P,
    std::void_t<decltype(Op::template recvLL<P>(
        std::declval<ThreadGroup&>(),
        std::declval<char*>(),
        std::declval<const char*>(),
        std::declval<std::size_t>(),
        std::declval<std::size_t>(),
        std::declval<typename P::FlagType>(),
        std::declval<const Timeout&>()))>> = true;

} // namespace comms::prims
