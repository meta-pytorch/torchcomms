// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

struct Memcpy {
  template <typename... Args>
  __device__ __forceinline__ static void send(
      char* staging,
      const char* src,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*byte_offset*/,
      Args...) {
    memcpy_vectorized(staging, src, nbytes, group);
  }

  template <typename... Args>
  __device__ __forceinline__ static void recv(
      char* dst,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*byte_offset*/,
      Args...) {
    memcpy_vectorized(dst, staging, nbytes, group);
  }

  template <typename... Args>
  __device__ __forceinline__ static void recv_forward(
      char* dst,
      char* fwd_staging,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*byte_offset*/,
      Args...) {
    if (dst) {
      memcpy_dual(dst, fwd_staging, staging, nbytes, group);
    } else {
      memcpy_vectorized(fwd_staging, staging, nbytes, group);
    }
  }
};

} // namespace comms::pipes
