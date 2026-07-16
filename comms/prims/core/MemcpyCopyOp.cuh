// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/prims/core/CopyUtils.cuh"
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
};

} // namespace comms::prims
