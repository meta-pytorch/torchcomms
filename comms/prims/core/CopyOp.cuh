// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/MemcpyCopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Tile.cuh"

namespace comms::prims {

template <typename T, typename AccumOp, int kTileElems, int kBlockSize>
struct TileReduce {
  // Fixed-size CopyOp policy (see AnsCompress for the variable-size one).
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
      std::size_t byte_offset,
      const char* local_input,
      Args...) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const T* staging_t = reinterpret_cast<const T*>(staging);
    T* dst_t = reinterpret_cast<T*>(dst);
    const T* local_t = reinterpret_cast<const T*>(local_input + byte_offset);
    std::size_t nelems = nbytes / sizeof(T);
    int ntiles = static_cast<int>((nelems + kTileElems - 1) / kTileElems);

    for (int t = 0; t < ntiles; t++) {
      std::size_t valid =
          min(static_cast<std::size_t>(kTileElems),
              nelems - static_cast<std::size_t>(t) * kTileElems);
      auto acc =
          tile_load<T, kTileElems, kBlockSize>(staging_t, t, group, valid);
      tile_load_accumulate<T, AccumOp, kTileElems, kBlockSize>(
          acc, local_t, t, group, valid);
      tile_store<T, kTileElems, kBlockSize>(dst_t, t, acc, group, valid);
    }
#endif
    return nbytes;
  }

  template <typename... Args>
  __device__ __forceinline__ static void forward(
      char* /*dst*/,
      char* fwd_staging,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t byte_offset,
      const char* local_input,
      Args...) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const T* staging_t = reinterpret_cast<const T*>(staging);
    T* fwd_t = reinterpret_cast<T*>(fwd_staging);
    const T* local_t = reinterpret_cast<const T*>(local_input + byte_offset);
    std::size_t nelems = nbytes / sizeof(T);
    int ntiles = static_cast<int>((nelems + kTileElems - 1) / kTileElems);

    for (int t = 0; t < ntiles; t++) {
      std::size_t valid =
          min(static_cast<std::size_t>(kTileElems),
              nelems - static_cast<std::size_t>(t) * kTileElems);
      auto acc =
          tile_load<T, kTileElems, kBlockSize>(staging_t, t, group, valid);
      tile_load_accumulate<T, AccumOp, kTileElems, kBlockSize>(
          acc, local_t, t, group, valid);
      tile_store<T, kTileElems, kBlockSize>(fwd_t, t, acc, group, valid);
    }
#endif
  }
};

// Register/tile-staged reduce.
template <typename T, typename AccumOp, int kTileElems, int kBlockSize>
struct TileReduceStaged {
  __host__ __device__ static constexpr std::size_t smem_bytes() {
    return 0;
  }

  // Fixed-size CopyOp policy (see AnsCompress for the variable-size one).
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
      std::size_t byte_offset,
      const char* local_input,
      Args...) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const T* staging_t = reinterpret_cast<const T*>(staging);
    T* dst_t = reinterpret_cast<T*>(dst);
    const T* local_t = reinterpret_cast<const T*>(local_input + byte_offset);
    std::size_t nelems = nbytes / sizeof(T);
    int ntiles = static_cast<int>((nelems + kTileElems - 1) / kTileElems);

    for (int t = 0; t < ntiles; t++) {
      std::size_t valid =
          min(static_cast<std::size_t>(kTileElems),
              nelems - static_cast<std::size_t>(t) * kTileElems);
      auto acc =
          tile_load<T, kTileElems, kBlockSize>(staging_t, t, group, valid);
      auto local =
          tile_load<T, kTileElems, kBlockSize>(local_t, t, group, valid);
      tile_accumulate<T, AccumOp, kTileElems, kBlockSize>(acc, local);
      tile_store<T, kTileElems, kBlockSize>(dst_t, t, acc, group, valid);
    }
#endif
    return nbytes;
  }

  template <typename... Args>
  __device__ __forceinline__ static void forward(
      char* /*dst*/,
      char* fwd_staging,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t byte_offset,
      const char* local_input,
      Args...) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const T* staging_t = reinterpret_cast<const T*>(staging);
    T* fwd_t = reinterpret_cast<T*>(fwd_staging);
    const T* local_t = reinterpret_cast<const T*>(local_input + byte_offset);
    std::size_t nelems = nbytes / sizeof(T);
    int ntiles = static_cast<int>((nelems + kTileElems - 1) / kTileElems);

    for (int t = 0; t < ntiles; t++) {
      std::size_t valid =
          min(static_cast<std::size_t>(kTileElems),
              nelems - static_cast<std::size_t>(t) * kTileElems);
      auto acc =
          tile_load<T, kTileElems, kBlockSize>(staging_t, t, group, valid);
      auto local =
          tile_load<T, kTileElems, kBlockSize>(local_t, t, group, valid);
      tile_accumulate<T, AccumOp, kTileElems, kBlockSize>(acc, local);
      tile_store<T, kTileElems, kBlockSize>(fwd_t, t, acc, group, valid);
    }
#endif
  }
};

} // namespace comms::prims
