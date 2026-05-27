// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/MemcpyCopyOp.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Tile.cuh"

namespace comms::pipes {

/**
 * MemcpyAndSelfCopy — Dual-destination send CopyOp.
 *
 * Fuses a local self-copy into the RDMA send staging copy. Reads src
 * once and writes to both the staging buffer and a caller-supplied
 * self_dst, eliminating a separate memcpy + barrier before the send.
 * The self_dst pointer is forwarded through the Args... variadic pack
 * of P2pIbgdaTransportDevice::send().
 */
struct MemcpyAndSelfCopy {
  template <typename... Args>
  __device__ __forceinline__ static void send(
      char* staging,
      const char* src,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t byte_offset,
      char* self_dst,
      Args...) {
    memcpy_vectorized(staging, self_dst + byte_offset, src, nbytes, group);
  }
};

template <typename T, typename AccumOp, int kTileElems, int kBlockSize>
struct TileReduce {
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

template <typename T, typename AccumOp, int kTileElems, int kBlockSize>
struct TileReduceStaged {
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

} // namespace comms::pipes
