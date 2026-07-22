// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDA_ARCH__)
#include <cuda_pipeline.h>
#endif

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Tile.cuh"

namespace comms::prims {

template <typename T, int kTileElems, int kBlockSize, int kStages = 2>
struct CpAsyncSmemTile {
  static_assert(kStages >= 2, "cp.async pipeline needs at least 2 stages");

  static constexpr int kEPV = VecOps<T>::kElemsPerVec;
  static constexpr int kTileVecs = kTileElems / kEPV;
  static_assert(
      kTileElems % kEPV == 0,
      "kTileElems must be a multiple of the 16B vector width");

  __host__ __device__ static constexpr std::size_t smem_bytes() {
    return static_cast<std::size_t>(kStages) * 2 *
        static_cast<std::size_t>(kTileElems) * sizeof(T);
  }

  __device__ __forceinline__ static char* smem() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) char cp_async_smem_reduce[];
    return cp_async_smem_reduce;
#else
    return nullptr;
#endif
  }

  __host__ __device__ static constexpr int prefetch_distance() {
    return kStages - 1;
  }

  __host__ __device__ static constexpr int stage_for_tile(int tile) {
    return tile % kStages;
  }

  __host__ __device__ static constexpr int num_tiles(std::size_t nelems) {
    return static_cast<int>((nelems + kTileElems - 1) / kTileElems);
  }

  __device__ __forceinline__ static void check_alignment(
      const void* ptr,
      const char* name) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (reinterpret_cast<uintptr_t>(ptr) % alignof(uint4) != 0) {
      printf(
          "%s: ptr %p is not 16-byte aligned (required for uint4 vectorized access)\n",
          name,
          ptr);
      __trap();
    }
#endif
  }

  __device__ __forceinline__ static void enqueue_load(
      char* smem_base,
      int tile,
      const T* staging,
      const T* local,
      std::size_t nelems,
      const ThreadGroup& group) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    Stage base = stage_base(smem_base, stage_for_tile(tile));
    const int full_vecs = static_cast<int>(valid_elems(tile, nelems) / kEPV);
    const std::size_t tile_vec_start =
        static_cast<std::size_t>(tile) * kTileVecs;
    enqueue_vector_load(
        base.staging,
        reinterpret_cast<const uint4*>(staging) + tile_vec_start,
        full_vecs,
        group);
    enqueue_vector_load(
        base.local,
        reinterpret_cast<const uint4*>(local) + tile_vec_start,
        full_vecs,
        group);
#endif
  }

  __device__ __forceinline__ static void commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __pipeline_commit();
#endif
  }

  __device__ __forceinline__ static void wait_prior(int in_flight) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __pipeline_wait_prior(in_flight);
#endif
  }

  template <typename Op>
  __device__ __forceinline__ static void reduce_store(
      T* dst,
      char* smem_base,
      int tile,
      const T* staging,
      const T* local,
      std::size_t nelems,
      const ThreadGroup& group) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    Stage base = stage_base(smem_base, stage_for_tile(tile));
    const std::size_t valid = valid_elems(tile, nelems);
    const int full_vecs = static_cast<int>(valid / kEPV);
    const int rem = static_cast<int>(valid % kEPV);
    const std::size_t global_tile_start =
        static_cast<std::size_t>(tile) * kTileVecs;
    uint4* dst_v = reinterpret_cast<uint4*>(dst);

    for (int v = static_cast<int>(group.thread_id_in_group); v < full_vecs;
         v += static_cast<int>(group.group_size)) {
      uint4 acc = base.staging[v];
      VecOps<T>::reduce(Op{}, acc, base.local[v]);
      dst_v[global_tile_start + v] = acc;
    }

    if (rem > 0 && group.thread_id_in_group == 0) {
      const std::size_t tile_start =
          static_cast<std::size_t>(tile) * static_cast<std::size_t>(kTileElems);
      const std::size_t rem_start = tile_start + full_vecs * kEPV;
      for (int e = 0; e < rem; ++e) {
        T acc = staging[rem_start + e];
        VecOps<T>::reduce_scalar(Op{}, acc, local[rem_start + e]);
        dst[rem_start + e] = acc;
      }
    }
#elif defined(__CUDA_ARCH__)
    static_assert(
        __CUDA_ARCH__ >= 800, "CpAsyncSmemTile requires cp.async support");
#endif
  }

 private:
  struct Stage {
    uint4* staging;
    uint4* local;
  };

  __device__ __forceinline__ static Stage stage_base(
      char* smem_base,
      int stage) {
    uint4* base = reinterpret_cast<uint4*>(smem_base);
    uint4* stage_base = base + static_cast<std::size_t>(stage) * 2 * kTileVecs;
    return Stage{
        .staging = stage_base,
        .local = stage_base + kTileVecs,
    };
  }

  __host__ __device__ static constexpr std::size_t valid_elems(
      int tile,
      std::size_t nelems) {
    const std::size_t tile_start =
        static_cast<std::size_t>(tile) * static_cast<std::size_t>(kTileElems);
    if (tile_start >= nelems) {
      return 0;
    }
    const std::size_t remaining = nelems - tile_start;
    return remaining < static_cast<std::size_t>(kTileElems)
        ? remaining
        : static_cast<std::size_t>(kTileElems);
  }

  __device__ __forceinline__ static void enqueue_vector_load(
      uint4* __restrict__ dst,
      const uint4* __restrict__ src,
      int full_vecs,
      const ThreadGroup& group) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    for (int v = static_cast<int>(group.thread_id_in_group); v < full_vecs;
         v += static_cast<int>(group.group_size)) {
      __pipeline_memcpy_async(&dst[v], &src[v], sizeof(uint4));
    }
#endif
  }
};

} // namespace comms::prims
