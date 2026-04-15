// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

namespace comms::pipes {

// Forward declaration — full definition in ThreadGroup.cuh
struct ThreadGroup;

/**
 * TiledBuffer - Typed view that partitions a buffer into aligned tiles.
 *
 * Divides a contiguous buffer into aligned tiles, one per group in a
 * ThreadGroup partition. Each tile is aligned to 16 bytes. The last tile
 * may be smaller than the rest.
 *
 * Two construction modes:
 *
 *   // Mode 1: explicit tile count (host or device)
 *   TiledBuffer<char> tiles(ptr, nbytes, numBlocks);
 *   p2p.send_tile(sub, tiles.tile_data(blockId), tiles.tile_bytes(blockId),
 * ...);
 *
 *   // Mode 2: bind to ThreadGroup (device only, preferred)
 *   TiledBuffer<char> tile(ptr, nbytes, sub);
 *   p2p.send_tile(sub, tile.data(), tile.bytes(), ...);
 *
 * Mode 2 derives num_tiles from group.total_groups and indexes by
 * group.group_id, eliminating manual blockId bookkeeping.
 *
 * @tparam T Element type (e.g., float, __nv_bfloat16, char)
 */
template <typename T>
struct TiledBuffer {
  T* __restrict__ buf;
  std::size_t num_elements;
  int num_tiles;
  std::size_t tile_elements; // elements per tile (aligned, computed)
  int my_tile; // this group's tile index (-1 if unbound)

  /// Construct with explicit tile count (host or device).
  __host__ __device__
  TiledBuffer(T* __restrict__ data, std::size_t num_elements, int num_tiles)
      : buf(data),
        num_elements(num_elements),
        num_tiles(num_tiles),
        my_tile(-1) {
    compute_tile_elements();
  }

  /// Construct bound to a ThreadGroup (device only).
  /// Derives num_tiles from group.total_groups, indexes by group.group_id.
  __device__ TiledBuffer(
      T* __restrict__ data,
      std::size_t num_elements,
      const ThreadGroup& group);

  /// This group's tile data pointer (requires group-bound construction)
  __device__ __forceinline__ T* __restrict__ data() const {
    return buf + my_tile * tile_elements;
  }

  /// This group's tile byte count (requires group-bound construction)
  __device__ __forceinline__ std::size_t bytes() const {
    return tile_bytes(my_tile);
  }

  /// Pointer to tile i's data (explicit indexing)
  __device__ __forceinline__ T* __restrict__ tile_data(int tile_id) const {
    return buf + tile_id * tile_elements;
  }

  /// Number of elements in tile i (last tile may be smaller)
  __device__ __forceinline__ std::size_t tile_size(int tile_id) const {
    std::size_t offset = tile_id * tile_elements;
    if (offset >= num_elements) {
      return 0;
    }
    std::size_t remaining = num_elements - offset;
    return remaining < tile_elements ? remaining : tile_elements;
  }

  /// Bytes in tile i
  __device__ __forceinline__ std::size_t tile_bytes(int tile_id) const {
    return tile_size(tile_id) * sizeof(T);
  }

 private:
  __host__ __device__ void compute_tile_elements() {
    constexpr std::size_t kAlignElems = 16 / sizeof(T) > 0 ? 16 / sizeof(T) : 1;
    tile_elements =
        (((num_elements + num_tiles - 1) / num_tiles) + kAlignElems - 1) &
        ~(kAlignElems - 1);
  }
};

} // namespace comms::pipes

// Include ThreadGroup for the group-bound constructor implementation.
// Placed after the struct to break the circular dependency.
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

template <typename T>
__device__ TiledBuffer<T>::TiledBuffer(
    T* __restrict__ data,
    std::size_t num_elements,
    const ThreadGroup& group)
    : buf(data),
      num_elements(num_elements),
      num_tiles(group.total_groups),
      my_tile(group.group_id) {
  compute_tile_elements();
}

} // namespace comms::pipes
