// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Topology-agnostic device utilities shared by every fused AllReduce kernel
// (Tree, hierarchical ring, ...) and every NVL variant (NvlDirect, NvlSharp).
// Namespace: ctran::allreduce::common. This header carries no algorithm- or
// transport-variant-specific implementation, so a new NVL or IB variant can
// reuse these helpers without pulling in another variant's implementation
// symbols.

#pragma once

#if defined(ENABLE_PRIMS)

#include <cstddef>
#include <cstdint>

#include "comms/ctran/algos/AllReduce/AllReduceFusedTypes.h"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Tile.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/Transport.cuh"

namespace ctran::allreduce::common {

/** Return true when a pointer satisfies the tile API's 16-byte alignment. */
__device__ __forceinline__ bool isAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

/**
 * Accumulate `staging` into `accum` without assuming tile alignment.
 *
 * This path covers small buffers and tails where using the Prims tile API is
 * not profitable or not legal.
 */
template <typename T>
__device__ __forceinline__ void scalarReduce(
    T* accum,
    const T* staging,
    size_t nelems,
    comms::prims::ThreadGroup& group) {
  for (size_t i = group.thread_id_in_group; i < nelems; i += group.group_size) {
    accum[i] += staging[i];
  }
}

/**
 * Accumulate `staging` into `accum` using Prims tiles when alignment allows.
 */
template <typename T, int kTileElems, int kGroupSize>
__device__ __forceinline__ void tileReduce(
    T* accum,
    const T* staging,
    size_t nelems,
    comms::prims::ThreadGroup& group) {
  if (!isAligned16(accum) || !isAligned16(staging) || nelems < kTileElems) {
    scalarReduce(accum, staging, nelems, group);
    return;
  }

  const size_t nFullTiles = nelems / kTileElems;
  const size_t rem = nelems % kTileElems;

  for (size_t t = 0; t < nFullTiles; t++) {
    auto acc =
        comms::prims::tile_load<T, kTileElems, kGroupSize>(accum, t, group);
    comms::prims::
        tile_load_accumulate<T, comms::prims::SumOp, kTileElems, kGroupSize>(
            acc, staging, t, group);
    comms::prims::tile_store<T, kTileElems, kGroupSize>(accum, t, acc, group);
  }
  if (rem > 0) {
    auto acc = comms::prims::tile_load<T, kTileElems, kGroupSize>(
        accum, nFullTiles, group, rem);
    comms::prims::
        tile_load_accumulate<T, comms::prims::SumOp, kTileElems, kGroupSize>(
            acc, staging, nFullTiles, group, rem);
    comms::prims::tile_store<T, kTileElems, kGroupSize>(
        accum, nFullTiles, acc, group, rem);
  }
}

/**
 * Compute the actual number of elements in a segment, including tail handling.
 */
__device__ __forceinline__ size_t
actualSegElems(size_t count, size_t segmentElems, int rank) {
  const size_t start = static_cast<size_t>(rank) * segmentElems;
  if (start >= count) {
    return 0;
  }
  return (start + segmentElems <= count) ? segmentElems : (count - start);
}

/**
 * Return the largest owner tile participating in local NVL exchange.
 */
template <typename T>
__device__ __forceinline__ size_t maxOwnerTileBytes(
    const common::CommonKernArgs& args,
    const comms::prims::ThreadGroup& group) {
  size_t maxBytes = 0;
  for (int owner = 0; owner < args.pMin; owner++) {
    const size_t ownerElems =
        actualSegElems(args.count, args.segmentElems, owner);
    const comms::prims::TiledBuffer<char> ownerTile(
        nullptr, ownerElems * sizeof(T), group);
    const size_t ownerTileBytes = ownerTile.bytes();
    maxBytes = ownerTileBytes > maxBytes ? ownerTileBytes : maxBytes;
  }
  return maxBytes;
}

/**
 * Return a pipeline window that is valid for all local NVL peers.
 */
__device__ __forceinline__ size_t nvlPipelineWindow(
    const common::CommonKernArgs& args,
    const comms::prims::ThreadGroup& group) {
  size_t window = 0;
  for (int peer = 0; peer < args.nLocalRanks; peer++) {
    if (peer == args.localRank) {
      continue;
    }
    const int peerGlobalRank = args.localRankToGlobalRank[peer];
    const size_t peerWindow =
        args.transports[peerGlobalRank].p2p_nvl.pipeline_window(
            group.total_groups);
    window = window == 0 || peerWindow < window ? peerWindow : window;
  }
  return window;
}

/**
 * Return the byte count for the current pipeline step.
 */
__device__ __forceinline__ size_t
pipelineStepBytes(size_t totalBytes, size_t offset, size_t pipelineWindow) {
  const size_t remaining = totalBytes - offset;
  return remaining < pipelineWindow ? remaining : pipelineWindow;
}

/**
 * Convert the physical block group into the logical num-block group used by
 * cooperative Prims operations.
 */
__device__ __forceinline__ comms::prims::ThreadGroup
logicalDataGroup(comms::prims::ThreadGroup group, int blockId, int numBlocks) {
  group.group_id = static_cast<uint32_t>(blockId);
  group.total_groups = static_cast<uint32_t>(numBlocks);
  return group;
}

/**
 * Copy operation used by IBGDA receives in Phase 2.
 *
 * IBGDA owns the transient recv staging ring. The fused kernel must consume
 * that staging inside this callback before the transport acknowledges and
 * reuses the slot. The operation reduces the staged data into the local
 * accumulator using tile-based vectorized reduction.
 */
template <typename T>
struct IbReduceCopy {
  template <typename... Args>
  __device__ __forceinline__ static void recv(
      char* dst,
      const char* staging,
      size_t nbytes,
      comms::prims::ThreadGroup& group,
      size_t /* byteOffset */,
      Args...) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    T* accum = reinterpret_cast<T*>(dst);
    const T* staged = reinterpret_cast<const T*>(staging);
    const size_t nelems = nbytes / sizeof(T);

    tileReduce<T, common::TileElems<T>::k, common::kBlockSize>(
        accum, staged, nelems, group);
#endif
  }
};

} // namespace ctran::allreduce::common

#endif // ENABLE_PRIMS
