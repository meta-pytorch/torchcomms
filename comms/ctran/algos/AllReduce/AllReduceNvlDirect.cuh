// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#if defined(ENABLE_PRIMS)

#include <cstddef>
#include <cstdint>

#include "comms/ctran/algos/AllReduce/AllReduceV2Types.h"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Tile.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/Transport.cuh"

namespace ctran::allreduce::nvl {

/** Return true when a pointer satisfies the tile API's 16-byte alignment. */
__device__ __forceinline__ bool isAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

/**
 * Accumulate `staging` into `accum` without assuming tile alignment.
 *
 * This path covers small buffers and tails where using the Pipes tile API is
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
 * Accumulate `staging` into `accum` using Pipes tiles when alignment allows.
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
 * Copy operation used by cooperative NVL receives in Phase 1.
 *
 * The transport writes the remote payload into staging first; this functor
 * then combines that staged payload with the current local input for the same
 * segment and writes the result into `dst`. Aligned slices use the tile
 * reduction path; unaligned tail segments fall back to scalar reduction
 * because the Pipes tile API requires 16-byte-aligned pointers.
 */
template <typename T>
struct NvlReduceCopy {
  template <typename... Args>
  __device__ __forceinline__ static void recv(
      char* dst,
      const char* staging,
      size_t nbytes,
      comms::prims::ThreadGroup& group,
      size_t byteOffset,
      const char* localInput,
      Args...) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    T* out = reinterpret_cast<T*>(dst);
    const T* staged = reinterpret_cast<const T*>(staging);
    const T* local = reinterpret_cast<const T*>(localInput + byteOffset);
    const size_t nelems = nbytes / sizeof(T);

    if (out != local) {
      for (size_t i = group.thread_id_in_group; i < nelems;
           i += group.group_size) {
        out[i] = local[i];
      }
      group.sync();
    }
    tileReduce<T, common::kNvlTileElems, common::kBlockSize>(
        out, staged, nelems, group);
#endif
  }
};

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
 * Tile view for one logical segment owned by the current num-block group.
 */
struct SegmentTile {
  size_t offsetBytes;
  size_t bytes;
};

__device__ __forceinline__ SegmentTile
segmentTileForBlock(size_t totalBytes, int numBlocks, int blockId) {
  comms::prims::TiledBuffer<char> tile(nullptr, totalBytes, numBlocks);
  return SegmentTile{
      .offsetBytes = static_cast<size_t>(blockId) * tile.tile_elements,
      .bytes = tile.tile_bytes(blockId),
  };
}

__device__ __forceinline__ SegmentTile
segmentTile(size_t totalBytes, const comms::prims::ThreadGroup& group) {
  return segmentTileForBlock(totalBytes, group.total_groups, group.group_id);
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
    const auto ownerTile = segmentTile(ownerElems * sizeof(T), group);
    maxBytes = ownerTile.bytes > maxBytes ? ownerTile.bytes : maxBytes;
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

// ============================================================================
// Phase 1: NVL ReduceScatter (AllToFewer pattern)
//
// Each GPU's input tensor M is partitioned into pMin segments of size
// segmentElems. All nLocalRanks GPUs participate. The first pMin GPUs
// (segment owners) each hold one locally-reduced segment after this phase.
//
// Communication: for each peer, the paired send/recv operations happen
// simultaneously. We iterate peers in a rotating order so that at each step,
// sends and recvs are correctly paired across GPUs.
//
// Owners: self-copy own segment to phase2Buf, then recv-reduce from each peer.
// Non-owners: send segment data for each owner to that owner.
//
// This phase is generic across fused AllReduce algorithms (Tree, Ring, ...).
// ============================================================================
template <typename T>
__device__ __noinline__ void phase1ReduceScatter(
    const common::CommonKernArgs& args,
    comms::prims::ThreadGroup& group) {
  const int localRank = args.localRank;
  const int nLocalRanks = args.nLocalRanks;
  const int pMin = args.pMin;
  const size_t segmentElems = args.segmentElems;
  const size_t segmentBytes = segmentElems * sizeof(T);

  const char* sendbuff = static_cast<const char*>(args.sendbuff);
  char* phase2Buf = static_cast<char*>(args.phase2Buf);

  comms::prims::Timeout timeout{};

  // Every owner initializes its tile before receiving peer contributions. The
  // tile view is derived from the group identity, so numBlocks=1 is just the
  // degenerate full-segment tile.
  SegmentTile myTile{};
  bool copiedLocalTile = false;
  if (localRank < pMin) {
    const size_t myActualElems =
        actualSegElems(args.count, segmentElems, localRank);
    myTile = segmentTile(myActualElems * sizeof(T), group);
    const size_t mySegmentOffset =
        static_cast<size_t>(localRank) * segmentBytes;
    const char* src = sendbuff + mySegmentOffset + myTile.offsetBytes;
    char* dst = phase2Buf + myTile.offsetBytes;
    if (myTile.bytes > 0 && dst != src) {
      comms::prims::memcpy_vectorized(dst, src, myTile.bytes, group);
      copiedLocalTile = true;
    }
  }

  if (nLocalRanks <= 1) {
    // Local-only topology has no NVL traffic. Synchronize only when this block
    // copied out-of-place data that Phase 2 may read immediately.
    if (copiedLocalTile) {
      group.sync();
    }
    return;
  }

  group.sync();

  const size_t pipelineWindow = nvlPipelineWindow(args, group);
  PIPES_DEVICE_CHECK_MSG(
      pipelineWindow != 0,
      "ctree Phase 1 NVL reduce-scatter pipeline window is zero");

  const size_t maxTileBytes = maxOwnerTileBytes<T>(args, group);
  char* myDst = phase2Buf + myTile.offsetBytes;

  for (size_t off = 0; off < maxTileBytes; off += pipelineWindow) {
    for (int owner = 0; owner < pMin; owner++) {
      if (owner == localRank) {
        continue;
      }

      const size_t ownerActualElems =
          actualSegElems(args.count, segmentElems, owner);
      const auto ownerTile = segmentTile(ownerActualElems * sizeof(T), group);
      if (off >= ownerTile.bytes) {
        continue;
      }

      const int ownerGlobalRank = args.localRankToGlobalRank[owner];
      const size_t ownerSegmentOffset =
          static_cast<size_t>(owner) * segmentBytes;
      const size_t window =
          pipelineStepBytes(ownerTile.bytes, off, pipelineWindow);
      args.transports[ownerGlobalRank].p2p_nvl.send(
          group,
          sendbuff + ownerSegmentOffset + ownerTile.offsetBytes + off,
          window,
          group.total_groups,
          0,
          timeout);
    }

    if (localRank < pMin && off < myTile.bytes) {
      const size_t window =
          pipelineStepBytes(myTile.bytes, off, pipelineWindow);
      for (int peer = 0; peer < nLocalRanks; peer++) {
        if (peer == localRank) {
          continue;
        }
        const int peerGlobalRank = args.localRankToGlobalRank[peer];
        args.transports[peerGlobalRank].p2p_nvl.recv<NvlReduceCopy<T>>(
            group,
            myDst + off,
            window,
            group.total_groups,
            0,
            timeout,
            myDst + off);
      }
    }
  }
}

// ============================================================================
// Phase 3: NVL AllGather
//
// Segment owners broadcast their globally-reduced segment to all local peers.
// After Phase 2, each owner's recvbuff contains its globally-reduced segment
// at offset [localRank * segmentBytes].
//
// Communication: uses rotating-peer order for globally consistent pairing.
// Owners send their segment; all GPUs receive from owners.
//
// This phase is generic across fused AllReduce algorithms (Tree, Ring, ...).
// ============================================================================
template <typename T>
__device__ __noinline__ void phase3AllGather(
    const common::CommonKernArgs& args,
    comms::prims::ThreadGroup& group) {
  const int localRank = args.localRank;
  const int nLocalRanks = args.nLocalRanks;
  const int pMin = args.pMin;
  const size_t segmentElems = args.segmentElems;
  const size_t segmentBytes = segmentElems * sizeof(T);

  char* recvbuff = static_cast<char*>(args.recvbuff);

  comms::prims::Timeout timeout{};

  if (nLocalRanks <= 1) {
    return;
  }

  const size_t pipelineWindow = nvlPipelineWindow(args, group);
  PIPES_DEVICE_CHECK_MSG(
      pipelineWindow != 0,
      "ctree Phase 3 NVL all-gather pipeline window is zero");

  const size_t maxTileBytes = maxOwnerTileBytes<T>(args, group);
  SegmentTile myTile{};
  if (localRank < pMin) {
    const size_t myActualElems =
        actualSegElems(args.count, segmentElems, localRank);
    myTile = segmentTile(myActualElems * sizeof(T), group);
  }

  for (size_t off = 0; off < maxTileBytes; off += pipelineWindow) {
    if (localRank < pMin && off < myTile.bytes) {
      const size_t mySegmentOffset =
          static_cast<size_t>(localRank) * segmentBytes;
      const size_t window =
          pipelineStepBytes(myTile.bytes, off, pipelineWindow);
      for (int peer = 0; peer < nLocalRanks; peer++) {
        if (peer == localRank) {
          continue;
        }
        const int peerGlobalRank = args.localRankToGlobalRank[peer];
        args.transports[peerGlobalRank].p2p_nvl.send(
            group,
            recvbuff + mySegmentOffset + myTile.offsetBytes + off,
            window,
            group.total_groups,
            0,
            timeout);
      }
    }

    for (int owner = 0; owner < pMin; owner++) {
      if (owner == localRank) {
        continue;
      }

      const size_t ownerActualElems =
          actualSegElems(args.count, segmentElems, owner);
      const auto ownerTile = segmentTile(ownerActualElems * sizeof(T), group);
      if (off >= ownerTile.bytes) {
        continue;
      }

      const int ownerGlobalRank = args.localRankToGlobalRank[owner];
      const size_t ownerSegmentOffset =
          static_cast<size_t>(owner) * segmentBytes;
      const size_t window =
          pipelineStepBytes(ownerTile.bytes, off, pipelineWindow);
      args.transports[ownerGlobalRank].p2p_nvl.recv(
          group,
          recvbuff + ownerSegmentOffset + ownerTile.offsetBytes + off,
          window,
          group.total_groups,
          0,
          timeout);
    }
  }
}

// ============================================================================
// Fused orchestrator
//
// Runs the shared NVL ReduceScatter (Phase 1), an algorithm-specific cross-node
// IB phase (Phase 2), and the shared NVL AllGather (Phase 3) for one logical
// data tile owned by `group`. `phase2` is any callable invoked as
// `phase2(group)`; it must reduce each owner segment across nodes and leave the
// globally reduced segment in `common.phase2Buf` (the Phase-2 contract). Passed
// as a template parameter so the algorithm-specific phase inlines and the
// kernel keeps a single `__launch_bounds__` register budget.
//
// Phase-transition `group.sync()`s and the degenerate-topology skips
// (nLocalRanks <= 1 → no NVL phases; nNodes <= 1 → no IB phase) are identical
// across algorithms and owned here. Correctness does not require inter-block
// synchronization.
// ============================================================================
template <typename T, typename Phase2Fn>
__device__ __forceinline__ void runAllReduceFused(
    const common::CommonKernArgs& args,
    comms::prims::ThreadGroup& group,
    Phase2Fn&& phase2) {
  phase1ReduceScatter<T>(args, group);
  if (args.nLocalRanks > 1) {
    group.sync();
  }
  if (args.nNodes > 1) {
    phase2(group);
    if (args.nLocalRanks > 1) {
      group.sync();
    }
  }
  if (args.nLocalRanks > 1) {
    phase3AllGather<T>(args, group);
  }
}

} // namespace ctran::allreduce::nvl

#endif // ENABLE_PRIMS
