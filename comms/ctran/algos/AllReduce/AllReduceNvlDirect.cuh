// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// NvlDirect NVL-phase implementation for the fused AllReduce framework:
// direct (AllToFewer) NVL ReduceScatter (Phase 1) and NVL AllGather (Phase 3),
// plus the `Ops` policy that exposes them to the shared orchestrator.
// Namespace: ctran::allreduce::nvl::direct. Topology-agnostic helpers live in
// AllReduceFusedCommon.cuh (ctran::allreduce::common); the phase-sequencing
// orchestrator lives in AllReduceFused.cuh
// (ctran::allreduce::fused). A future NvlSharp variant mirrors this file under
// ctran::allreduce::nvl::sharp with the same `Ops` shape and reuses
// AllReduceFusedCommon.cuh.

#pragma once

#if defined(ENABLE_PRIMS)

#include <cstddef>
#include <cstdint>

#include "comms/ctran/algos/AllReduce/AllReduceFusedCommon.cuh"
#include "comms/ctran/algos/AllReduce/AllReduceFusedTypes.h"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/Transport.cuh"

namespace ctran::allreduce::nvl::direct {

/**
 * Copy operation used by cooperative NVL receives in Phase 1.
 *
 * The transport writes the remote payload into staging first; this functor
 * then combines that staged payload with the current local input for the same
 * segment and writes the result into `dst`. Aligned slices use the tile
 * reduction path; unaligned tail segments fall back to scalar reduction
 * because the Prims tile API requires 16-byte-aligned pointers.
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
    common::tileReduce<T, common::kNvlTileElems, common::kBlockSize>(
        out, staged, nelems, group);
#endif
  }
};

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
__device__ __noinline__ void nvlDirectReduceScatter(
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
  common::SegmentTile myTile{};
  bool copiedLocalTile = false;
  if (localRank < pMin) {
    const size_t myActualElems =
        common::actualSegElems(args.count, segmentElems, localRank);
    myTile = common::segmentTile(myActualElems * sizeof(T), group);
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

  const size_t pipelineWindow = common::nvlPipelineWindow(args, group);
  PIPES_DEVICE_CHECK_MSG(
      pipelineWindow != 0,
      "Phase 1 NVL reduce-scatter pipeline window is zero");

  const size_t maxTileBytes = common::maxOwnerTileBytes<T>(args, group);
  char* myDst = phase2Buf + myTile.offsetBytes;

  for (size_t off = 0; off < maxTileBytes; off += pipelineWindow) {
    for (int owner = 0; owner < pMin; owner++) {
      if (owner == localRank) {
        continue;
      }

      const size_t ownerActualElems =
          common::actualSegElems(args.count, segmentElems, owner);
      const auto ownerTile =
          common::segmentTile(ownerActualElems * sizeof(T), group);
      if (off >= ownerTile.bytes) {
        continue;
      }

      const int ownerGlobalRank = args.localRankToGlobalRank[owner];
      const size_t ownerSegmentOffset =
          static_cast<size_t>(owner) * segmentBytes;
      const size_t window =
          common::pipelineStepBytes(ownerTile.bytes, off, pipelineWindow);
      args.transports[ownerGlobalRank].p2p_nvl.send(
          group,
          sendbuff + ownerSegmentOffset + ownerTile.offsetBytes + off,
          window,
          0,
          timeout);
    }

    if (localRank < pMin && off < myTile.bytes) {
      const size_t window =
          common::pipelineStepBytes(myTile.bytes, off, pipelineWindow);
      for (int peer = 0; peer < nLocalRanks; peer++) {
        if (peer == localRank) {
          continue;
        }
        const int peerGlobalRank = args.localRankToGlobalRank[peer];
        args.transports[peerGlobalRank].p2p_nvl.recv<NvlReduceCopy<T>>(
            group, myDst + off, window, 0, timeout, myDst + off);
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
__device__ __noinline__ void nvlDirectAllGather(
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

  const size_t pipelineWindow = common::nvlPipelineWindow(args, group);
  PIPES_DEVICE_CHECK_MSG(
      pipelineWindow != 0, "Phase 3 NVL all-gather pipeline window is zero");

  const size_t maxTileBytes = common::maxOwnerTileBytes<T>(args, group);
  common::SegmentTile myTile{};
  if (localRank < pMin) {
    const size_t myActualElems =
        common::actualSegElems(args.count, segmentElems, localRank);
    myTile = common::segmentTile(myActualElems * sizeof(T), group);
  }

  for (size_t off = 0; off < maxTileBytes; off += pipelineWindow) {
    if (localRank < pMin && off < myTile.bytes) {
      const size_t mySegmentOffset =
          static_cast<size_t>(localRank) * segmentBytes;
      const size_t window =
          common::pipelineStepBytes(myTile.bytes, off, pipelineWindow);
      for (int peer = 0; peer < nLocalRanks; peer++) {
        if (peer == localRank) {
          continue;
        }
        const int peerGlobalRank = args.localRankToGlobalRank[peer];
        args.transports[peerGlobalRank].p2p_nvl.send(
            group,
            recvbuff + mySegmentOffset + myTile.offsetBytes + off,
            window,
            0,
            timeout);
      }
    }

    for (int owner = 0; owner < pMin; owner++) {
      if (owner == localRank) {
        continue;
      }

      const size_t ownerActualElems =
          common::actualSegElems(args.count, segmentElems, owner);
      const auto ownerTile =
          common::segmentTile(ownerActualElems * sizeof(T), group);
      if (off >= ownerTile.bytes) {
        continue;
      }

      const int ownerGlobalRank = args.localRankToGlobalRank[owner];
      const size_t ownerSegmentOffset =
          static_cast<size_t>(owner) * segmentBytes;
      const size_t window =
          common::pipelineStepBytes(ownerTile.bytes, off, pipelineWindow);
      args.transports[ownerGlobalRank].p2p_nvl.recv(
          group,
          recvbuff + ownerSegmentOffset + ownerTile.offsetBytes + off,
          window,
          0,
          timeout);
    }
  }
}

/**
 * NVL-variant policy for the fused orchestrator.
 *
 * `runAllReduceFused<T, NvlOps, IbAllReduce>` dispatches the NVL phases through
 * this policy, so swapping the NVL transport (e.g. to NvlSharp) is a matter of
 * passing a different `Ops` at the call site. `ExtraArgs` is the per-variant
 * device-state slot embedded in each algorithm's `KernArgs`; it is empty for
 * NvlDirect (the direct transport needs no extra state) and lives in the
 * host-safe AllReduceFusedTypes.h so `KernArgs` headers stay host-includable.
 */
struct Ops {
  template <typename T>
  __device__ __forceinline__ static void nvlReduceScatter(
      const common::CommonKernArgs& args,
      comms::prims::ThreadGroup& group) {
    nvlDirectReduceScatter<T>(args, group);
  }
  template <typename T>
  __device__ __forceinline__ static void nvlAllGather(
      const common::CommonKernArgs& args,
      comms::prims::ThreadGroup& group) {
    nvlDirectAllGather<T>(args, group);
  }
  using ExtraArgs = ctran::allreduce::nvl::direct::ExtraArgs;
};

} // namespace ctran::allreduce::nvl::direct

#endif // ENABLE_PRIMS
