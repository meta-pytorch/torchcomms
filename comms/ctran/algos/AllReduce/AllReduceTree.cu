// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#if defined(ENABLE_PIPES)

#include <cuda_fp16.h>

#include <cstdint>

#include "comms/ctran/algos/AllReduce/AllReduceTree.cuh"
#include "comms/pipes/CopyOp.cuh"
#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/DeviceCheck.cuh"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Tile.cuh"
#include "comms/pipes/TiledBuffer.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/Transport.cuh"

/** Elements processed by one NVL tile operation in the reduction helpers. */
static constexpr int kNvlTileElems = 15360;
/** Elements processed by one IB tile operation in the reduction helpers. */
static constexpr int kIbTileElems = 5120;

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
    comms::pipes::ThreadGroup& group) {
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
    comms::pipes::ThreadGroup& group) {
  if (!isAligned16(accum) || !isAligned16(staging) || nelems < kTileElems) {
    scalarReduce(accum, staging, nelems, group);
    return;
  }

  const size_t nFullTiles = nelems / kTileElems;
  const size_t rem = nelems % kTileElems;

  for (size_t t = 0; t < nFullTiles; t++) {
    auto acc =
        comms::pipes::tile_load<T, kTileElems, kGroupSize>(accum, t, group);
    comms::pipes::
        tile_load_accumulate<T, comms::pipes::SumOp, kTileElems, kGroupSize>(
            acc, staging, t, group);
    comms::pipes::tile_store<T, kTileElems, kGroupSize>(accum, t, acc, group);
  }
  if (rem > 0) {
    auto acc = comms::pipes::tile_load<T, kTileElems, kGroupSize>(
        accum, nFullTiles, group, rem);
    comms::pipes::
        tile_load_accumulate<T, comms::pipes::SumOp, kTileElems, kGroupSize>(
            acc, staging, nFullTiles, group, rem);
    comms::pipes::tile_store<T, kTileElems, kGroupSize>(
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
struct TreeNvlReduceCopy {
  template <typename... Args>
  __device__ __forceinline__ static void recv(
      char* dst,
      const char* staging,
      size_t nbytes,
      comms::pipes::ThreadGroup& group,
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
    tileReduce<T, kNvlTileElems, ctran::allreduce::tree::kBlockSize>(
        out, staged, nelems, group);
#endif
  }
};

/**
 * Copy operation used by IBGDA child receives in Phase 2.
 *
 * IBGDA owns the transient recv staging ring. CTREE must consume that staging
 * inside this callback before the transport acknowledges and reuses the slot.
 */
template <typename T>
struct TreeIbReduceCopy {
  template <typename... Args>
  __device__ __forceinline__ static void recv(
      char* dst,
      const char* staging,
      size_t nbytes,
      comms::pipes::ThreadGroup& group,
      size_t /* byteOffset */,
      Args...) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    T* accum = reinterpret_cast<T*>(dst);
    const T* staged = reinterpret_cast<const T*>(staging);
    const size_t nelems = nbytes / sizeof(T);

    tileReduce<T, kIbTileElems, ctran::allreduce::tree::kBlockSize>(
        accum, staged, nelems, group);
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
// ============================================================================
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
 * Convert the physical block group into the logical num-block group used by
 * cooperative Pipes operations.
 */
__device__ __forceinline__ comms::pipes::ThreadGroup
logicalDataGroup(comms::pipes::ThreadGroup group, int blockId, int numBlocks) {
  group.group_id = static_cast<uint32_t>(blockId);
  group.total_groups = static_cast<uint32_t>(numBlocks);
  return group;
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
  comms::pipes::TiledBuffer<char> tile(nullptr, totalBytes, numBlocks);
  return SegmentTile{
      .offsetBytes = static_cast<size_t>(blockId) * tile.tile_elements,
      .bytes = tile.tile_bytes(blockId),
  };
}

__device__ __forceinline__ SegmentTile
segmentTile(size_t totalBytes, const comms::pipes::ThreadGroup& group) {
  return segmentTileForBlock(totalBytes, group.total_groups, group.group_id);
}

/**
 * Return the largest owner tile participating in local NVL exchange.
 */
template <typename T>
__device__ __forceinline__ size_t maxOwnerTileBytes(
    const ctran::allreduce::tree::KernArgs& args,
    const comms::pipes::ThreadGroup& group) {
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
    const ctran::allreduce::tree::KernArgs& args,
    const comms::pipes::ThreadGroup& group) {
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

template <typename T>
__device__ __noinline__ void phase1ReduceScatter(
    const ctran::allreduce::tree::KernArgs& args,
    comms::pipes::ThreadGroup& group) {
  const int localRank = args.localRank;
  const int nLocalRanks = args.nLocalRanks;
  const int pMin = args.pMin;
  const size_t segmentElems = args.segmentElems;
  const size_t segmentBytes = segmentElems * sizeof(T);

  const char* sendbuff = static_cast<const char*>(args.sendbuff);
  char* phase2Buf = static_cast<char*>(args.phase2Buf);

  comms::pipes::Timeout timeout{};

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
      comms::pipes::memcpy_vectorized(dst, src, myTile.bytes, group);
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
        args.transports[peerGlobalRank].p2p_nvl.recv<TreeNvlReduceCopy<T>>(
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

enum class TreeLanePhase : uint8_t {
  ReduceLeafSend,
  ReduceRecvChild,
  ReduceInternalSend,
  BcastRootSendChild,
  BcastInternalRecv,
  BcastInternalSendChild,
  BcastLeafRecv,
  Done,
};

__device__ __forceinline__ TreeLanePhase
firstReducePhase(const ctran::allreduce::TreeTopology& tree) {
  if (tree.isRoot && tree.isLeaf) {
    return TreeLanePhase::Done;
  }
  return tree.isLeaf ? TreeLanePhase::ReduceLeafSend
                     : TreeLanePhase::ReduceRecvChild;
}

__device__ __forceinline__ TreeLanePhase
firstBcastPhase(const ctran::allreduce::TreeTopology& tree) {
  if (tree.isRoot && tree.isLeaf) {
    return TreeLanePhase::Done;
  }
  if (tree.isRoot) {
    return TreeLanePhase::BcastRootSendChild;
  }
  return tree.isLeaf ? TreeLanePhase::BcastLeafRecv
                     : TreeLanePhase::BcastInternalRecv;
}

/**
 * Persistent state for one half of the dual IB tree.
 *
 * A full CUDA block repeatedly calls `progressTreeLane()` for both lane states.
 * Each call performs at most one bounded transport step or one local reduction
 * step, and returns immediately if the current transport dependency is not
 * ready.
 */
template <typename T>
struct TreeLaneProgressState {
  ctran::allreduce::TreeTopology tree;
  T* dataBuf{nullptr};
  size_t dataBytes{0};
  size_t pipelineWindow{0};
  size_t offsetBytes{0};
  int activeBlocks{0};
  int childIdx{0};
  bool ibOpActive{false};
  TreeLanePhase phase{TreeLanePhase::Done};
  comms::pipes::IbgdaSendRecvProgressState ibOp{};
};

template <typename T>
__device__ __forceinline__ size_t
currentLaneWindowBytes(const TreeLaneProgressState<T>& state) {
  return pipelineStepBytes(
      state.dataBytes, state.offsetBytes, state.pipelineWindow);
}

template <typename T>
__device__ __forceinline__ void advanceReduceWindow(
    TreeLaneProgressState<T>& state) {
  state.offsetBytes += currentLaneWindowBytes(state);
  state.childIdx = 0;
  state.ibOpActive = false;
  if (state.offsetBytes >= state.dataBytes) {
    state.offsetBytes = 0;
    state.phase = firstBcastPhase(state.tree);
  } else {
    state.phase = firstReducePhase(state.tree);
  }
}

template <typename T>
__device__ __forceinline__ void advanceBcastWindow(
    TreeLaneProgressState<T>& state) {
  state.offsetBytes += currentLaneWindowBytes(state);
  state.childIdx = 0;
  state.ibOpActive = false;
  if (state.offsetBytes >= state.dataBytes) {
    state.phase = TreeLanePhase::Done;
  } else {
    state.phase = firstBcastPhase(state.tree);
  }
}

template <typename T>
__device__ __forceinline__ TreeLaneProgressState<T> makeTreeLaneState(
    const ctran::allreduce::tree::KernArgs& args,
    const ctran::allreduce::TreeTopology& tree,
    T* dataBuf,
    size_t dataElems,
    int activeBlocks) {
  TreeLaneProgressState<T> state{};
  state.tree = tree;
  state.dataBuf = dataBuf;
  state.dataBytes = dataElems * sizeof(T);
  state.activeBlocks = activeBlocks;

  if (state.dataBytes == 0) {
    state.phase = TreeLanePhase::Done;
    return state;
  }

  state.pipelineWindow = state.dataBytes;
  if (!tree.isLeaf && tree.numChildren > 0) {
    auto* t = args.transports[tree.childRanks[0]].p2p_ibgda;
    state.pipelineWindow = t->pipeline_window(activeBlocks);
  } else if (!tree.isRoot && tree.parentRank >= 0) {
    auto* t = args.transports[tree.parentRank].p2p_ibgda;
    state.pipelineWindow = t->pipeline_window(activeBlocks);
  }
  PIPES_DEVICE_CHECK_MSG(
      state.pipelineWindow != 0, "ctree Phase 2 IB pipeline window is zero");
  state.phase = firstReducePhase(tree);
  return state;
}

template <typename T>
__device__ __forceinline__ comms::pipes::IbgdaSendRecvProgressStatus
progressLaneSend(
    TreeLaneProgressState<T>& state,
    comms::pipes::P2pIbgdaTransportDevice* transport,
    const char* src,
    size_t bytes,
    comms::pipes::ThreadGroup& group,
    const comms::pipes::Timeout& timeout) {
  if (!state.ibOpActive) {
    state.ibOp = transport->init_send_progress(
        group, bytes, state.activeBlocks, 0 /* max_signal_bytes */);
    state.ibOpActive = true;
  }
  return transport->progress_send_once(group, state.ibOp, src, timeout);
}

template <typename T, typename CopyOp = comms::pipes::Memcpy>
__device__ __forceinline__ comms::pipes::IbgdaSendRecvProgressStatus
progressLaneRecv(
    TreeLaneProgressState<T>& state,
    comms::pipes::P2pIbgdaTransportDevice* transport,
    char* dst,
    size_t bytes,
    comms::pipes::ThreadGroup& group,
    const comms::pipes::Timeout& timeout) {
  if (!state.ibOpActive) {
    state.ibOp = transport->init_recv_progress(
        group, bytes, state.activeBlocks, 0 /* max_signal_bytes */);
    state.ibOpActive = true;
  }
  return transport->progress_recv_once<CopyOp>(group, state.ibOp, dst, timeout);
}

template <typename T, int kGroupSize>
__device__ __forceinline__ comms::pipes::IbgdaSendRecvProgressStatus
progressTreeLane(
    const ctran::allreduce::tree::KernArgs& args,
    TreeLaneProgressState<T>& state,
    comms::pipes::ThreadGroup& group,
    const comms::pipes::Timeout& timeout) {
  using comms::pipes::IbgdaSendRecvProgressStatus;

  if (state.phase == TreeLanePhase::Done) {
    return IbgdaSendRecvProgressStatus::Done;
  }

  const size_t window = currentLaneWindowBytes(state);
  char* const data = reinterpret_cast<char*>(state.dataBuf) + state.offsetBytes;

  switch (state.phase) {
    case TreeLanePhase::ReduceLeafSend: {
      auto* parentTransport = args.transports[state.tree.parentRank].p2p_ibgda;
      const auto status = progressLaneSend(
          state, parentTransport, data, window, group, timeout);
      if (status == IbgdaSendRecvProgressStatus::Done) {
        advanceReduceWindow(state);
        return IbgdaSendRecvProgressStatus::Progressed;
      }
      return status;
    }
    case TreeLanePhase::ReduceRecvChild: {
      if (state.childIdx >= state.tree.numChildren) {
        if (state.tree.isRoot) {
          advanceReduceWindow(state);
        } else {
          state.phase = TreeLanePhase::ReduceInternalSend;
        }
        return IbgdaSendRecvProgressStatus::Progressed;
      }
      auto* childTransport =
          args.transports[state.tree.childRanks[state.childIdx]].p2p_ibgda;
      const auto status = progressLaneRecv<T, TreeIbReduceCopy<T>>(
          state, childTransport, data, window, group, timeout);
      if (status == IbgdaSendRecvProgressStatus::Done) {
        state.ibOpActive = false;
        state.childIdx++;
        if (state.childIdx >= state.tree.numChildren) {
          if (state.tree.isRoot) {
            advanceReduceWindow(state);
          } else {
            state.phase = TreeLanePhase::ReduceInternalSend;
          }
        }
        return IbgdaSendRecvProgressStatus::Progressed;
      }
      return status;
    }
    case TreeLanePhase::ReduceInternalSend: {
      auto* parentTransport = args.transports[state.tree.parentRank].p2p_ibgda;
      const auto status = progressLaneSend(
          state, parentTransport, data, window, group, timeout);
      if (status == IbgdaSendRecvProgressStatus::Done) {
        advanceReduceWindow(state);
        return IbgdaSendRecvProgressStatus::Progressed;
      }
      return status;
    }
    case TreeLanePhase::BcastRootSendChild:
    case TreeLanePhase::BcastInternalSendChild: {
      if (state.childIdx >= state.tree.numChildren) {
        advanceBcastWindow(state);
        return IbgdaSendRecvProgressStatus::Progressed;
      }
      auto* childTransport =
          args.transports[state.tree.childRanks[state.childIdx]].p2p_ibgda;
      const auto status =
          progressLaneSend(state, childTransport, data, window, group, timeout);
      if (status == IbgdaSendRecvProgressStatus::Done) {
        state.ibOpActive = false;
        state.childIdx++;
        if (state.childIdx >= state.tree.numChildren) {
          advanceBcastWindow(state);
        }
        return IbgdaSendRecvProgressStatus::Progressed;
      }
      return status;
    }
    case TreeLanePhase::BcastInternalRecv: {
      auto* parentTransport = args.transports[state.tree.parentRank].p2p_ibgda;
      const auto status = progressLaneRecv(
          state, parentTransport, data, window, group, timeout);
      if (status == IbgdaSendRecvProgressStatus::Done) {
        state.ibOpActive = false;
        state.childIdx = 0;
        state.phase = TreeLanePhase::BcastInternalSendChild;
        return IbgdaSendRecvProgressStatus::Progressed;
      }
      return status;
    }
    case TreeLanePhase::BcastLeafRecv: {
      auto* parentTransport = args.transports[state.tree.parentRank].p2p_ibgda;
      const auto status = progressLaneRecv(
          state, parentTransport, data, window, group, timeout);
      if (status == IbgdaSendRecvProgressStatus::Done) {
        advanceBcastWindow(state);
        return IbgdaSendRecvProgressStatus::Progressed;
      }
      return status;
    }
    case TreeLanePhase::Done:
      return IbgdaSendRecvProgressStatus::Done;
  }

  return IbgdaSendRecvProgressStatus::Waiting;
}

// ============================================================================
// Phase 2: Inter-Node IB Dual Tree AllReduce
//
// Tree-0 operates on the first half of each block-owned tile and Tree-1 on the
// second half. One 640-thread block cooperatively drives both tree lanes from
// a progress loop. Each lane uses a disjoint transport group id; if one lane
// is waiting on a remote signal or free transport staging slot, its progress
// call returns immediately and the same block tries the other lane.
//
// Only GPUs with localRank < pMin participate. Non-owners skip this phase.
// Single-node case (nNodes == 1) is a no-op (root-and-leaf in both trees).
// ============================================================================
template <typename T>
__device__ __noinline__ void phase2IbDualTree(
    const ctran::allreduce::tree::KernArgs& args,
    comms::pipes::ThreadGroup& blockGroup) {
  if (args.localRank >= args.pMin) {
    return;
  }

  // Single-node: both trees have isRoot && isLeaf — nothing to do. Phase 1
  // already wrote the segment slice into recvbuff.
  if (args.nNodes <= 1) {
    return;
  }

  const size_t actualElems =
      actualSegElems(args.count, args.segmentElems, args.localRank);
  const auto tile = segmentTile(actualElems * sizeof(T), blockGroup);
  const size_t tileOffsetElems = tile.offsetBytes / sizeof(T);
  const size_t tileElems = tile.bytes / sizeof(T);

  // If the whole segment has at most one element per block, lane 1 would be
  // empty for every block. Compress transport group ids to a single lane for
  // that tiny-message shape; otherwise keep the stable two-lane mapping.
  const bool useSingleLane = actualElems <= static_cast<size_t>(args.numBlocks);
  const int activeIbLanesPerBlock =
      useSingleLane ? 1 : ctran::allreduce::tree::kTreeLanes;

  const size_t halfElems0 = useSingleLane ? tileElems : (tileElems + 1) / 2;
  const size_t halfElems1 = useSingleLane ? 0 : tileElems - halfElems0;
  T* phase2Buf = static_cast<T*>(args.phase2Buf);
  const int activeIbGroups = args.numBlocks * activeIbLanesPerBlock;

  auto lane0Group = blockGroup;
  lane0Group.group_id = blockGroup.group_id * activeIbLanesPerBlock;
  lane0Group.total_groups = static_cast<uint32_t>(activeIbGroups);
  auto lane1Group = blockGroup;
  lane1Group.group_id = blockGroup.group_id * activeIbLanesPerBlock + 1;
  lane1Group.total_groups = static_cast<uint32_t>(activeIbGroups);

  auto lane0 = makeTreeLaneState<T>(
      args,
      args.tree0,
      phase2Buf + tileOffsetElems,
      halfElems0,
      activeIbGroups);
  auto lane1 = makeTreeLaneState<T>(
      args,
      args.tree1,
      phase2Buf + tileOffsetElems + halfElems0,
      halfElems1,
      activeIbGroups);

  comms::pipes::Timeout timeout{};
  bool lane0Active = lane0.phase != TreeLanePhase::Done;
  bool lane1Active = lane1.phase != TreeLanePhase::Done;
  while (lane0Active || lane1Active) {
    if (lane0Active) {
      (void)progressTreeLane<T, ctran::allreduce::tree::kBlockSize>(
          args, lane0, lane0Group, timeout);
      lane0Active = lane0.phase != TreeLanePhase::Done;
    }
    if (lane1Active) {
      (void)progressTreeLane<T, ctran::allreduce::tree::kBlockSize>(
          args, lane1, lane1Group, timeout);
      lane1Active = lane1.phase != TreeLanePhase::Done;
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
// ============================================================================
template <typename T>
__device__ __noinline__ void phase3AllGather(
    const ctran::allreduce::tree::KernArgs& args,
    comms::pipes::ThreadGroup& group) {
  const int localRank = args.localRank;
  const int nLocalRanks = args.nLocalRanks;
  const int pMin = args.pMin;
  const size_t segmentElems = args.segmentElems;
  const size_t segmentBytes = segmentElems * sizeof(T);

  char* recvbuff = static_cast<char*>(args.recvbuff);

  comms::pipes::Timeout timeout{};

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

/**
 * Run the three ctree phases for one logical data tile.
 *
 * A tile is fully owned by one CUDA block. Multi-block launches only add more
 * independent tiles; the algorithm does not require inter-block coordination.
 */
template <typename T>
__device__ __forceinline__ void runAllReduceTree(
    const ctran::allreduce::tree::KernArgs& args,
    comms::pipes::ThreadGroup& group) {
  phase1ReduceScatter<T>(args, group);
  if (args.nLocalRanks > 1) {
    group.sync();
  }
  if (args.nNodes > 1) {
    phase2IbDualTree<T>(args, group);
    if (args.nLocalRanks > 1) {
      group.sync();
    }
  }
  if (args.nLocalRanks > 1) {
    phase3AllGather<T>(args, group);
  }
}

// ============================================================================
// Main kernel: dispatches datatype then runs Phase 1, both Phase 2 tree lanes,
// and Phase 3 for the block-owned tile.
// ============================================================================
__global__
__launch_bounds__(ctran::allreduce::tree::kBlockSize, 1) void ctranKernelAllReduceTree(
    int* /* flag */,
    CtranAlgoDeviceState* /* devState */,
    ctran::allreduce::tree::KernArgs args) {
  auto blockGroup = comms::pipes::make_block_group();
  const int blockId = static_cast<int>(blockIdx.x);
  if (blockId >= args.numBlocks) {
    return;
  }
  auto group = logicalDataGroup(blockGroup, blockId, args.numBlocks);

  if (args.datatype == commFloat32) {
    runAllReduceTree<float>(args, group);
  } else if (args.datatype == commFloat16) {
    runAllReduceTree<__half>(args, group);
  }
}

#endif // ENABLE_PIPES
