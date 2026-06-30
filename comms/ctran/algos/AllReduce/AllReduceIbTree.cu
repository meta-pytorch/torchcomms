// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#if defined(ENABLE_PRIMS)

#include <cuda_fp16.h>

#include <cstdint>

#include "comms/ctran/algos/AllReduce/AllReduceFused.cuh"
#include "comms/ctran/algos/AllReduce/AllReduceFusedCommon.cuh"
#include "comms/ctran/algos/AllReduce/AllReduceIbTree.cuh"
#include "comms/ctran/algos/AllReduce/AllReduceNvlDirect.cuh"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/Transport.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

// Shared device utilities (tileReduce, IbReduceCopy, logicalDataGroup,
// segmentTile, actualSegElems, pipelineStepBytes, ...) live in
// AllReduceFusedCommon.cuh (ctran::allreduce::common). The NVL phases live in
// AllReduceNvlDirect.cuh (ctran::allreduce::nvl::direct) and the
// phase-sequencing orchestrator in AllReduceFused.cuh
// (ctran::allreduce::fused).
using namespace ctran::allreduce::common;

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

  // Safe to query a single tree edge: pipeline_window() is comm-uniform, not
  // per-peer. If per-peer buffer sizes diverge, active edges must agree.
  state.pipelineWindow = state.dataBytes;
  if (!tree.isLeaf && tree.numChildren > 0) {
    auto* t = args.common.transports[tree.childRanks[0]].p2p_ib.ibgda;
    state.pipelineWindow = t->pipeline_window(activeBlocks);
  } else if (!tree.isRoot && tree.parentRank >= 0) {
    auto* t = args.common.transports[tree.parentRank].p2p_ib.ibgda;
    state.pipelineWindow = t->pipeline_window(activeBlocks);
  }
  PIPES_DEVICE_CHECK_MSG(
      state.pipelineWindow != 0, "ctree Phase 2 IB pipeline window is zero");
  state.phase = firstReducePhase(tree);
  return state;
}

template <typename T>
__device__ __forceinline__ comms::prims::IbgdaSendRecvProgressStatus
progressLaneSend(
    TreeLaneProgressState<T>& state,
    comms::prims::P2pIbgdaTransportDevice* transport,
    const char* src,
    size_t bytes,
    comms::prims::ThreadGroup& group,
    const comms::prims::Timeout& timeout) {
  if (!state.ibOpActive) {
    transport->init_send_progress(
        group, bytes, state.activeBlocks, 0 /* max_signal_bytes */);
    state.ibOpActive = true;
  }
  return transport->progress_send_once(
      group, src, bytes, state.activeBlocks, 0 /* max_signal_bytes */, timeout);
}

template <typename T, typename CopyOp = comms::prims::Memcpy>
__device__ __forceinline__ comms::prims::IbgdaSendRecvProgressStatus
progressLaneRecv(
    TreeLaneProgressState<T>& state,
    comms::prims::P2pIbgdaTransportDevice* transport,
    char* dst,
    size_t bytes,
    comms::prims::ThreadGroup& group,
    const comms::prims::Timeout& timeout) {
  if (!state.ibOpActive) {
    transport->init_recv_progress(
        group, bytes, state.activeBlocks, 0 /* max_signal_bytes */);
    state.ibOpActive = true;
  }
  return transport->progress_recv_once<CopyOp>(
      group, dst, bytes, state.activeBlocks, 0 /* max_signal_bytes */, timeout);
}

template <typename T, int kGroupSize>
__device__ __forceinline__ comms::prims::IbgdaSendRecvProgressStatus
progressTreeLane(
    const ctran::allreduce::tree::KernArgs& args,
    TreeLaneProgressState<T>& state,
    comms::prims::ThreadGroup& group,
    const comms::prims::Timeout& timeout) {
  using comms::prims::IbgdaSendRecvProgressStatus;

  if (state.phase == TreeLanePhase::Done) {
    return IbgdaSendRecvProgressStatus::Done;
  }

  const size_t window = currentLaneWindowBytes(state);
  char* const data = reinterpret_cast<char*>(state.dataBuf) + state.offsetBytes;

  switch (state.phase) {
    case TreeLanePhase::ReduceLeafSend: {
      auto* parentTransport =
          args.common.transports[state.tree.parentRank].p2p_ib.ibgda;
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
          args.common.transports[state.tree.childRanks[state.childIdx]]
              .p2p_ib.ibgda;
      const auto status = progressLaneRecv<T, IbReduceCopy<T>>(
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
      auto* parentTransport =
          args.common.transports[state.tree.parentRank].p2p_ib.ibgda;
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
          args.common.transports[state.tree.childRanks[state.childIdx]]
              .p2p_ib.ibgda;
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
      auto* parentTransport =
          args.common.transports[state.tree.parentRank].p2p_ib.ibgda;
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
      auto* parentTransport =
          args.common.transports[state.tree.parentRank].p2p_ib.ibgda;
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
    comms::prims::ThreadGroup& blockGroup) {
  if (args.common.localRank >= args.common.pMin) {
    return;
  }

  // Single-node: both trees have isRoot && isLeaf — nothing to do. Phase 1
  // already wrote the segment slice into recvbuff.
  if (args.common.nNodes <= 1) {
    return;
  }

  const size_t actualElems = actualSegElems(
      args.common.count, args.common.segmentElems, args.common.localRank);
  const auto tile = segmentTile(
      actualElems * sizeof(T), blockGroup, args.common.blockTileBytes);
  const size_t tileOffsetElems = tile.offsetBytes / sizeof(T);
  const size_t tileElems = tile.bytes / sizeof(T);

  const size_t lane0PartitionElems = compute_aligned_tile_parition_size(
      tileElems, sizeof(T), ctran::allreduce::tree::kTreeLanes);
  const size_t halfElems0 =
      lane0PartitionElems < tileElems ? lane0PartitionElems : tileElems;
  const size_t halfElems1 = tileElems - halfElems0;
  T* phase2Buf = static_cast<T*>(args.common.phase2Buf);
  const int activeIbGroups = args.ibSendRecvGroups;
  PIPES_DEVICE_CHECK_MSG(
      activeIbGroups >=
          args.common.numBlocks * ctran::allreduce::tree::kTreeLanes,
      "ctree Phase 2 fixed IB group geometry is too small");

  auto lane0Group = blockGroup;
  lane0Group.group_id =
      blockGroup.group_id * ctran::allreduce::tree::kTreeLanes;
  lane0Group.total_groups = static_cast<uint32_t>(activeIbGroups);
  auto lane1Group = blockGroup;
  lane1Group.group_id =
      blockGroup.group_id * ctran::allreduce::tree::kTreeLanes + 1;
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

  comms::prims::Timeout timeout{};
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
// Main kernel: dispatches datatype then runs Phase 1, both Phase 2 tree lanes,
// and Phase 3 for the block-owned tile.
// ============================================================================
__global__
__launch_bounds__(ctran::allreduce::tree::kBlockSize, 1) void ctranKernelAllReduceTree(
    int* /* flag */,
    CtranAlgoDeviceState* /* devState */,
    ctran::allreduce::tree::KernArgs args) {
  auto blockGroup = comms::prims::make_block_group();
  const int blockId = static_cast<int>(blockIdx.x);
  if (blockId >= args.common.numBlocks) {
    return;
  }
  auto group = logicalDataGroup(blockGroup, blockId, args.common.numBlocks);

  if (args.common.datatype == commFloat32) {
    ctran::allreduce::fused::
        runAllReduceFused<float, ctran::allreduce::nvl::direct::Ops>(
            args.common, group, [&](comms::prims::ThreadGroup& phaseGroup) {
              phase2IbDualTree<float>(args, phaseGroup);
            });
  } else if (args.common.datatype == commFloat16) {
    ctran::allreduce::fused::
        runAllReduceFused<__half, ctran::allreduce::nvl::direct::Ops>(
            args.common, group, [&](comms::prims::ThreadGroup& phaseGroup) {
              phase2IbDualTree<__half>(args, phaseGroup);
            });
  }
}

#endif // ENABLE_PRIMS
