// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#if defined(ENABLE_PRIMS)

#include "comms/ctran/algos/AllReduce/AllReduceFusedTypes.h"
#include "comms/ctran/algos/AllReduce/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/topo/TreeConstants.h"
#include "comms/utils/commSpecs.h"

namespace comms::prims {
struct Transport;
}

namespace ctran::allreduce::tree {

/** Number of independent inter-node tree lanes in the dual-tree phase. */
static constexpr int kTreeLanes = 2;

/** CUDA threads per block used by the ctree kernel. */
static constexpr int kBlockSize = ctran::allreduce::common::kBlockSize;

/**
 * Device kernel arguments for the Prims-backed CTRAN tree AllReduce.
 *
 * The topology-agnostic fields (buffers, sizes, transports, NVL phase inputs)
 * live in the shared `common::CommonKernArgs`; only the tree-specific dual-tree
 * topology is layered on here.
 */
struct KernArgs {
  /** Shared topology-agnostic kernel arguments (Phase 1 / Phase 3 + entry). */
  ctran::allreduce::common::CommonKernArgs common;

  /**
   * Per-NVL-variant device-state slot. Empty for NvlDirect; the orchestrator
   * dispatches the NVL phases through `nvl::direct::Ops`.
   * `[[no_unique_address]]` keeps `sizeof(KernArgs)` unchanged while empty.
   */
  [[no_unique_address]] ctran::allreduce::nvl::direct::ExtraArgs nvl;

  /** Inter-node tree used for the first half of each data partition. */
  ctran::allreduce::TreeTopology tree0;
  /** Inter-node tree used for the second half of each data partition. */
  ctran::allreduce::TreeTopology tree1;
  /** Fixed IB send/recv group count used for transport staging geometry. */
  int ibSendRecvGroups{0};
};

} // namespace ctran::allreduce::tree

/**
 * CUDA kernel entry point for CTRAN tree AllReduce.
 *
 * Each block owns one tile partition and runs Phase 1, both Phase 2 tree
 * lanes, and Phase 3 for that tile. Phase 2 uses one full-block work group to
 * poll and progress the two disjoint IB tree lanes cooperatively.
 * Multi-block launches are independent tiling for performance; correctness
 * does not require inter-block sync.
 */
__global__ void ctranKernelAllReduceTree(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allreduce::tree::KernArgs args);

#endif // ENABLE_PRIMS
