// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Phase-sequencing orchestrator for the fused AllReduce framework. Namespace:
// ctran::allreduce::fused. Sequences the NVL phases (supplied by an NvlOps
// policy, e.g. ctran::allreduce::nvl::direct::Ops) around an algorithm-specific
// cross-node Phase 2 callable. This header is variant-agnostic: it does NOT
// include any NvlOps implementation — the policy is a template parameter the
// caller supplies, so adding an NVL or IB variant never touches this file.

#pragma once

#if defined(ENABLE_PRIMS)

#include "comms/ctran/algos/AllReduce/AllReduceFusedTypes.h"
#include "comms/prims/core/ThreadGroup.cuh"

namespace ctran::allreduce::fused {

// ============================================================================
// Fused orchestrator
//
// Runs the NVL ReduceScatter (Phase 1) and NVL AllGather (Phase 3) supplied by
// `NvlOps`, around an algorithm-specific cross-node IB phase (Phase 2), for one
// logical data tile owned by `group`. `phase2` is any callable invoked as
// `phase2(group)`; it must reduce each owner segment across nodes and leave the
// globally reduced segment in `args.phase2Buf` (the Phase-2 contract).
//
// `NvlOps` is a policy struct exposing `phase1<T>(args, group)` and
// `phase3<T>(args, group)` (see nvl::direct::Ops). Both the policy and `phase2`
// are template parameters so the variant-specific phases inline and the kernel
// keeps a single `__launch_bounds__` register budget.
//
// Phase-transition `group.sync()`s and the degenerate-topology skips
// (nLocalRanks <= 1 → no NVL phases; nNodes <= 1 → no IB phase) are identical
// across algorithms and owned here. Correctness does not require inter-block
// synchronization.
// ============================================================================
template <typename T, typename NvlOps, typename Phase2Fn>
__device__ __forceinline__ void runAllReduceFused(
    const common::CommonKernArgs& args,
    comms::prims::ThreadGroup& group,
    Phase2Fn&& phase2) {
  NvlOps::template phase1<T>(args, group);
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
    NvlOps::template phase3<T>(args, group);
  }
}

} // namespace ctran::allreduce::fused

#endif // ENABLE_PRIMS
