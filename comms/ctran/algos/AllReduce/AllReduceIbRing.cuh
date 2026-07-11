// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace ctran::allreduce::hierring {

// Host-safe ring topology POD shared by the host launcher (AllReduceIbRing.cc)
// and the device KernArgs below. Kept outside the ENABLE_PRIMS guard so it is
// visible on host-only builds where ENABLE_PRIMS is undefined (e.g. AMD/HIP),
// mirroring how ctran::allreduce::TreeTopology lives in Types.h.
struct RingTopology {
  int prevRank{-1};
  int nextRank{-1};
  int nNodes{0};
  int myNodeIdx{0};
};

} // namespace ctran::allreduce::hierring

#if defined(ENABLE_PRIMS)

#include "comms/ctran/algos/AllReduce/AllReduceFusedTypes.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/utils/commSpecs.h"

namespace ctran::allreduce::hierring {

static constexpr int kBlockSize = ctran::allreduce::common::kBlockSize;

struct KernArgs {
  ctran::allreduce::common::CommonKernArgs common;

  /**
   * Per-NVL-variant device-state slot. Empty for NvlDirect; the orchestrator
   * dispatches the NVL phases through `nvl::direct::Ops`.
   * `[[no_unique_address]]` keeps `sizeof(KernArgs)` unchanged while empty.
   */
  [[no_unique_address]] ctran::allreduce::nvl::direct::ExtraArgs nvl;

  RingTopology ring;

  /**
   * Fixed IB send/recv group reservation (the transport `active_blocks`), kept
   * independent of `common.numBlocks` so the staging-buffer layout is stable
   * across launches. Set on the host from `fused::get_num_block_cap()` (the
   * same clamped `NCCL_CTRAN_MAX_NBLOCKS` source that caps `numBlocks`); see
   * AllReduceIbRing.cc.
   */
  int ibSendRecvGroups{0};
};

} // namespace ctran::allreduce::hierring

__global__ void ctranKernelAllReduceHierarchicalRing(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allreduce::hierring::KernArgs args);

#endif // ENABLE_PRIMS
