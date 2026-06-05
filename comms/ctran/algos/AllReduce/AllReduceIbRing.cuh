// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#if defined(ENABLE_PIPES)

#include "comms/ctran/algos/AllReduce/AllReduceV2Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/utils/commSpecs.h"

namespace ctran::allreduce::hierring {

static constexpr int kRingLanes = 1;
static constexpr int kBlockSize = ctran::allreduce::common::kBlockSize;

struct RingTopology {
  int prevRank{-1};
  int nextRank{-1};
  int nNodes{0};
  int myNodeIdx{0};
};

struct KernArgs {
  ctran::allreduce::common::CommonKernArgs common;
  RingTopology ring;
};

} // namespace ctran::allreduce::hierring

__global__ void ctranKernelAllReduceHierarchicalRing(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allreduce::hierring::KernArgs args);

#endif // ENABLE_PIPES
