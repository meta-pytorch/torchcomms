// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comm.h"
#include "meta/algoconf/InfoExt.h"
#include "meta/collectives/PatAvgHelper.h" // reuse computePatAvgChannelsAndWarps

namespace ncclx {

// Set up ncclInfoExt for quantized ReduceScatter override.
// Call at ncclReduceScatterQuantize entry to force PAT algorithm + SIMPLE
// protocol, following the same pattern as setupPatAvgInfoExt() in
// PatAvgHelper.h.
inline algoconf::ncclInfoExt setupQuantizeInfoExt(
    struct ncclComm* comm,
    size_t nBytes,
    uint64_t* seedPtr,
    ncclDataType_t transportType) {
  int nMaxChannels = 0, nWarps = 0;
  computePatAvgChannelsAndWarps(comm, nBytes, &nMaxChannels, &nWarps);
  return algoconf::ncclInfoExt(
      NCCL_ALGO_PAT,
      NCCL_PROTO_SIMPLE,
      nMaxChannels,
      nWarps,
      /*opDev=*/std::nullopt,
      /*quantizeRandomSeedPtr=*/seedPtr);
}

} // namespace ncclx
