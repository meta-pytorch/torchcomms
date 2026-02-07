// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comm.h"
#include "device.h"
#include "meta/algoconf/InfoExt.h"

namespace ncclx {

// Compute nMaxChannels and nWarps for PAT algorithm with SIMPLE protocol.
// This mirrors the computation in topoGetAlgoInfo() but can be called early
// to enable per-communicator PAT AVG control via ncclInfoExt.
inline void computePatAvgChannelsAndWarps(
    struct ncclComm* comm,
    size_t nBytes,
    int* outNMaxChannels,
    int* outNWarps) {
  int nc = comm->nChannels;
  int nt = comm->maxThreads[NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE];
  int threadThreshold =
      comm->threadThresholds[NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE];

  // Reduce channels based on data size (same logic as topoGetAlgoInfo)
  while (nBytes < static_cast<size_t>(nc * nt * threadThreshold) && nc >= 2) {
    nc--;
  }

  // PAT always uses max threads
  nt = NCCL_MAX_NTHREADS;

  *outNMaxChannels = nc;
  *outNWarps = nt / WARP_SIZE;
}

// Set up ncclInfoExt for PAT AVG override.
// Call at ncclReduceScatter entry when comm->usePatAvg_ && op == ncclAvg.
// This populates all required fields so that algoInfoMayOverride() will
// apply the override and skip algorithm selection.
inline void setupPatAvgInfoExt(
    struct ncclComm* comm,
    size_t nBytes,
    algoconf::ncclInfoExt* ext) {
  ext->algorithm = NCCL_ALGO_PAT;
  ext->protocol = NCCL_PROTO_SIMPLE;

  // Set up ncclDevPatAvg with nRanks as the divisor
  ext->opDev.op = ncclDevPatAvg;
  ext->opDev.scalarArg = static_cast<uint64_t>(comm->nRanks);
  ext->opDev.scalarArgIsPtr = false;
  ext->opDevSet = true;

  computePatAvgChannelsAndWarps(comm, nBytes, &ext->nMaxChannels, &ext->nWarps);
}

} // namespace ncclx
