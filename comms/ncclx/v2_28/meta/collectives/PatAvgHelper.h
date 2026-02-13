// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comm.h"
#include "device.h"
#include "meta/algoconf/InfoExt.h"

namespace ncclx {

// PatAvg is restricted to types with enough exponent range to avoid
// overflow during intermediate sum accumulation. fp16 and fp8 types are
// excluded.
inline bool isPatAvgSupportedType(ncclDataType_t dt) {
  switch (dt) {
    case ncclFloat16:
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
      return false;
    default:
      return true;
  }
}

// Compute nMaxChannels and nWarps for PAT algorithm with SIMPLE protocol.
// This mirrors the channel-reduction logic in topoGetAlgoInfo() but uses
// static constants because baseline tuning (ncclTopoTuneModel) does not
// initialize maxThreads[NCCL_ALGO_PAT], leaving it at 0 and making the
// channel reduction loop dead code when using comm values directly.
inline void computePatAvgChannelsAndWarps(
    struct ncclComm* comm,
    size_t nBytes,
    int* outNMaxChannels,
    int* outNWarps) {
  int nc = comm->nChannels;
  int nt = NCCL_MAX_NTHREADS;
  int threadThreshold = NCCL_SIMPLE_THREAD_THRESHOLD;

  // Reduce channels based on data size (same logic as topoGetAlgoInfo)
  while (nBytes < static_cast<size_t>(nc * nt * threadThreshold) && nc >= 2) {
    nc--;
  }

  *outNMaxChannels = nc;
  *outNWarps = nt / WARP_SIZE;
}

// Set up ncclInfoExt for PAT AVG override.
// Call at ncclReduceScatter entry when comm->usePatAvg_ && op == ncclAvg.
// Returns a fully constructed ncclInfoExt so that algoInfoMayOverride() will
// apply the override and skip algorithm selection.
inline algoconf::ncclInfoExt setupPatAvgInfoExt(
    struct ncclComm* comm,
    size_t nBytes,
    ncclDataType_t datatype) {
  int nMaxChannels = 0, nWarps = 0;
  computePatAvgChannelsAndWarps(comm, nBytes, &nMaxChannels, &nWarps);

  bool isSigned =
      (datatype == ncclInt8 || datatype == ncclInt32 || datatype == ncclInt64);
  ncclDevRedOpFull opDev{};
  opDev.op = ncclDevPatSumPostDiv;
  // Encode (divisor << 1 | isSigned): signed int types map to unsigned kernels
  // via equivalent_primary(), so the device-side divide() needs isSigned to
  // reinterpret the accumulated sum as signed before dividing.
  opDev.scalarArg = (static_cast<uint64_t>(comm->nRanks) << 1) |
      static_cast<uint64_t>(isSigned);
  opDev.scalarArgIsPtr = false;

  return algoconf::ncclInfoExt(
      NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, nMaxChannels, nWarps, opDev);
}

} // namespace ncclx
