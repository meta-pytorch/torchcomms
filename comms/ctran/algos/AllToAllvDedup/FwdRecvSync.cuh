// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/algos/AllToAllvDedup/WorkerGroupDev.cuh"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/SpscP2pSync.h"
#include "comms/ctran/algos/common/SpscP2pSyncDev.cuh"

namespace ctran::alltoallvdedup {
using namespace ctran::algos;

__device__ __forceinline__ FwdRecvSync* getFwdRecvSync(
    const ExecKernArgs& args,
    const int owner, // owner local rank of the sync
    const int peer, // peer local rank that the sync is for
    const int step,
    const bool kIsIntra = false) {
  const auto& config = args.config;
  const auto& kSync = args.kSync;
  void* ptr = kIsIntra ? kSync.remIntraFwdRecvSyncs[owner]
                       : kSync.remFwdRecvSyncs[owner];
  return ptrElemOffset<FwdRecvSync>(
      ptr, peer * config.tmpNumChunks + getTmpChunkIdx(config, step));
}

__device__ __forceinline__ void fwdRecvSyncComplete(FwdRecvSync* sync) {
  SpscP2pSyncDev::complete(&sync->spsc);
}

__device__ __forceinline__ void fwdRecvSyncWaitReady(FwdRecvSync* sync) {
  SpscP2pSyncDev::waitReady(&sync->spsc);
}

__device__ __forceinline__ bool fwdRecvSyncCheckReady(FwdRecvSync* sync) {
  return SpscP2pSyncDev::checkReady(&sync->spsc);
}

__device__ __forceinline__ void fwdRecvSyncPost(
    FwdRecvSync* sync,
    const int step) {
  SpscP2pSyncDev::post(&sync->spsc, step);
}

__device__ __forceinline__ void fwdRecvSyncWaitPost(
    FwdRecvSync* sync,
    const int step) {
  SpscP2pSyncDev::waitPost(&sync->spsc, step);
}

__device__ __forceinline__ bool fwdRecvSyncCheckPost(
    FwdRecvSync* sync,
    const int step) {
  return SpscP2pSyncDev::checkPost(&sync->spsc, step);
}
} // namespace ctran::alltoallvdedup
