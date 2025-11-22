// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stdint.h>
#include <cstddef>
#include <iostream>
#include <tuple>
#include "comms/ctran/commstate/CommStateXDev.h"
#include "comms/ctran/utils/MemFence.h"

// Flags used in stagedSend/stagedRecv
#define CTRAN_ALGO_STEP_RESET (-1)
// Flags used in putNotify/waitNotify
#define CTRAN_ALGO_NOTIFY_RESET (-1)
#define CTRAN_ALGO_NOTIFY_SET (1)

#define CTRAN_ALGO_MAX_THREAD_BLOCKS (64)
#if defined(__HIP_PLATFORM_AMD__)
#define CTRAN_MAX_NVL_PEERS (32)
#else
#ifndef CTRAN_MAX_NVL_PEERS
#define CTRAN_MAX_NVL_PEERS (32)
#endif
#endif

// Statically define the warp size (unlike warpSize, this can be used in
// constexpr expressions)
constexpr int kWarpSize = 32;

struct alignas(16) CtranAlgoDeviceSyncPadded {
  // The step values are padded to take up entire cache lines to reduce latency
  // (different step value updates won't invalidate each others' cache lines)
  static constexpr int kCacheLineSizeBytes = 128;

  int stepOnSameBlockIdx;
  // padding for sync values to be on different cache lines
  int unused[(kCacheLineSizeBytes - sizeof(int)) / sizeof(int)];
};

static_assert(sizeof(CtranAlgoDeviceSyncPadded) == 128);

struct alignas(16) CtranAlgoDeviceSync {
  // Separate flag per thread block to coordinate with remote ranks
  // independently
  CtranAlgoDeviceSyncPadded syncs[CTRAN_ALGO_MAX_THREAD_BLOCKS];
};

struct alignas(16) CtranAlgoDeviceState {
  // Shared buffers for intra-node inter-process communication.
  // Both sync and buf are pointers to device memory.
  // TODO: use two 1D arrays instead of 2D array for allPeerSyncMap
  CtranAlgoDeviceSync* allPeerSyncMap[CTRAN_MAX_NVL_PEERS][CTRAN_MAX_NVL_PEERS];

  // indexed by rank ID
  void* remoteStagingBufsMap[CTRAN_MAX_NVL_PEERS];
  void* localStagingBufsMap[CTRAN_MAX_NVL_PEERS];

  void* peerBcastBufsMap[CTRAN_MAX_NVL_PEERS];
  void* peerAllToAllvDynamicBufsMap[CTRAN_MAX_NVL_PEERS];
  // FIXME: this should become per-block resource.
  void** alltoallvDynamicSendbuffsMap[CTRAN_ALGO_MAX_THREAD_BLOCKS];

  // Comm info copied from ncclComm
  size_t bufSize;
  size_t bcastBufSize;
  bool enableTraceLog;
  ctran::CommStateXDev statex;

  // Enables device sides waits to be cancellable. Host side can set the
  // CtranGpeCmd::kernelFlag_ to `KERNEL_HOST_ABORT` to cancel any currently
  // blocking waits, on e.g. devSyncs / kernelElems / GpeKernelSyncs.
  //
  // This enables host code to perform early termination of the device kernels.
  // (Note Collectives algos must explicitly support early termination as well.)
  //
  // This feature can be enabled per CtranComm instance, and is set when abort
  // is enabled for the CtranComm.
  bool enableCancellableWaits;

  // Collective info optionally specified at collective kernel start time
  // TODO: move opCount update at GPE submit time to cover all collectives
  uint64_t opCount;
};

#define ALLREDUCE_DEQUANT_FUNCMAP(dtype, type, reddtype, redtype, fn) \
  {std::make_tuple(dtype, reddtype, commSum),                         \
   reinterpret_cast<const void*>(&fn<type, redtype, commSum>)},       \
      {std::make_tuple(dtype, reddtype, commProd),                    \
       reinterpret_cast<const void*>(&fn<type, redtype, commProd>)},  \
      {std::make_tuple(dtype, reddtype, commAvg),                     \
       reinterpret_cast<const void*>(&fn<type, redtype, commAvg>)},   \
      {std::make_tuple(dtype, reddtype, commMax),                     \
       reinterpret_cast<const void*>(&fn<type, redtype, commMax>)},   \
  {                                                                   \
    std::make_tuple(dtype, reddtype, commMin),                        \
        reinterpret_cast<const void*>(&fn<type, redtype, commMin>)    \
  }
