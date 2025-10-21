// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>
#include <cstddef>

#include "comms/ctran/algos/AllToAllvDedup/CommonDev.h"
#include "comms/ctran/algos/AllToAllvDedup/FwdGroupSyncDev.cuh"
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/algos/AllToAllvDedup/WorkerSyncDev.cuh"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/commstate/CommStateXDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"

namespace ctran::alltoallvdedup {
using namespace ctran::algos;

__device__ __forceinline__ bool checkWarpRole(
    unsigned int warpId,
    const PrepareKernRole role) {
  return warpId == (unsigned int)role;
}

__device__ __forceinline__ void computeOffsetsFromCounts(
    const size_t nLocalRanks,
    const size_t nNodes,
    const size_t* numBlocks,
    size_t* offsets,
    size_t* totalNumBlocks,
    size_t numRecvBuckets,
    size_t nElemPerBlock = 1) {
  // traverse numRecvBuckets array so that the buckets on each rank are
  // contiguous in memory
  size_t sum = 0;
  for (auto bucket = 0; bucket < numRecvBuckets; bucket++) {
    for (auto localRank = 0; localRank < nLocalRanks; localRank++) {
      for (auto node = 0; node < nNodes; node++) {
        auto rank = node * nLocalRanks + localRank;
        offsets[rank * numRecvBuckets + bucket] = sum * nElemPerBlock;
        sum += numBlocks[rank * numRecvBuckets + bucket];
      }
    }
  }
  *totalNumBlocks = sum;
}

__device__ __forceinline__ void gatherNumRecvCounts(PrepareKernArgs& args) {
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nNodes = statex->nNodes();
  const auto myLocalRank = statex->localRank();
  const auto myNode = statex->node();
  const auto nRanks = statex->nRanks();

  // nNodes * nNodes * (args.pArgs.numRecvBuckets * nLocalRanks + nLocalRanks +
  // 1)
  auto tmpNumSendBlocks = args.tmpNumSendBlocksBuffH;

  // Each thread updates the numRecvBlocks from a sendRank to each localRank
  // E.g., assuming nLocalRanks = 4, nNodes = 2, numRecvBuckets = 2 every rank
  // hosts numRecvBlocks[2][4][2];
  // - Each local forwarding rank i updates numRecvBlocks[][i][] for all local
  //   ranks, in total nLocalRanks * nNodes * numRecvBuckets = nRanks *
  //   numRecvBuckets = 16 P2P updates
  // - The total number of buckets = 16 P2P updates is executed by different
  //   thread on rank i
  const auto nLocalBuckets = args.pArgs.numRecvBuckets * nLocalRanks;
  const auto nUpdates = nNodes * nLocalBuckets;
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = gtIdx; i < nUpdates; i += blockDim.x * gridDim.x) {
    const auto sendNode = i / nLocalBuckets;
    const auto sendRank = statex->localRankToRank(myLocalRank, sendNode);
    const auto sendBucket = sendRank * args.pArgs.numRecvBuckets +
        (i & (args.pArgs.numRecvBuckets - 1));
    const auto recvLocalBucket = i & (nLocalBuckets - 1);
    const auto recvLocalRank = bucketToRank(args.pArgs, recvLocalBucket);

    const auto sendNodeOffset =
        sendNode * nNodes * (nLocalBuckets + nLocalRanks + 1);
    const auto rankOffsetInSendNode =
        myNode * (nLocalBuckets + nLocalRanks + 1) + recvLocalBucket;

    // H2D load
    const auto numSendBlocks =
        tmpNumSendBlocks[sendNodeOffset + rankOffsetInSendNode];
    // P2P store
    auto remNumRecvBlocks = args.tmpRemNumRecvBlocksBuffs[recvLocalRank];
    remNumRecvBlocks[sendBucket] = numSendBlocks;
  }

  for (int i = gtIdx; i < nRanks; i += blockDim.x * gridDim.x) {
    const auto sendNode = i / nLocalRanks;
    const auto sendNodeOffset =
        sendNode * nNodes * (nLocalBuckets + nLocalRanks + 1);
    const auto recvLocalRank = i & (nLocalRanks - 1);
    const auto rankOffsetInSendNode =
        myNode * (nLocalBuckets + nLocalRanks + 1) + nLocalBuckets +
        recvLocalRank;

    // H2D load
    const auto numSendBlocks =
        tmpNumSendBlocks[sendNodeOffset + rankOffsetInSendNode];
    // P2P store
    const auto sendRank = statex->localRankToRank(myLocalRank, sendNode);
    auto remNumRecvBlocks = args.tmpRemNumRecvBlocksBuffs[recvLocalRank];
    remNumRecvBlocks[nUpdates + sendRank] = numSendBlocks;
  }

  // Wait for all P2P remNumRecvBlocks updates from intra-node ranks
  barrier(myLocalRank, nLocalRanks);

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "gathered numRecvBlocks %p: %ld %ld %ld %ld\n",
        args.tmpNumRecvBlocksBuff,
        args.tmpNumRecvBlocksBuff[0],
        args.tmpNumRecvBlocksBuff[1],
        args.tmpNumRecvBlocksBuff[2],
        args.tmpNumRecvBlocksBuff[3]);
  }
}

__device__ void __inline__ copyBlockRecvBuckets(PrepareKernArgs& args) {
  PersistArgs& pArgs = args.pArgs;
  const auto count = pArgs.totalNumSendBlocks * pArgs.blockNumRecvBuckets;
  const auto myNode = statex->node();
  auto dData = args.blockRecvBucketsH + myNode * count;
  auto sData = args.prepareArgs.blockRecvBuckets;

  copy<int>(dData, sData, count, blockIdx.x, gridDim.x);

  __syncthreads();
  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "copied blockRecvBuckets %p: %d %d %d %d -> blockRecvBucketsH %p: %d %d ... %d %d, count %ld (%ld * %ld)\n",
        sData,
        sData[0],
        sData[1],
        sData[count - 2],
        sData[count - 1],
        dData,
        dData[0],
        dData[1],
        dData[count - 2],
        dData[count - 1],
        count,
        pArgs.totalNumSendBlocks,
        pArgs.blockNumRecvBuckets);
  }
}

__device__ void __inline__ copyNumSendBlocks(PrepareKernArgs& args) {
  auto& prepareArgs = args.prepareArgs;
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nRanks = statex->nRanks();
  const auto nNodes = statex->nNodes();
  const auto myNode = statex->node();

  auto dPtr = prepareArgs.numSendBlocks;
  const auto nLocalBuckets = args.pArgs.numRecvBuckets * nLocalRanks;
  auto sPtr = args.tmpNumSendBlocksBuffH +
      myNode * nNodes * (nLocalBuckets + nLocalRanks + 1);

  const auto laneId = threadIdx.x & (kWarpSize - 1);

  for (int i = laneId; i < nRanks; i += kWarpSize) {
    const auto node = i / nLocalRanks;
    const auto localRank = i & (nLocalRanks - 1);
    dPtr[i] = 0;
    for (int j = 0; j < args.pArgs.numRecvBuckets; j++) {
      const auto localBucket = localRank * args.pArgs.numRecvBuckets + j;
      const auto sIdx = node * (nLocalBuckets + nLocalRanks + 1) + localBucket;
      dPtr[i] += sPtr[sIdx];
    }
  }

  syncwarp();
  if (laneId == 0) {
    const auto warpId = threadIdx.x / kWarpSize;
    CTRAN_DEV_TRACE(
        "warpId %d copied tmpNumSendBlocksBuffH %p -> numSendBlocks %p: %ld %ld\n",
        warpId,
        sPtr,
        dPtr,
        dPtr[0],
        dPtr[1]);
  }
}

__device__ void __inline__ copyNumForwardBlocks(PrepareKernArgs& args) {
  auto& prepareArgs = args.prepareArgs;
  auto dPtr = prepareArgs.numForwardBlocks;
  const auto sPtr = args.numForwardBlocksH;

  const auto count = statex->nRanks();
  const auto laneId = threadIdx.x & (kWarpSize - 1);

  copyWarp<size_t>(dPtr, sPtr, count);

  syncwarp();
  if (laneId == 0) {
    const auto warpId = threadIdx.x / kWarpSize;
    CTRAN_DEV_TRACE(
        "warpId %d copied numForwardBlocksH %p -> numForwardBlocks %p: %ld %ld\n",
        warpId,
        sPtr,
        dPtr,
        dPtr[0],
        dPtr[1]);
  }
}

__device__ void __inline__ copyNumRecvBlocks(PrepareKernArgs& args) {
  const auto myLocalRank = statex->localRank();
  const auto nRanks = statex->nRanks();
  auto dPtr = args.prepareArgs.numRecvBlocks;
  const auto sPtr = args.tmpNumRecvBlocksBuff;

  const auto count = nRanks * args.pArgs.numRecvBuckets;
  const auto laneId = threadIdx.x & (kWarpSize - 1);

  copyWarp<size_t>(dPtr, sPtr, count);

  syncwarp();
  if (laneId == 0) {
    const auto warpId = threadIdx.x / kWarpSize;
    CTRAN_DEV_TRACE(
        "copied warpId %d tmpNumRecvBlocksBuff %p (%p + %d) -> numRecvBlocks %p: %ld %ld\n",
        warpId,
        sPtr,
        args.tmpNumRecvBlocksBuff,
        myLocalRank * nRanks,
        dPtr,
        dPtr[0],
        dPtr[1]);
  }
}

__device__ void __inline__ copyNumRecvBlocksH(PrepareKernArgs& args) {
  const auto nRanks = statex->nRanks();
  auto dPtr = args.tmpNumRecvBlocksBuffH;
  const auto sPtr = args.tmpNumRecvBlocksBuff;

  const auto count = nRanks * args.pArgs.numRecvBuckets + nRanks;
  const auto laneId = threadIdx.x & (kWarpSize - 1);

  copyWarp<size_t>(dPtr, sPtr, count);
  syncwarp();
  if (laneId == 0) {
    const auto warpId = threadIdx.x / kWarpSize;
    CTRAN_DEV_TRACE(
        "copied warpId %d tmpNumRecvBlocksBuff %p (%p + %d) -> tmpNumRecvBlocksBuffH %p: %ld %ld %ld %ld\n",
        warpId,
        sPtr,
        args.tmpNumRecvBlocksBuff,
        count,
        dPtr,
        dPtr[0],
        dPtr[1],
        dPtr[2],
        dPtr[3]);
  }
}

__device__ __forceinline__ void gatherLocalOutputSplits(PrepareKernArgs& args) {
  const auto nLocalRanks = statex->nLocalRanks();
  const auto myLocalRank = statex->localRank();

  // need volatile since accessing with different threads?
  volatile auto tmpLocalOutputSplits = args.tmpLocalOutputSplits;
  auto tmpLocalOutputSplitsH = args.tmpLocalOutputSplitsH;

  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const auto nUpdates = nLocalRanks * nLocalRanks;
  for (int i = gtIdx; i < nUpdates; i += blockDim.x * gridDim.x) {
    const auto peerRank = i / nLocalRanks;
    const auto offset = myLocalRank * nLocalRanks;
    const auto localOffset = i & (nLocalRanks - 1);

    // H2D load
    const auto sendCount = tmpLocalOutputSplitsH[offset + localOffset];
    CTRAN_DEV_TRACE("sendCount: %d\n", sendCount);

    // P2P store
    auto tmpRemLocalOutputSplits = args.tmpRemLocalOutputSplits[peerRank];
    // fixme branching
    if (peerRank == myLocalRank) {
      tmpRemLocalOutputSplits = tmpLocalOutputSplits;
    }
    tmpRemLocalOutputSplits[offset + localOffset] = sendCount;
  }
}

__device__ __forceinline__ void copyLocalOutputSplits(PrepareKernArgs& args) {
  const auto nLocalRanks = statex->nLocalRanks();

  // need volatile since accessing with different threads?
  volatile auto tmpLocalOutputSplits = args.tmpLocalOutputSplits;
  auto tmpLocalOutputSplitsH = args.tmpLocalOutputSplitsH;

  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  // copy to host pinned
  for (int i = gtIdx; i < nLocalRanks * nLocalRanks;
       i += blockDim.x * gridDim.x) {
    tmpLocalOutputSplitsH[i] = tmpLocalOutputSplits[i];
  }

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "gathered tmpLocalOutputSplits %p: %d %d %d %d\n",
        args.tmpLocalOutputSplits,
        args.tmpLocalOutputSplits[0],
        args.tmpLocalOutputSplits[1],
        args.tmpLocalOutputSplits[2],
        args.tmpLocalOutputSplits[3]);
  }
}

__device__ __forceinline__ void gatherRankBitmaps(PrepareKernArgs& args) {
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nNodes = statex->nNodes();
  const auto myLocalRank = statex->localRank();

  // need volatile since accessing with different threads?
  volatile auto tmpRankBitmaps = args.tmpRankBitmaps;
  auto tmpRankBitmapsH = args.tmpRankBitmapsH;

  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const auto countPerRank =
      nLocalRanks * nNodes * args.pArgs.totalNumSendBlocks;
  const auto nUpdates = nLocalRanks * countPerRank;
  for (int i = gtIdx; i < nUpdates; i += blockDim.x * gridDim.x) {
    const auto peerRank = i / countPerRank;
    const auto offset = myLocalRank * countPerRank;
    const auto localOffset = i & (countPerRank - 1);

    // H2D load
    const auto bitmap = tmpRankBitmapsH[offset + localOffset];

    // P2P store
    auto tmpRemRankBitmaps = args.tmpRemRankBitmaps[peerRank];
    // fixme branching
    if (peerRank == myLocalRank) {
      tmpRemRankBitmaps = tmpRankBitmaps;
    }
    tmpRemRankBitmaps[offset + localOffset] = bitmap;
  }
}

__device__ __forceinline__ void copyRankBitmaps(PrepareKernArgs& args) {
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nNodes = statex->nNodes();

  // need volatile since accessing with different threads?
  volatile auto tmpRankBitmaps = args.tmpRankBitmaps;
  auto tmpRankBitmapsH = args.tmpRankBitmapsH;

  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const auto countPerRank =
      nLocalRanks * nNodes * args.pArgs.totalNumSendBlocks;
  const auto nUpdates = nLocalRanks * countPerRank;

  // copy to host pinned
  for (int i = gtIdx; i < nUpdates; i += blockDim.x * gridDim.x) {
    tmpRankBitmapsH[i] = tmpRankBitmaps[i];
  }

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "gathered tmpRankBitmaps %p: %d %d %d %d\n",
        args.tmpRankBitmaps,
        args.tmpRankBitmaps[0],
        args.tmpRankBitmaps[1],
        args.tmpRankBitmaps[2],
        args.tmpRankBitmaps[3]);
  }
}

__global__ void ncclKernelAllToAllvDedupPrepare(
    int* flag,
    CtranAlgoDeviceState* devState,
    PrepareKernArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::KernelStartGpe(flag);
  }

  // TODO: move it into a generic routine for all kernels
  devState->opCount = args.opCount;
  devStateLoadToShm(devState);

  auto prepareGKSync = args.kSync.prepareGKSync;
  auto workerId = blockIdx.x;

  // Skip waitPost for step 0;

  // Barrier among all local ranks to ensure any previous call to prepare &
  // exec has finished, so P2P updating to other local ranks doesn't overwrite
  // previous call. Barrier among rail peers will be handled by GPE thread.
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nNodes = statex->nNodes();
  const auto myLocalRank = statex->localRank();
  barrier(myLocalRank, nLocalRanks);

  // Copies blockRecvBuckets -> blockRecvBucketsH (D2H) by all threads
  copyBlockRecvBuckets(args);

  // Notify GPE thread to compute tmpNumSendBlocks based on blockRecvBucketsH
  // and exchange
  GpeKernelSyncDev::complete(
      prepareGKSync, workerId, (int)PrepareSyncStep::kCopyBlockRecvBuckets);

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "completed prepareGKSync %p postFlag[%d] %d\n",
        prepareGKSync,
        workerId,
        prepareGKSync->postFlag[workerId]);
  }

  // Wait for GPE thread to receive remote tmpNumSendBlocks
  GpeKernelSyncDev::waitPost(
      prepareGKSync, workerId, (int)PrepareSyncStep::kPostTmpNumSendBlocksBuff);

  const auto warpId = threadIdx.x / kWarpSize;
  const auto laneId = threadIdx.x & (kWarpSize - 1);

  // Intranode gather numRecvCounts from all ranks based on
  // tmpNumSendBlocksBuffH (H2D, P2P) in each forwarding rank.
  // Handled by all threads
  gatherNumRecvCounts(args);

  // Each of rest tasks are lightweight, assign to different warps
  if (checkWarpRole(warpId, PrepareKernRole::kCompOffset)) {
    const auto nRanks = statex->nRanks();
    const auto count = nRanks * args.pArgs.numRecvBuckets;

    // laneId 0 compute offsets based on myTmpNumRecvBlocks (D2D)
    // FIXME: how to use full warp?
    if (laneId == 0) {
      computeOffsetsFromCounts(
          nLocalRanks,
          nNodes,
          args.tmpNumRecvBlocksBuff,
          args.prepareArgs.recvOffsets,
          args.prepareArgs.totalNumRecvBlocks,
          args.pArgs.numRecvBuckets,
          args.pArgs.blockCount);
    }
    syncwarp();

    // Full warp copies to tmpRecvOffsets for exec to track receive progress
    // (i.e., value will be updated, thus need separate from recvOffsets
    // returned to user)
    copyWarp<size_t>(args.tmpRecvOffsets, args.prepareArgs.recvOffsets, count);
    syncwarp();

    if (laneId == 0) {
      CTRAN_DEV_TRACE(
          "computed recvOffsets %p: %ld %ld, tmpRecvOffsets %p: %ld %ld, totalNumRecvBlocks %ld\n",
          args.prepareArgs.recvOffsets,
          args.prepareArgs.recvOffsets[0],
          args.prepareArgs.recvOffsets[1],
          args.tmpRecvOffsets,
          args.tmpRecvOffsets[0],
          args.tmpRecvOffsets[1],
          *args.prepareArgs.totalNumRecvBlocks);
    }
  } else if (checkWarpRole(warpId, PrepareKernRole::kCopyNumRecvBlocks)) {
    // myTmpNumRecvBlocks -> numRecvBlocks (D2D)
    copyNumRecvBlocks(args);
  } else if (checkWarpRole(warpId, PrepareKernRole::kCopyNumForwardBlocks)) {
    // numForwardBlocksH -> numForwardBlocks (H2D)

    // Wait for GPE thread to finish compute of numForwardBlocks
    GpeKernelSyncDev::waitPostWarp(
        prepareGKSync, workerId, (int)PrepareSyncStep::kPostNumForwardBlocks);
    copyNumForwardBlocks(args);
  } else if (checkWarpRole(warpId, PrepareKernRole::kCopyNumSendBlocks)) {
    // tmpNumSendBlocksBuffH -> numSendBlocks (H2D)
    copyNumSendBlocks(args);
  } else if (checkWarpRole(warpId, PrepareKernRole::kResetSync)) {
    FwdGroupSyncDev::resetWarp(
        args.config.numFwdGroups,
        args.config.numFwdWorkers,
        args.kSync.fwdGroupSync);
    WorkerSyncDev::resetWarp(args.kSync.workerSync);
  }

  // myTmpNumRecvBlocks -> numRecvBlocksH (D2H)
  copyNumRecvBlocksH(args);

  GpeKernelSyncDev::complete(
      prepareGKSync, workerId, (int)PrepareSyncStep::kCopyNumRecvBlocksH);

  GpeKernelSyncDev::complete(
      prepareGKSync, workerId, (int)PrepareSyncStep::kKernelDone);

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }

  // reset for next prepare to use.
  // syncthreads so we don't reset while other threads are still using it,
  // otherwise warp 0 may hit reset while warp kCopyNumForwardBlocks
  // is still waiting on the sync
  __syncthreads();
  GpeKernelSyncDev::reset(prepareGKSync, blockIdx.x);
}

__global__ void ncclKernelAllToAllvDedupNvlGatherMetadata(
    int* flag,
    CtranAlgoDeviceState* devState,
    PrepareKernArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::KernelStartGpe(flag);
  }

  // TODO: move it into a generic routine for all kernels
  devState->opCount = args.opCount;
  devStateLoadToShm(devState);

  // Barrier among all local ranks to ensure any previous call to prepare &
  // exec has finished, so P2P updating to other local ranks doesn't overwrite
  // previous call. Barrier among rail peers will be handled by GPE thread.
  const auto nLocalRanks = statex->nLocalRanks();
  const auto myLocalRank = statex->localRank();
  barrier(myLocalRank, nLocalRanks);

  gatherRankBitmaps(args);

  // Wait for all P2P updates from intra-node ranks
  barrier(myLocalRank, nLocalRanks);

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

} // namespace ctran::alltoallvdedup
