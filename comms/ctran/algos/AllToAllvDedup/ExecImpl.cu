// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>
#include <cstddef>

#include "comms/ctran/algos/AllToAllvDedup/CommonDev.h"
#include "comms/ctran/algos/AllToAllvDedup/ExecCommon.cuh"
#include "comms/ctran/algos/AllToAllvDedup/IndexMapDev.cuh"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/commstate/CommStateXDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"

using namespace ctran::algos;

#define ENABLE_NONBLOCKING_WAIT 1

// #define VERBOSE 1
#define VERBOSE 0
constexpr int kNumBlocksToTrace = 5;
#define DATATRACE_COND(b) (b < kNumBlocksToTrace && VERBOSE)

namespace ctran::alltoallvdedup {

template <typename T>
__device__ void progressSendCopy(ExecKernArgs& args, WorkerGroup& sendG) {
  const auto workerId = sendG.workerId();
  const auto numWorkers = sendG.numWorkers;
  const auto myNode = statex->node();
  const auto rank = statex->rank();
  const auto myRank = statex->rank();

  // Each group handles different nodeIds
  GroupNodes range = assignGroupNodes(args, sendG);
  NODE_RANGE_TRACE(sendG, range);

  const auto maxNumStepBlks = args.pArgs.maxNumStepBlks;
  const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;

  ProgressSendState state;
  state.setupState(args, range);

  // Each group iterates on different node range;
  while (state.numTotalPending > 0) {
    for (auto n = 0; n < range.numNodes; n++) {
      const auto recvNode = n + range.start;
      if (state.numPendingBlocks[n] == 0 || recvNode == myNode) {
        // Copied all blocks for this node, move to next
        // Skip blocks sending to local ranks, since they will be handled by
        // intra forward workers.
        continue;
      }

      // Wait host side to post tmpSendBuff ready
      auto gkSync = args.kSync.sendGKSyncs + recvNode;

      const auto step = state.steps[n];
      bool posted = false;
      if (workerId == 0) {
        posted = GpeKernelSyncDev::checkPost(gkSync, 0, step);
      }

      // All workers to wait for worker 0 to dispatch this step
      sendG.syncBcast(posted);
      if (!posted) {
        // If no available chunk is posted for this node, move to next
        continue;
      }

      const auto* tmpSendIdx = args.tmpSendIdx;
      const auto* tmpNumSendIdx = args.tmpNumSendIdx;

      // Copy blocks into corresponding chunk in tmpSendBuff
      void* tmpBuff =
          getTmpChunkPtr(args.config, args.tmpSendBuff, step, recvNode);

      // Count total number of blocks to copy to the recvNode, so we can
      // define the range for bitmaps and data to keep contig send chunk
      // without padding
      const auto numToCopy = state.numPendingBlocks[n] < maxNumStepBlks
          ? state.numPendingBlocks[n]
          : maxNumStepBlks;

      FwdChkHdr* hdr = getTmpChunkHdr(tmpBuff);
      int* remBlockIds = getTmpChunkBlockIds(tmpBuff);
      T* remBlocks = getTmpChunkData<T>(tmpBuff, maxNumStepBlks);

      const auto blockCount = static_cast<int>(args.pArgs.blockCount);
      const auto numFinished =
          tmpNumSendIdx[recvNode] - state.numPendingBlocks[n];
      const auto idxOffset = recvNode * totalNumSendBlocks + numFinished;
      const T* sendBuff = reinterpret_cast<const T*>(args.execArgs.sendBuff);

      SEND_DATACOPY_TRACE(numToCopy, recvNode, myRank, step);

      // Each worker handle different block
      for (auto rb = workerId; rb < numToCopy; rb += numWorkers) {
        const auto sIdx = tmpSendIdx[idxOffset + rb];
        if (threadIdx.x == 0) {
          remBlockIds[rb] = sIdx;
        }

        // All threads in a worker copy a block
        const T* sData =
            reinterpret_cast<const T*>(sendBuff + sIdx * blockCount);
        T* dData = reinterpret_cast<T*>(remBlocks + rb * blockCount);
        ctranKernCopy<T>(sData, dData, blockCount, 0, 1);
        DATACOPY_TRACE(DATATRACE_COND(rb), blockCount, sIdx, rb, sIdx, sData);
      }

      // Global tid 0 updates header
      if (workerId == 0 && threadIdx.x == 0) {
        hdr->numBlocks = numToCopy;
        hdr->sendRank = rank;
        hdr->opCount = (int)args.opCount;
      }

      // Worker 0 waits till all workers have finished
      __threadfence();
      sendG.syncJoin();

      // Worker 0 notifies GPE thread to send out the chunk
      if (workerId == 0) {
        GpeKernelSyncDev::complete(gkSync, 0, step);
      }

      state.updateStep(n, numToCopy);
    }
  }
}

template <typename T>
__device__ void progressIntraFwd(ExecKernArgs& args, WorkerGroup& intraFwdG) {
  const auto workerId = intraFwdG.workerId();
  const auto numWorkers = intraFwdG.numWorkers;

  PROFILE_INTRAFWD_STEP_START(args, 0, intraFwdG);

  GroupLocalRanks range = assignGroupLocalRanks(args, intraFwdG);
  LOCAL_RANK_RANGE_TRACE(intraFwdG, range);

  const auto myRank = statex->rank();
  const auto localRank = statex->localRank();
  const auto myNode = statex->node();
  const auto nNodes = statex->nNodes();
  constexpr bool kIsIntra = true;
  constexpr bool kIsExec = true;

  ProgressIntraFwdState state;
  state.setupState(args, range);

  const auto& config = args.config;
  const auto blockCount = args.pArgs.blockCount;
  const auto maxNumStepBlks = args.pArgs.maxNumStepBlks;
  const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;
  const auto* tmpIntraFwdIdx = args.tmpFwdIdx + myNode * totalNumSendBlocks;

  while (state.numTotalPending > 0) {
    // All workers in the group traverse range of local ranks. Each time copys a
    // chunk for the receive local rank, with index specified in tmpFwdIdx
    for (auto r = range.start; r <= range.end; r++) {
      // No more blocks to the given local receive rank
      if (state.numPendingBlocks[r] == 0) {
        continue;
      }

      const auto remStep = state.steps[r];
      const auto startBlock =
          args.tmpNumIntraFwdIdx[r] - state.numPendingBlocks[r];
      // offset since tmpIntraFwdIdx (tmpFwdIdx[localRank][myNode])
      const auto idxOffset = r * nNodes * totalNumSendBlocks + startBlock;
      const auto numToCopy = state.numPendingBlocks[r] > maxNumStepBlks
          ? maxNumStepBlks
          : state.numPendingBlocks[r];

      void* remTmpBuff = getTmpChunkPtr(
          config, args.remTmpIntraRecvBuffs[r], remStep, localRank);
      FwdChkHdr* remHdr = getTmpChunkHdr(remTmpBuff);
      int* remRecvBlockIds = getTmpChunkBlockIds(remTmpBuff);
      T* remRecvBlocks = getTmpChunkData<T>(remTmpBuff, maxNumStepBlks);
      auto* frSync = getFwdRecvSync(args, r, localRank, remStep, kIsIntra);

#ifdef ENABLE_NONBLOCKING_WAIT
      // Check if the remote tmpRecvBuff is ready; if not, skip and move to next
      // local rank
      bool ready = false;
      if (workerId == 0) {
        FWD_RECVSYNC_WAIT_TRACE(
            args, state, r, remStep, numToCopy, frSync, kIsExec);
        ready = fwdRecvSyncCheckReady(frSync);
      }
      intraFwdG.syncBcast(ready);
      if (!ready) {
        continue;
      }
#else
      if (workerId == 0) {
        fwdRecvSyncWaitReady(frSync);
      }
      intraFwdG.syncDispatch();
#endif
      FWD_DATACOPY_TRACE(numToCopy, r, myRank, remStep);

      // Each worker handle different block
      for (auto rb = workerId; rb < numToCopy; rb += numWorkers) {
        // index in the current fwd chunk
        const auto sIdx = tmpIntraFwdIdx[idxOffset + rb];
        if (threadIdx.x == 0) {
          remRecvBlockIds[rb] = sIdx;
        }

        const T* sData = reinterpret_cast<const T*>(args.execArgs.sendBuff) +
            sIdx * blockCount;

        // Record dest ptr for the peer
        ctranKernCopy<T>(
            sData, remRecvBlocks + rb * blockCount, blockCount, 0, 1);
        DATACOPY_TRACE(DATATRACE_COND(rb), blockCount, sIdx, rb, sIdx, sData);
      }

      if (workerId == 0 && threadIdx.x == 0) {
        remHdr->sendRank = myRank;
        remHdr->numBlocks = numToCopy;
        remHdr->opCount = (int)args.opCount;
      }

      __threadfence();
      intraFwdG.syncJoin();

      // Notify recv rank to consume
      if (workerId == 0) {
        fwdRecvSyncPost(frSync, remStep);
      }
      state.updateStep(r, numToCopy);
    }
  }

  PROFILE_INTRAFWD_STEP_END(args, 0, intraFwdG);
}

// Return true if forwarded to the specified recvLocalRank; false if the
// recvLocalRank is not ready.
template <typename T>
__device__ __forceinline__ bool handleFwdCopy(
    ExecKernArgs& args,
    ProgressFwdState& state,
    const int sendNode,
    const int netStep,
    const int recvLocalRank,
    const int remStep,
    const int numToCopy,
    WorkerGroup& fwdG) {
  const auto localRank = statex->localRank();
  auto frSync = getFwdRecvSync(args, recvLocalRank, localRank, remStep);
  const auto workerId = fwdG.workerId();
  const auto numWorkers = fwdG.numWorkers;

  constexpr bool kIsExec = true;
  bool ready = false;
  if (workerId == 0) {
    FWD_RECVSYNC_WAIT_TRACE(
        args, state, recvLocalRank, remStep, numToCopy, frSync, kIsExec);
    ready = fwdRecvSyncCheckReady(frSync);
  }
  fwdG.syncBcast(ready);
  if (!ready) {
    return false;
  }

  // Receiver chunk is ready, now copy by all workers
  const auto nNodes = statex->nNodes();
  const auto slotId = recvLocalRank * nNodes + sendNode;
  const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;
  const auto blockCount = args.pArgs.blockCount;
  const auto maxNumStepBlks = args.pArgs.maxNumStepBlks;

  void* tmpBuff =
      getTmpChunkPtr(args.config, args.tmpFwdBuff, netStep, sendNode);
  FwdChkHdr* netHdr = reinterpret_cast<FwdChkHdr*>(tmpBuff);
  const auto sendRank = netHdr->sendRank;
  int* netBlockIds = getTmpChunkBlockIds(tmpBuff);
  T* netBlocks = getTmpChunkData<T>(tmpBuff, maxNumStepBlks);

  void* remTmpRecvBuff = getTmpChunkPtr(
      args.config, args.remTmpRecvBuffs[recvLocalRank], remStep, localRank);
  int* remRecvBlockIds = getTmpChunkBlockIds(remTmpRecvBuff);
  T* remRecvBlocks =
      getTmpChunkData<T>(remTmpRecvBuff, args.pArgs.maxNumStepBlks);

  FWD_DATACOPY_TRACE(numToCopy, recvLocalRank, sendRank, remStep);

  // Each worker handles a different block
  const auto* currTmpFwdIdx = args.tmpFwdIdx + slotId * totalNumSendBlocks;
  for (auto rb = workerId; rb < numToCopy; rb += numWorkers) {
    // relative_index_in_remRecvBlocks : relative_index_in_fwdBlocks
    const auto b = currTmpFwdIdx[rb];
    int sIdx = netBlockIds[b];
    if (threadIdx.x == 0) {
      remRecvBlockIds[rb] = sIdx;
    }

    const T* sData = netBlocks + b * blockCount;
    ctranKernCopy<T>(sData, remRecvBlocks + rb * blockCount, blockCount, 0, 1);
    DATACOPY_TRACE(DATATRACE_COND(rb), blockCount, b, rb, sIdx, sData);
  }

  // Worker 0 waits for every local forward to finish by all workers
  __threadfence();
  fwdG.syncJoin();

  // Update header and notify receiver
  FwdChkHdr* remHdr = getTmpChunkHdr(remTmpRecvBuff);
  if (workerId == 0 && threadIdx.x == 0) {
    remHdr->sendRank = sendRank;
    remHdr->numBlocks = numToCopy;
    remHdr->opCount = (int)args.opCount;
  }
  if (workerId == 0) {
    fwdRecvSyncPost(frSync, remStep);
  }
  return true;
}

template <typename T>
__device__ void progressFwd(ExecKernArgs& args, WorkerGroup& fwdG) {
  const auto workerId = fwdG.workerId();
  const auto numWorkers = fwdG.numWorkers;
  const auto nNodes = statex->nNodes();
  const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;

  GroupNodes range = assignGroupNodes(args, fwdG);
  NODE_RANGE_TRACE(fwdG, range);

  ProgressFwdState state;
  state.setupState(args, range);

  int* tmpFwdIdx = args.tmpFwdIdx;
  const int* fwdIdx = args.execArgs.fwdIdx;

  while (state.numTotalPending > 0) {
    for (auto n = 0; n < nNodes; n++) {
      auto sendNode = n;
      // skip if has finished all steps for this sendNode
      if (state.numPendingBlocks[n] == 0) {
        continue;
      }

      const auto netStep = state.steps[n];
      bool posted = false;
      if (workerId == 0) {
        posted = GpeKernelSyncDev::checkPost(
            args.kSync.recvGKSyncs + sendNode, 0, netStep);
      }
      // All workers wait for worker 0 to check whether RDMA is received
      fwdG.syncBcast(posted);

      if (posted) {
        // Since only worker 0 checks post with implicit memory fence, we need
        // explicit fence on other workers to see the posted data.
        __threadfence();
      } else {
        // If no available chunk is posted for this node, move to next
        continue;
      }

      // Obtain hdr, blockIds, and data blocks arrays from the corresponding
      // chunk in tmpFwdBuff.
      // Format: FwdChkHdr + blockIds[numBlocks] + blockArray[numBlocks]
      void* tmpBuff =
          getTmpChunkPtr(args.config, args.tmpFwdBuff, netStep, sendNode);
      FwdChkHdr* hdr = reinterpret_cast<FwdChkHdr*>(tmpBuff);
      const auto numBlocks = hdr->numBlocks;
      int* remFwdBlockIds = getTmpChunkBlockIds(tmpBuff);
      FWDHDR_TRACE_CHECK(
          VERBOSE, args, hdr, "sendNode", sendNode, netStep, remFwdBlockIds);

      // record step metadata for combine() to replay
      // no sync after, since it would be used only at combine()
      if (workerId == 0) {
        FwdStepInfo stepInfo = {numBlocks, remFwdBlockIds};
        recordFwdStep(args, netStep, sendNode, stepInfo);
      }

      const auto nLocalRanks = statex->nLocalRanks();
      const auto localRank = statex->localRank();
      const auto nNodes = statex->nNodes();

      // All workers convert tmpFwdIdx for the current remFwdBlockIds; use
      // tmpFwdIdx[r][sendNode] as temp buffer for each local rank `r`
      // tmpFwdIdx format: tmpFwdIdx[nLocalRank][nNodes][totalNumSendBlocks],
      // each tmpFwdIdx[nLocalRank][nNodes] corresponds to the lookupMap of
      // blockIdxs sent from the node to the local receive rank, at length of
      // totalNumSendBlocks.
      //              sendnode0    sendnode1
      // localrank0 | lookupMap  | lookupMap |...
      // localrank1 | lookupMap  | lookupMap |...
      // ...
      // For each lookupMap, it reprensents sendBlockIdx : recvIdx mapping.
      // To lookup each blockIdx in received chunk `fwdBlockIdx` from sendNode n
      // to local rank r, it looks up blockIdx (sb) in tmpFwdIdx[r][n]. The
      // matching recvIdx (rb) indicates the index on receive rank r.
      //
      // To make copy loop fast, we further convert the matching map to use
      // relative index in the to-be-forwarded chunk for local rank r
      // (`remBlockIds`) as key, and relative index in the received chunk from
      // sendnode n as value. The convertion is handled within
      // IndexMapDev::transposeSubset().
      //
      // E.g.,
      // Inputs:
      // - fwdIdx[][sendnode0]:
      //                        sendnode0
      //               (sb     0   1   2   3   4)
      //    localrank0 (rb)    2   3   4  -1   5
      //    localrank1 (rb)   -1  -1   5   6  -1
      //
      // - blockIdx in received chunk `fwdBlockIdx` from sendnode0: 2 3 4
      //
      // Output:
      // - tmpFwdIdx[][sendnode0] after convert:
      //            (rb   0   1)
      // localrank0 (sb)  0   2
      //            (rb   0   1)
      // localrank0 (sb)  0   1
      for (auto r = workerId; r < nLocalRanks; r += numWorkers) {
        const auto slotId = r * nNodes + sendNode;
        const auto offset = slotId * totalNumSendBlocks;
        // count number of blocks to be forwarded to each localRank
        const auto count = IndexMapDev::transposeSubset(
            &fwdIdx[offset], remFwdBlockIds, numBlocks, &tmpFwdIdx[offset]);
        if (threadIdx.x == 0) {
          recordFwdStepRecvrNumBlocks(args, netStep, r, count);
        }
      }
      // Different workers handle different local ranks. We need fence to ensure
      // cross-thread-block read-modify-write memory consistency, as each
      // worker can see tmpFwdIdx updated by other workers.
      __threadfence();
      fwdG.syncBarrier();
      __threadfence();

      // Number of blocks forwarded to each local rank for current step.
      // Load into shm for repeated fast access
      // MIN_TODO: use prepopulated data from prepare()
      __shared__ int stepRecvrNumBlocks[CTRAN_MAX_NVL_PEERS];
      loadFwdStepRecvrNumBlocks(args, netStep, stepRecvrNumBlocks);

      // MIN_TODO: this can be prepopulated in prepare()
      int numPendingFwds = 0;
      for (auto r = 0; r < nLocalRanks; r++) {
        const auto numToCopy = stepRecvrNumBlocks[r];
        numPendingFwds += (numToCopy > 0 ? 1 : 0);
        if (numToCopy > 0) {
          state.updatePostStep(r);
        }
      }

      // Traverse each local rank and handle forwarding copy
      // Each fwd rank shifts to right, try to avoid incast congestion on recv
      // rank. However, the actual copy order is not guaranteed, if any recv
      // rank is delayed on complete the chunk. The loop ends till all
      // forwarding has finished, tracked by numPendingFwds.
      while (numPendingFwds > 0) {
        for (auto x = 0; x < nLocalRanks; x++) {
          auto r = (localRank + x) % nLocalRanks;
          const auto numToCopy = stepRecvrNumBlocks[r];
          if (numToCopy == 0 || state.fwdDone[r] == state.fwdReady[r]) {
            continue;
          }

          const auto remStep = state.fwdDone[r];
          const bool copied = handleFwdCopy<T>(
              args, state, sendNode, netStep, r, remStep, numToCopy, fwdG);
          if (copied) {
            // Local update total number of blocks copied for debugging
            state.numFwdBlocksToRank[r] += numToCopy;
            state.updateFwdStep(r);
            numPendingFwds--;
          }
        }
      }

      // Worker 0 notifies GPE thread that we have finished all forwarding of
      // this chunk.
      if (workerId == 0) {
        GpeKernelSyncDev::complete(
            args.kSync.recvGKSyncs + sendNode, 0, netStep);
      }
      state.updateStep(n, numBlocks);
    }
  }
}

// Temporary buffers used in progressRecvCopy pipeline step
// - Each recv group may handle different chunk in parallel, thus prepare
//   different tmp buffers per group. Slot [MAX_NUM_GROUPS_PER_ROLE] is for
//   intraRecv which has only 1 group.
// - We transpose indexMap by using all workers in the group, and then let
// each
//   worker handle different block copy to all buckets. Thus, prepare tmp
//   buffer also per bucket.
__device__ int tmpBktFirstRecvIdx[MAX_NUM_GROUPS_PER_ROLE + 1]
                                 [MAX_NUM_RECV_BUCKETS];
__device__ int tmpBktNumRecvIdx[MAX_NUM_GROUPS_PER_ROLE + 1]
                               [MAX_NUM_RECV_BUCKETS];

template <typename T>
__device__ inline int handleRecvCopyStep(
    ExecKernArgs& args,
    ProgressRecvState& state,
    const int fwdLocalRank,
    const int step,
    WorkerGroup& recvG,
    const bool kIsIntra) {
  const auto workerId = recvG.workerId();
  const auto groupId = recvG.groupId();
  const auto numWorkers = recvG.numWorkers;
  const auto localRank = statex->localRank();

  constexpr bool kIsExec = false;
  void* tmpBase = kIsIntra ? args.tmpIntraRecvBuff : args.tmpRecvBuff;
  void* tmpRecvBuff = getTmpChunkPtr(args.config, tmpBase, step, fwdLocalRank);
  auto* frSync = getFwdRecvSync(args, localRank, fwdLocalRank, step, kIsIntra);

  const auto nRanks = statex->nRanks();
  const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;
  const auto maxNumStepBlks = args.pArgs.maxNumStepBlks;
  const auto* recvIdx = args.execArgs.recvIdx;
  auto* tmpRecvIdx = args.tmpRecvIdx;
  auto* tmpRecvOffsets = args.tmpRecvOffsets;

#ifdef ENABLE_NONBLOCKING_WAIT
  bool posted = false;
  if (workerId == 0) {
    // local worker 0 waits for all fwd workers to finish
    RECVCOPY_RECVSYNC_WAIT_TRACE(
        args, state, fwdLocalRank, step, frSync, kIsIntra, kIsExec);
    posted = fwdRecvSyncCheckPost(frSync, step);
  }

  // all other workers waits for worker 0 to kick off this step.
  recvG.syncBcast(posted);
  if (posted) {
    // Since only worker 0 checks post with implicit memory fence, we need
    // explicit fence on other workers to see the posted data.
    __threadfence();
  } else {
    return 0;
  }
#else
  if (workerId == 0) {
    RECVCOPY_RECVSYNC_WAIT_TRACE(
        args, state, fwdLocalRank, step, frSync, kIsIntra, kIsExec);
    fwdRecvSyncWaitPost(frSync, step);
  }
  recvG.syncDispatch();
#endif

  PROFILE_RECVCOPY_STEP_START(args, step, recvG, fwdLocalRank, kIsIntra);

  // Copy from tmpRecvBuff to user recvBuff
  FwdChkHdr* hdr = getTmpChunkHdr(tmpRecvBuff);
  const auto numBlocks = hdr->numBlocks;
  const auto sendRank = hdr->sendRank;
  int* remBlockIds = getTmpChunkBlockIds(tmpRecvBuff);
  T* remFwdBlocks = getTmpChunkData<T>(tmpRecvBuff, maxNumStepBlks);

  FWDHDR_TRACE_CHECK(
      VERBOSE, args, hdr, "fwdLocalRank", fwdLocalRank, step, remBlockIds);

  // record step metadata for combine() to replay
  // no sync after record, since data would be used only at combine()
  if (workerId == 0) {
    const RecvCopyStepInfo stepInfo = {numBlocks, sendRank};
    recordRecvCopyStep(
        args, step, fwdLocalRank, stepInfo, remBlockIds, kIsIntra);
  }

  // Similar to how progressFwd uses tmpFwdIdx to route block to each
  // localRank destinations, here use tmpRecvIdx to route to the buckets.
  // IndexMapDev::transposeSubset() will find the matching indices in the
  // current received remBlockIds for each local bucket. It returns the
  // count of matching ones, and record the relative indices in
  // tmpRecvIdx[bucket][sendRank].
  // See detailed conversion example in progressFwd.
  //
  // >> recvIdx format:
  //              sendRank0      sendRank 1
  // bucket 0    [indexMap]      [indexMap]
  // bucket 1    [indexMap]      [indexMap]

  const auto tmpGroupId = kIsIntra ? MAX_NUM_GROUPS_PER_ROLE : groupId;
  const auto numRecvBuckets = args.pArgs.numRecvBuckets;
  for (auto bkt = workerId; bkt < numRecvBuckets; bkt += numWorkers) {
    const auto slotId = bkt * nRanks + sendRank;
    const auto offset = slotId * totalNumSendBlocks;
    int lastRecvIdx = 0, firstRecvIdx = 0;
    // tmpRecvIdx stores relative_rIdx : relative_idx_in_tmpBlocks mapping.
    // relative_rIdx is continuous, starting from 0. `firstRecvIdx +
    // relative_rIdx` is absolute index in recvBuff for a bucket and sendRank.
    int* tmpBlkRecvIdx = &tmpRecvIdx[offset];
    const auto count = IndexMapDev::transposeSubset(
        &recvIdx[offset],
        remBlockIds,
        numBlocks,
        tmpBlkRecvIdx,
        &firstRecvIdx,
        &lastRecvIdx);
    if (threadIdx.x == 0) {
      tmpBktNumRecvIdx[tmpGroupId][bkt] = count;
      tmpBktFirstRecvIdx[tmpGroupId][bkt] = firstRecvIdx;
    }
    if (threadIdx.x == 0 && lastRecvIdx != count - 1) {
      CTRAN_DEV_FATAL(
          "Wrong tmpBlkRecvIdx convertion for bkt %d sendRank %d, expected last recvIdx %d but %d\n",
          bkt,
          sendRank,
          count - 1,
          lastRecvIdx);
    }
  }

  // ensure cross-thread-block read-modify-write memory consistency
  __threadfence();
  recvG.syncBarrier();
  __threadfence();

  const auto blockCount = args.pArgs.blockCount;
  T* recvBuff = reinterpret_cast<T*>(args.execArgs.recvBuff);
  int* recvBlockIds = args.execArgs.recvBlockIds;

  for (auto bkt = 0; bkt < numRecvBuckets; bkt++) {
    const auto slotId = bkt * nRanks + sendRank;
    const auto numToCopy = tmpBktNumRecvIdx[tmpGroupId][bkt];
    if (numToCopy == 0) {
      continue;
    }

    const auto firstRecvIdx = tmpBktFirstRecvIdx[tmpGroupId][bkt];
    const int* tmpBlkRecvIdx = &tmpRecvIdx[slotId * totalNumSendBlocks];

    const auto bktOffset = tmpRecvOffsets[bkt * nRanks + sendRank];

    RECVCOPY_DATACOPY_TRACE(numToCopy, bkt, sendRank, firstRecvIdx);

    // All threads in each worker copy a single block, different workers
    // copy different blocks
    for (auto b = workerId; b < numToCopy; b += numWorkers) {
      const auto rb = tmpBlkRecvIdx[b];
      const auto recvOffset = bktOffset + firstRecvIdx + b;
      const auto sIdx = remBlockIds[rb];
      if (threadIdx.x == 0) {
        // blockId of each received block, return to user
        recvBlockIds[recvOffset] = sIdx;
      }

      const T* sData = remFwdBlocks + rb * blockCount;
      // dest is per-bucket-per-sendRank offset (bktOffset) + offset of
      // current step (firstRecvIdx) + index in current step (b)
      T* dData = recvBuff + recvOffset * blockCount;

      ctranKernCopy<T>(sData, dData, blockCount, 0, 1);
      DATACOPY_TRACE(DATATRACE_COND(rb), blockCount, rb, b, sIdx, sData);
    }
  } // end of loop of numRecvBuckets

  // worker 0 waits all workers to join before sync with forward rank and host.
  recvG.syncJoin();

  // Notify forward rank that the buffer is free to reuse;
  if (workerId == 0) {
    fwdRecvSyncComplete(frSync);
  }
  PROFILE_RECVCOPY_STEP_END(args, step, recvG, fwdLocalRank, kIsIntra);

  return numBlocks;
}

template <typename T>
__device__ void progressIntraRecvCopy(
    ExecKernArgs& args,
    WorkerGroup& intraRecvG) {
  const auto nLocalRanks = statex->nLocalRanks();

  GroupLocalRanks range = assignGroupLocalRanks(args, intraRecvG);
  LOCAL_RANK_RANGE_TRACE(intraRecvG, range);

  ProgressRecvState state;
  state.setupIntraRecvState(args);
  constexpr bool kIsIntra = true;

  while (state.numTotalPending > 0) {
    for (auto r = 0; r < nLocalRanks; r++) {
      // already finished
      if (state.numPendingBlocks[r] == 0) {
        continue;
      }

      const auto step = state.steps[r];
      const auto numBlocks =
          handleRecvCopyStep<T>(args, state, r, step, intraRecvG, kIsIntra);

      // numBlocks can be zero if fwd rank not posted; skip update if so.
      if (numBlocks > 0) {
        state.updateStep(r, numBlocks);
      }
    }
  }
}

template <typename T>
__device__ void progressRecvCopy(ExecKernArgs& args, WorkerGroup& recvG) {
  GroupLocalRanks range = assignGroupLocalRanks(args, recvG);
  LOCAL_RANK_RANGE_TRACE(recvG, range);

  const bool kIsIntra = false;

  ProgressRecvState state;
  state.setupState(args, range);

  while (state.numTotalPending > 0) {
    for (auto r = range.start; r <= range.end; r++) {
      // already finished
      if (state.numPendingBlocks[r] == 0) {
        continue;
      }

      const auto step = state.steps[r];
      const auto numBlocks =
          handleRecvCopyStep<T>(args, state, r, step, recvG, kIsIntra);

      // numBlocks can be zero if fwd rank not posted; skip update if so.
      if (numBlocks > 0) {
        state.updateStep(r, numBlocks);
      }
    }
  }
}

template <typename T>
__global__ void ncclKernelAllToAllvDedup(
    int* flag,
    CtranAlgoDeviceState* devState,
    ExecKernArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  if (threadIdx.x == 0) {
    // TODO: move it into a generic routine for all kernels
    devState->opCount = args.opCount;
  }

  devStateLoadToShm(devState);
  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE("loaded devState %p\n", devState);
  }

  WorkerGroup sendG, fwdG, recvG, intraFwdG, intraRecvG;
  assignWorkerGroups(args, sendG, fwdG, recvG, intraFwdG, intraRecvG);

  if (sendG.contains(blockIdx.x)) {
    progressSendCopy<T>(args, sendG);
  } else if (fwdG.contains(blockIdx.x)) {
    progressFwd<T>(args, fwdG);
  } else if (recvG.contains(blockIdx.x)) {
    progressRecvCopy<T>(args, recvG);
  } else if (intraFwdG.contains(blockIdx.x)) {
    progressIntraFwd<T>(args, intraFwdG);
  } else if (intraRecvG.contains(blockIdx.x)) {
    progressIntraRecvCopy<T>(args, intraRecvG);
  }

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE("Finished progress loop\n");
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE("exit kernel\n");
  }
}

#define DECL_ALLTOALLVDEDUP_KERN(T)                     \
  template __global__ void ncclKernelAllToAllvDedup<T>( \
      int* flag, CtranAlgoDeviceState* devState, ExecKernArgs args)

DECL_ALLTOALLVDEDUP_KERN(int8_t);
DECL_ALLTOALLVDEDUP_KERN(uint8_t);
DECL_ALLTOALLVDEDUP_KERN(int32_t);
DECL_ALLTOALLVDEDUP_KERN(uint32_t);
DECL_ALLTOALLVDEDUP_KERN(int64_t);
DECL_ALLTOALLVDEDUP_KERN(uint64_t);
DECL_ALLTOALLVDEDUP_KERN(half);
DECL_ALLTOALLVDEDUP_KERN(float);
DECL_ALLTOALLVDEDUP_KERN(double);
#if defined(__CUDA_BF16_TYPES_EXIST__)
DECL_ALLTOALLVDEDUP_KERN(__nv_bfloat16);
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
DECL_ALLTOALLVDEDUP_KERN(__nv_fp8_e4m3);
DECL_ALLTOALLVDEDUP_KERN(__nv_fp8_e5m2);
#endif

} // namespace ctran::alltoallvdedup
