// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/AllToAllvDedup/ExecCommon.cuh"
#include "comms/ctran/algos/AllToAllvDedup/IndexMapDev.cuh"
#include "comms/ctran/algos/DevAlgoImpl.cuh"

#define ENABLE_NONBLOCKING_WAIT 1

namespace ctran::alltoallvdedup {
using namespace ctran::algos;

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

} // namespace ctran::alltoallvdedup
