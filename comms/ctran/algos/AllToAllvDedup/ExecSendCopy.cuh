// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/AllToAllvDedup/ExecCommon.cuh"
#include "comms/ctran/algos/DevAlgoImpl.cuh"

namespace ctran::alltoallvdedup {
using namespace ctran::algos;

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

} // namespace ctran::alltoallvdedup
