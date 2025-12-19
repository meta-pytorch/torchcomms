// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/AllToAllvDedup/CommonDev.h"
#include "comms/ctran/algos/AllToAllvDedup/ExecCommon.cuh"
#include "comms/ctran/algos/AllToAllvDedup/IndexMapDev.cuh"
#include "comms/ctran/algos/DevAlgoImpl.cuh"

#define ENABLE_NONBLOCKING_WAIT 1

namespace ctran::alltoallvdedup {
using namespace ctran::algos;

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

} // namespace ctran::alltoallvdedup
