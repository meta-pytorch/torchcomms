// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstddef>

#include "comms/ctran/algos/AllToAllvDedup/CommonDev.h"
#include "comms/ctran/algos/AllToAllvDedup/ResourceImpl.h"
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran::alltoallvdedup {
using algos::GpeKernelSync;
using ::ctran::algos::bufmanager::MemType;

#define ADD_BUF(map, bname, memtype, buflen) \
  map.try_emplace(ResourceBufName::bname, memtype, buflen, ARGTOSTR(bname));

void ResourceImpl::initBufMd(
    const PersistArgs& args,
    const PersistConfig& config,
    const int nNodes,
    const int nLocalRanks) {
  auto& map = bufMdMap_;
  const auto nRanks = nNodes * nLocalRanks;

  const auto totalNumSendBlocks = args.totalNumSendBlocks;
  const auto maxNumStepBlks = args.maxNumStepBlks;
  const auto maxNumSteps = args.maxNumSteps;
  const auto numRecvBuckets = args.numRecvBuckets;

  const auto tmpNumChunks = config.tmpNumChunks;
  const auto tmpChunkSize = config.tmpChunkSize;

  size_t perRankBufLen = tmpChunkSize * tmpNumChunks;
  size_t buflen = perRankBufLen * nNodes;
  ADD_BUF(map, kTmpFwdBuff, MemType::kDevice, buflen);
  ADD_BUF(map, kTmpSendBuff, MemType::kDevice, buflen);

  buflen = nNodes * totalNumSendBlocks * sizeof(int);
  ADD_BUF(map, kTmpSendIdx, MemType::kDevice, buflen);

  buflen = nNodes * sizeof(int);
  ADD_BUF(map, kTmpNumSendIdx, MemType::kDevice, buflen);

  buflen = nNodes * nLocalRanks * totalNumSendBlocks * sizeof(int);
  ADD_BUF(map, kTmpFwdIdx, MemType::kDevice, buflen);

  buflen = nNodes * sizeof(int);
  ADD_BUF(map, kTmpNumFwdIdx, MemType::kDevice, buflen);

  buflen = nLocalRanks * sizeof(int);
  ADD_BUF(map, kTmpNumIntraFwdIdx, MemType::kDevice, buflen);

  buflen = nRanks * numRecvBuckets * totalNumSendBlocks * sizeof(int);
  ADD_BUF(map, kTmpRecvIdx, MemType::kDevice, buflen);

  buflen = nLocalRanks * sizeof(int);
  ADD_BUF(map, kTmpNumFwdRecvIdx, MemType::kDevice, buflen);
  ADD_BUF(map, kTmpNumIntraRecvIdx, MemType::kDevice, buflen);

  buflen = numRecvBuckets * nRanks * sizeof(int);
  ADD_BUF(map, kTmpRecvOffsets, MemType::kDevice, buflen);

  buflen = nNodes * sizeof(int);
  ADD_BUF(map, kTmpNumSendIdxH, MemType::kHostPinned, buflen);
  ADD_BUF(map, kTmpNumFwdIdxH, MemType::kHostPinned, buflen);

  buflen = nLocalRanks * sizeof(int);
  ADD_BUF(map, kTmpNumIntraFwdIdxH, MemType::kHostPinned, buflen);
  ADD_BUF(map, kTmpNumFwdRecvIdxH, MemType::kHostPinned, buflen);
  ADD_BUF(map, kTmpNumIntraRecvIdxH, MemType::kHostPinned, buflen);

  // separate buffer for each local FWD ranks
  buflen = perRankBufLen * nLocalRanks;
  ADD_BUF(map, kTmpRecvBuff, MemType::kDevice, buflen);
  ADD_BUF(map, kTmpIntraRecvBuff, MemType::kDevice, buflen);

  buflen = maxNumSteps * nNodes * sizeof(int);
  ADD_BUF(map, kTmpSendRedStepNumPending, MemType::kDevice, buflen);

  buflen = totalNumSendBlocks * nNodes * sizeof(int);
  ADD_BUF(map, kTmpSendRedIdxSrcIds, MemType::kDevice, buflen);

  buflen = totalNumSendBlocks * sizeof(int);
  ADD_BUF(map, kTmpSendRedIdxNumSrcs, MemType::kDevice, buflen);
  ADD_BUF(map, kTmpSendRedIdxNumPendingSrcs, MemType::kDevice, buflen);

  buflen = maxNumSteps * nLocalRanks * sizeof(int);
  ADD_BUF(map, kTmpIntraRedStepNumPending, MemType::kDevice, buflen);

  buflen = totalNumSendBlocks * nLocalRanks * sizeof(int);
  ADD_BUF(map, kTmpIntraRedIdxSrcIds, MemType::kDevice, buflen);

  buflen = totalNumSendBlocks * sizeof(int);
  ADD_BUF(map, kTmpIntraRedIdxNumSrcs, MemType::kDevice, buflen);
  ADD_BUF(map, kTmpIntraRedIdxNumPendingSrcs, MemType::kDevice, buflen);

  buflen = nRanks * totalNumSendBlocks * sizeof(int);
  ADD_BUF(map, kTmpRecvRedIdxNumSrcs, MemType::kDevice, buflen);

  buflen = nRanks * totalNumSendBlocks * numRecvBuckets * sizeof(int);
  ADD_BUF(map, kTmpRecvRedIdxSrcIds, MemType::kDevice, buflen);

  buflen = maxNumSteps * nLocalRanks * sizeof(RecvCopyStepInfo);
  ADD_BUF(map, kRecvStepInfo, MemType::kDevice, buflen);
  ADD_BUF(map, kIntraRecvStepInfo, MemType::kDevice, buflen);

  buflen = maxNumSteps * nLocalRanks * maxNumStepBlks * sizeof(int);
  ADD_BUF(map, kRecvStepFwdBlockIds, MemType::kDevice, buflen);
  ADD_BUF(map, kIntraRecvStepFwdBlockIds, MemType::kDevice, buflen);

  buflen = maxNumSteps * nLocalRanks * sizeof(int);
  ADD_BUF(map, kFwdStepRecvrNumBlocks, MemType::kDevice, buflen);

  buflen = maxNumSteps * nNodes * maxNumStepBlks * sizeof(int);
  ADD_BUF(map, kFwdStepBlocksIds, MemType::kDevice, buflen);

  buflen = maxNumSteps * nNodes * sizeof(int);
  ADD_BUF(map, kFwdStepNumBlocks, MemType::kDevice, buflen);

  // one sync per local rank per tmpChunk
  buflen = sizeof(FwdRecvSync) * tmpNumChunks * nLocalRanks;
  ADD_BUF(map, kFwdRecvSync, MemType::kDevice, buflen);
  ADD_BUF(map, kIntraFwdRecvSync, MemType::kDevice, buflen);

  buflen = sizeof(IntraRedSync);
  ADD_BUF(map, kIntraRedSync, MemType::kDevice, buflen);

  buflen = nNodes * sizeof(GpeKernelSync);
  ADD_BUF(map, kSendGKSyncs, MemType::kHostPinned, buflen);
  ADD_BUF(map, kRecvGKSyncs, MemType::kHostPinned, buflen);
  buflen = nLocalRanks * sizeof(GpeKernelSync);
  ADD_BUF(map, kIntraFwdGKSyncs, MemType::kHostPinned, buflen);
  ADD_BUF(map, kRecvCopyGKSyncs, MemType::kHostPinned, buflen);
  ADD_BUF(map, kIntraRecvCopyGKSyncs, MemType::kHostPinned, buflen);

  buflen = MAX_NUM_GROUPS_PER_ROLE * (int)WorkerGroupType::kNumTypes *
      sizeof(WorkerGroupSync);
  ADD_BUF(map, kWorkerGroupSync, MemType::kDevice, buflen);

  // Sanity check we don't miss any
  const auto numBufNames = static_cast<size_t>(ResourceBufName::kNumBufsNames);
  FB_CHECKTHROW(
      map.size() == numBufNames,
      "map.size() {} != {}",
      map.size(),
      numBufNames);
}

void ResourceImpl::assignToKernArgs(
    ExecKernArgs& kernArgs,
    const bool skipRem) {
  auto& resRef = ref_;
  // if remote exchange has been skiped, buffer are not registered
  const auto useReg = skipRem ? false : true;

  GET_RESOURCE_BUFPTR(&resRef, kTmpFwdBuff, useReg, kernArgs.tmpFwdBuff);
  GET_RESOURCE_BUFPTR(&resRef, kTmpSendBuff, useReg, kernArgs.tmpSendBuff);
  GET_RESOURCE_BUFPTR(&resRef, kTmpSendIdx, useReg, kernArgs.tmpSendIdx);
  GET_RESOURCE_BUFPTR(&resRef, kTmpNumSendIdx, useReg, kernArgs.tmpNumSendIdx);
  GET_RESOURCE_BUFPTR(&resRef, kTmpFwdIdx, useReg, kernArgs.tmpFwdIdx);
  GET_RESOURCE_BUFPTR(&resRef, kTmpNumFwdIdx, useReg, kernArgs.tmpNumFwdIdx);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpNumIntraFwdIdx, useReg, kernArgs.tmpNumIntraFwdIdx);
  GET_RESOURCE_BUFPTR(&resRef, kTmpRecvIdx, useReg, kernArgs.tmpRecvIdx);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpNumFwdRecvIdx, useReg, kernArgs.tmpNumFwdRecvIdx);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpNumIntraRecvIdx, useReg, kernArgs.tmpNumIntraRecvIdx);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpNumSendIdxH, useReg, kernArgs.tmpNumSendIdxH);
  GET_RESOURCE_BUFPTR(&resRef, kTmpNumFwdIdxH, useReg, kernArgs.tmpNumFwdIdxH);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpNumIntraFwdIdxH, useReg, kernArgs.tmpNumIntraFwdIdxH);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpNumIntraRecvIdxH, useReg, kernArgs.tmpNumIntraRecvIdxH);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpNumFwdRecvIdxH, useReg, kernArgs.tmpNumFwdRecvIdxH);

  GET_RESOURCE_BUFPTR(&resRef, kTmpRecvBuff, useReg, kernArgs.tmpRecvBuff);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpIntraRecvBuff, useReg, kernArgs.tmpIntraRecvBuff);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpRecvOffsets, useReg, kernArgs.tmpRecvOffsets);

  // metadata recorded in exec() and used in combine
  GET_RESOURCE_BUFPTR(&resRef, kRecvStepInfo, useReg, kernArgs.recvStepInfo);
  GET_RESOURCE_BUFPTR(
      &resRef, kRecvStepFwdBlockIds, useReg, kernArgs.recvStepFwdBlockIds);
  GET_RESOURCE_BUFPTR(
      &resRef, kIntraRecvStepInfo, useReg, kernArgs.intraRecvStepInfo);
  GET_RESOURCE_BUFPTR(
      &resRef,
      kIntraRecvStepFwdBlockIds,
      useReg,
      kernArgs.intraRecvStepFwdBlockIds);

  GET_RESOURCE_BUFPTR(
      &resRef, kFwdStepBlocksIds, useReg, kernArgs.fwdStepBlockIds);
  GET_RESOURCE_BUFPTR(
      &resRef, kFwdStepNumBlocks, useReg, kernArgs.fwdStepNumBlocks);
  GET_RESOURCE_BUFPTR(
      &resRef, kFwdStepRecvrNumBlocks, useReg, kernArgs.fwdStepRecvrNumBlocks);

  // medadata used only in combine
  GET_RESOURCE_BUFPTR(
      &resRef,
      kTmpSendRedStepNumPending,
      useReg,
      kernArgs.tmpSendRedStepNumPending);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpSendRedIdxNumSrcs, useReg, kernArgs.tmpSendRedIdxNumSrcs);
  GET_RESOURCE_BUFPTR(
      &resRef,
      kTmpSendRedIdxNumPendingSrcs,
      useReg,
      kernArgs.tmpSendRedIdxNumPendingSrcs);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpSendRedIdxSrcIds, useReg, kernArgs.tmpSendRedIdxSrcIds);

  GET_RESOURCE_BUFPTR(
      &resRef,
      kTmpIntraRedStepNumPending,
      useReg,
      kernArgs.tmpIntraRedStepNumPending);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpIntraRedIdxNumSrcs, useReg, kernArgs.tmpIntraRedIdxNumSrcs);
  GET_RESOURCE_BUFPTR(
      &resRef,
      kTmpIntraRedIdxNumPendingSrcs,
      useReg,
      kernArgs.tmpIntraRedIdxNumPendingSrcs);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpIntraRedIdxSrcIds, useReg, kernArgs.tmpIntraRedIdxSrcIds);

  GET_RESOURCE_BUFPTR(
      &resRef, kTmpRecvRedIdxNumSrcs, useReg, kernArgs.tmpRecvRedIdxNumSrcs);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpRecvRedIdxSrcIds, useReg, kernArgs.tmpRecvRedIdxSrcIds);

  if (!skipRem) {
    GET_RESOURCE_REM_BUFPTRS(&resRef, kTmpRecvBuff, kernArgs.remTmpRecvBuffs);
    GET_RESOURCE_REM_BUFPTRS(
        &resRef, kTmpIntraRecvBuff, kernArgs.remTmpIntraRecvBuffs);
  }

  kernArgs.kSync = resRef.kSync;
}
} // namespace ctran::alltoallvdedup
