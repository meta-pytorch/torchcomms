// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>
#include <cstddef>

#include "comms/ctran/algos/AllToAllvDedup/CommonDev.h"
#include "comms/ctran/algos/AllToAllvDedup/ExecCommon.cuh"
#include "comms/ctran/algos/AllToAllvDedup/IndexMapDev.cuh"
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/algos/AllToAllvDedup/WorkerGroupDev.cuh"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/algos/common/MultiTbSyncDev.cuh"
#include "comms/ctran/commstate/CommStateXDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"

namespace ctran::alltoallvdedup {
using namespace ctran::algos;

__device__ inline int countMaps(
    const int* inMaps,
    const int numMaps,
    const int b,
    const int stride,
    int* mapIds) {
  // TODO: how to parallelize while maintain the order of mapId
  int count = 0;
  for (auto i = 0; i < numMaps; i++) {
    const auto rIdx = inMaps[i * stride + b];
    if (rIdx != -1) {
      mapIds[count++] = i;
    }
  }
  // DEBUG only; can be deleted later
  for (auto i = count; i < numMaps; i++) {
    mapIds[i] = -1;
  }
  return count;
}

__device__ inline void prepareTmpSendIdx(
    ExecKernArgs& args,
    const WorkerGroup& sendG) {
  const auto workerId = sendG.workerId();
  const auto numWorkers = sendG.numWorkers;

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE("workerId %d/%d starts \n", workerId, numWorkers);
  }

  const auto nNodes = statex->nNodes();
  const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;
  const auto maxNumSteps = args.pArgs.maxNumSteps;
  const auto maxNumStepBlks = args.pArgs.maxNumStepBlks;

  const auto* sendIdx = args.execArgs.sendIdx;
  // nNodes; used to trackblocks to be forwarded to remote node in exec()
  auto* tmpNumSendIdx = args.tmpNumSendIdx;
  auto* tmpNumSendIdxH = args.tmpNumSendIdxH;
  // nNodes * maxNumSteps
  auto* tmpSendIdx = args.tmpSendIdx;
  // nNodes * maxNumSteps
  auto* tmpSendRedStepNumPending = args.tmpSendRedStepNumPending;

  // convert tmpIdx for each recv node
  for (auto recvNode = workerId; recvNode < nNodes; recvNode += numWorkers) {
    const auto nodeOffset = recvNode * totalNumSendBlocks;
    const auto count = IndexMapDev::transpose(
        &sendIdx[nodeOffset], totalNumSendBlocks, &tmpSendIdx[nodeOffset]);
    if (threadIdx.x == 0) {
      tmpNumSendIdx[recvNode] = count;
      tmpNumSendIdxH[recvNode] = count;
      CTRAN_DEV_TRACE("tmpNumSendIdx[%d] %d\n", recvNode, count);
    }

    // update per step count
    const auto remain = count % maxNumStepBlks;
    const auto numSteps = (count + maxNumStepBlks - 1) / maxNumStepBlks;
    for (auto s = threadIdx.x; s < maxNumSteps; s += blockDim.x) {
      const auto idx = recvNode * maxNumSteps + s;
      if (s < numSteps - 1) {
        tmpSendRedStepNumPending[idx] = maxNumStepBlks;
      } else if (s == numSteps - 1) {
        tmpSendRedStepNumPending[idx] = remain;
      } else {
        tmpSendRedStepNumPending[idx] = 0;
      }
    }
    CTRAN_DEV_TRACE_IF(
        threadIdx.x == 0,
        "updated tmpSendRedStepNumPending: node %d numSteps %d countPerStep %d, last %d\n",
        recvNode,
        numSteps,
        maxNumStepBlks,
        remain);
  }

  // prepare reduce vector for each block
  const auto tid = threadIdx.x + workerId * blockDim.x;
  const auto numThreads = blockDim.x * numWorkers;
  for (auto b = tid; b < totalNumSendBlocks; b += numThreads) {
    // totalNumSendBlocks * nNodes
    auto* mapIds = args.tmpSendRedIdxSrcIds + b * nNodes;
    const auto count =
        countMaps(sendIdx, nNodes, b, totalNumSendBlocks, mapIds);

    args.tmpSendRedIdxNumSrcs[b] = count;
    args.tmpSendRedIdxNumPendingSrcs[b] = count;
    CTRAN_DEV_TRACE_IF(
        b < 20 && count > 0,
        "tmpSendRedIdxNumSrcs[%d] %d/%d, srcIds %d %d...\n",
        b,
        count,
        nNodes,
        mapIds[0],
        mapIds[1]);
  }
}

__device__ inline void prepareTmpIntraFwdIdx(
    ExecKernArgs& args,
    const WorkerGroup& intraFwdG) {
  const auto workerId = intraFwdG.workerId();
  const auto numWorkers = intraFwdG.numWorkers;

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE("workerId %d/%d starts \n", workerId, numWorkers);
  }

  // Prepare tmpFwdIdx[][myNodes], indicating number of blocks to be forwarded
  // to each local rank from my node (i.e., sendRank itself)
  const auto myNode = statex->node();
  const auto nNodes = statex->nNodes();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;
  const auto maxNumStepBlks = args.pArgs.maxNumStepBlks;
  const auto maxNumSteps = args.pArgs.maxNumSteps;

  const int* fwdIdx = args.execArgs.fwdIdx;
  const auto* intraFwdIdx = fwdIdx + myNode * totalNumSendBlocks;

  // nLocalRanks; used to track number of blocks to be forwarded to each local
  // rank in exec()
  int* tmpNumIntraFwdIdx = args.tmpNumIntraFwdIdx;
  int* tmpNumIntraFwdIdxH = args.tmpNumIntraFwdIdxH;
  int* tmpFwdIdx = args.tmpFwdIdx;

  // nLocalRanks * maxNumSteps; used to track whether a remote chunk can be
  // freed after local reduction in combine intraFwdRed
  int* tmpIntraRedStepNumPending = args.tmpIntraRedStepNumPending;

  for (auto localRank = workerId; localRank < nLocalRanks;
       localRank += numWorkers) {
    const auto slotId = localRank * nNodes + myNode;
    const auto offset = slotId * totalNumSendBlocks;

    const auto count = IndexMapDev::transpose(
        &fwdIdx[offset], totalNumSendBlocks, &tmpFwdIdx[offset]);
    if (threadIdx.x == 0) {
      tmpNumIntraFwdIdx[localRank] = count;
      tmpNumIntraFwdIdxH[localRank] = count;
      CTRAN_DEV_TRACE("tmpNumIntraFwdIdx[%d] %d\n", localRank, count);
    }

    // update per step count
    const auto remain = count % maxNumStepBlks;
    const auto numSteps = (count + maxNumStepBlks - 1) / maxNumStepBlks;
    for (auto s = threadIdx.x; s < maxNumSteps; s += blockDim.x) {
      const auto idx = localRank * maxNumSteps + s;
      if (s < numSteps - 1) {
        tmpIntraRedStepNumPending[idx] = maxNumStepBlks;
      } else if (s == numSteps - 1) {
        tmpIntraRedStepNumPending[idx] = remain;
      } else {
        tmpIntraRedStepNumPending[idx] = 0;
      }
    }
    CTRAN_DEV_TRACE_IF(
        threadIdx.x == 0,
        "updated tmpIntraRedStepNumPending: localRank %d numSteps %d count %d, last %d\n",
        localRank,
        numSteps,
        maxNumStepBlks,
        remain);
  }

  // prepare reduce vector for each block
  const auto tid = threadIdx.x + workerId * blockDim.x;
  const auto numThreads = blockDim.x * numWorkers;

  for (auto b = tid; b < totalNumSendBlocks; b += numThreads) {
    // totalNumSendBlocks * nLocalRanks
    auto* mapIds = args.tmpIntraRedIdxSrcIds + b * nLocalRanks;
    // tmpFwdIdx format: nLocalRanks * nNodes * totalNumSendBlocks
    const auto count = countMaps(
        intraFwdIdx, nLocalRanks, b, nNodes * totalNumSendBlocks, mapIds);
    // totalNumSendBlocks
    args.tmpIntraRedIdxNumSrcs[b] = count;
    args.tmpIntraRedIdxNumPendingSrcs[b] = count;
    CTRAN_DEV_TRACE_IF(
        b < 20 && count > 0,
        "tmpIntraRedIdxNumSrcs[%d] %d/%d, srcIds %d %d...\n",
        b,
        count,
        nLocalRanks,
        mapIds[0],
        mapIds[1]);
  }
}

__device__ inline void prepareTmpFwdIdx(
    ExecKernArgs& args,
    const WorkerGroup& fwdG) {
  const auto workerId = fwdG.workerId();
  const auto numWorkers = fwdG.numWorkers;

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE("workerId %d/%d starts \n", workerId, numWorkers);
  }

  // Prepare tmpNumFwdIdx[node] for each localRank and node, indicating
  // number of blocks to be forwarded to each local rank from the node.
  const auto myNode = statex->node();
  const auto nNodes = statex->nNodes();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;

  const int* fwdIdx = args.execArgs.fwdIdx;
  auto* tmpNumFwdIdx = args.tmpNumFwdIdx;
  auto* tmpNumFwdIdxH = args.tmpNumFwdIdxH;

  for (auto node = workerId; node < nNodes; node += numWorkers) {
    // intraFwd is handled separately
    if (node == myNode) {
      continue;
    }

    // compute count of merged indices, as number of blocks from each send node
    // to all local ranks. used for progress tracking. IndexMapDev::countMerge()
    // counds a block only once if it exists in multiple merged lookupMaps.
    //
    // fwdIdx format:
    //              sendnode0    sendnode1
    // localrank0 | lookupMap  | lookupMap |...
    // localrank1 | lookupMap  | lookupMap |...
    const int* maps[CTRAN_MAX_NVL_PEERS];
    for (auto r = 0; r < nLocalRanks; r++) {
      const auto slotId = r * nNodes + node;
      maps[r] = &fwdIdx[slotId * totalNumSendBlocks];
    }
    const auto count =
        IndexMapDev::countMerge(maps, totalNumSendBlocks, nLocalRanks);
    if (threadIdx.x == 0) {
      tmpNumFwdIdx[node] = count;
      tmpNumFwdIdxH[node] = count;
      CTRAN_DEV_TRACE("tmpNumFwdIdx[%d] %d\n", node, count);
    }
  }
}
__device__ inline void prepareTmpRecvOffsets(
    ExecKernArgs& args,
    const WorkerGroup& recvG) {
  const auto workerId = recvG.workerId();
  const auto numWorkers = recvG.numWorkers;
  const auto nRanks = statex->nRanks();

  const auto numRecvBuckets = args.pArgs.numRecvBuckets;
  const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;
  const auto numOffsets = numRecvBuckets * nRanks;

  // numRecvBuckets * nRanks, already reset in prepare
  int* tmpRecvOffsets = args.tmpRecvOffsets;
  // nBuckets * nRanks * totalNumSendBlocks
  const auto recvIdx = args.execArgs.recvIdx;

  // compute count of indices received by each bucket from each sendRank, used
  // as offset for recv worker to copy block into corresponding locaiton in
  // recvBuff.
  for (auto i = workerId; i < numOffsets; i += numWorkers) {
    if (i == numOffsets - 1) {
      break; // skip last slotId since no more slot after it
    }

    // each worker counts number of blocks received per bucket per sendRank
    const auto count = IndexMapDev::count(
        &recvIdx[i * totalNumSendBlocks], totalNumSendBlocks);
    // threads increases later slots' offest in parallel
    for (auto nxt = i + 1 + threadIdx.x; nxt < numOffsets; nxt += blockDim.x) {
      atomicAdd(&tmpRecvOffsets[nxt], count);
    }
  }
}

__device__ inline void prepareTmpRecvRedIdx(
    ExecKernArgs& args,
    const WorkerGroup& recvG) {
  const auto workerId = recvG.workerId();
  const auto numWorkers = recvG.numWorkers;
  const auto nRanks = statex->nRanks();

  const auto numRecvBuckets = args.pArgs.numRecvBuckets;
  const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;
  // nBuckets * nRanks * totalNumSendBlocks
  const auto recvIdx = args.execArgs.recvIdx;

  // prepare reduce vector for each block
  for (auto sendRank = workerId; sendRank < nRanks; sendRank += numWorkers) {
    // nBuckets * nRanks * totalNumSendBlocks
    const auto* rankIdxMap = recvIdx + sendRank * totalNumSendBlocks;
    const auto mapStride = nRanks * totalNumSendBlocks;
    const auto rankOffset = sendRank * totalNumSendBlocks;
    for (auto b = threadIdx.x; b < totalNumSendBlocks; b += blockDim.x) {
      // nRanks * totalNumSendBlocks * numRecvBuckets
      auto* mapIds = args.tmpRecvRedIdxSrcIds + rankOffset * numRecvBuckets +
          b * numRecvBuckets;
      const auto count =
          countMaps(rankIdxMap, numRecvBuckets, b, mapStride, mapIds);
      // nRanks * totalNumSendBlocks
      args.tmpRecvRedIdxNumSrcs[rankOffset + b] = count;
      CTRAN_DEV_TRACE_IF(
          b < 20 && count > 0,
          "tmpRecvRedIdxNumSrcs[rank %d][%d/%d] %d/%d, srcIds: %d %d...\n",
          sendRank,
          b,
          totalNumSendBlocks,
          count,
          numRecvBuckets,
          mapIds[0],
          mapIds[1]);
    }
  }
}

__device__ inline void prepareTmpRecvIdx(
    ExecKernArgs& args,
    const WorkerGroup& recvG) {
  const auto workerId = recvG.workerId();
  const auto numWorkers = recvG.numWorkers;

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE("workerId %d/%d starts \n", workerId, numWorkers);
  }

  const auto numRecvBuckets = args.pArgs.numRecvBuckets;
  const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;
  const auto nRanks = statex->nRanks();
  const auto myNode = statex->node();

  // nBuckets * nRanks * totalNumSendBlocks
  const auto recvIdx = args.execArgs.recvIdx;
  // nLocalRanks
  int* tmpNumFwdRecvIdx = args.tmpNumFwdRecvIdx;
  int* tmpNumIntraRecvIdx = args.tmpNumIntraRecvIdx;
  int* tmpNumFwdRecvIdxH = args.tmpNumFwdRecvIdxH;
  int* tmpNumIntraRecvIdxH = args.tmpNumIntraRecvIdxH;

  // Count the number of received blocks from each forward local rank for
  // progress tracking. Each worker counts for different sendRank
  for (auto sendRank = workerId; sendRank < nRanks; sendRank += numWorkers) {
    const auto lr = statex->localRank(sendRank);
    const auto node = statex->node(sendRank);

    // compute count of merged indices, as number of unique blocks from each
    // send rank to all local buckets
    const int* maps[MAX_NUM_RECV_BUCKETS];
    for (auto bkt = 0; bkt < numRecvBuckets; bkt++) {
      const auto slotId = bkt * nRanks + sendRank;
      maps[bkt] = &recvIdx[slotId * totalNumSendBlocks];
    }

    // there is no duplicate blocks beween different sendRanks, thus sum up as
    // number of blocks forwarded from each local rank
    const auto count =
        IndexMapDev::countMerge(maps, totalNumSendBlocks, numRecvBuckets);
    if (threadIdx.x == 0) {
      // track intraFwd and fwd separately
      if (node == myNode) {
        tmpNumIntraRecvIdx[lr] = count;
        tmpNumIntraRecvIdxH[lr] = count;
      } else {
        atomicAdd(&tmpNumFwdRecvIdx[lr], count);
        atomicAdd(&tmpNumFwdRecvIdxH[lr], count);
      }
    }
  }
}

__device__ inline void
resetTmpRecvIdx(ExecKernArgs& args, const int nRanks, const int nLocalRanks) {
  const auto numRecvBuckets = args.pArgs.numRecvBuckets;

  // nLocalRanks
  int* tmpNumFwdRecvIdx = args.tmpNumFwdRecvIdx;
  int* tmpRecvOffsets = args.tmpRecvOffsets;
  const auto numOffsets = numRecvBuckets * nRanks;

  for (auto i = threadIdx.x; i < numOffsets; i += blockDim.x) {
    tmpRecvOffsets[i] = 0;
  }
  for (auto i = threadIdx.x; i < nLocalRanks; i += blockDim.x) {
    tmpNumFwdRecvIdx[i] = 0;
  }
}

// single thread block to reset all sync objects
__device__ inline void resetSync(ExecKernArgs& args, const bool kIsExec) {
  const auto warpId = threadIdx.x / comms::device::kWarpSize;
  const auto numWraps = blockDim.x / comms::device::kWarpSize;
  const auto& config = args.config;

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "workerId %d/%d starts numSendWorkers %d,%d numFwdWorkers %d numRecvWorkers %d,%d\n",
        0,
        1,
        config.numSendGroups,
        config.numSendWorkers,
        config.numFwdWorkers,
        config.numRecvGroups,
        config.numRecvWorkers);
  }

  const int numTypes = 5;
  int numGroups[numTypes];
  numGroups[(int)WorkerGroupType::kSend] = config.numSendGroups;
  numGroups[(int)WorkerGroupType::kFwd] = 1;
  numGroups[(int)WorkerGroupType::kRecv] = config.numRecvGroups;
  numGroups[(int)WorkerGroupType::kIntraFwd] = 1;
  numGroups[(int)WorkerGroupType::kIntraRecv] = 1;

  // Now all threads reset the barrier objects in parallel
  const auto numSyncs = numTypes * MAX_NUM_GROUPS_PER_ROLE;
  for (int i = (int)threadIdx.x; i < numSyncs; i += blockDim.x) {
    const int t = i % numTypes;
    const int g = i / numTypes;
    if (g < numGroups[t]) {
      // reset only used counters
      auto* sync = getWgSync(args, static_cast<WorkerGroupType>(t), g);
      for (int c = 0; c < WorkerGroupSync::kNumCnts; c++) {
        MultiTbSyncDev::reset(&sync->cnts[c]);
      }
    }
  }

  const auto nNodes = statex->nNodes();
  const auto nLocalRanks = statex->nLocalRanks();

  // Reset GPE sync; always before both next kernel and GPE exec.
  GpeKernelSync* nNodeSyncs[2];
  nNodeSyncs[0] = args.kSync.sendGKSyncs;
  nNodeSyncs[1] = args.kSync.recvGKSyncs;
  for (auto i = warpId; i < nNodes * 2; i += numWraps) {
    const auto syncId = i / nNodes;
    const auto n = i % nNodes;
    auto sync = nNodeSyncs[syncId] + n;
    // always worker 0 to sync with GPE
    GpeKernelSyncDev::resetWarp(sync, 1);
  }

  GpeKernelSync* nLocalRankSyncs[3];
  nLocalRankSyncs[0] = args.kSync.intraFwdGKSyncs;
  nLocalRankSyncs[1] = args.kSync.recvCopyGKSyncs;
  nLocalRankSyncs[2] = args.kSync.intraRecvCopyGKSyncs;
  for (auto i = warpId; i < nLocalRanks * 3; i += numWraps) {
    const auto syncId = i / nLocalRanks;
    const auto r = i % nLocalRanks;
    auto sync = nLocalRankSyncs[syncId] + r;
    // always worker 0 to sync with GPE
    GpeKernelSyncDev::resetWarp(sync, 1);
  }

  // Unlike other SpscP2pSync objects, consumer of intraRedSync (sendRed worker)
  // doesn't reset after waited post because the data buffer (recvBuff) doesn't
  // need to be reused by producer. Thus, this is a producer-one-way-post sync
  // at combine(), and we need reset before the next combine().
  if (threadIdx.x == 0) {
    args.kSync.intraRedSync->cnt = -1;
  }
}

// Reset any tmpIdx before prepare; expect 1 thread block
__global__ void ncclKernelAllToAllvDedupPrepareReset(
    ExecKernArgs args,
    int nRanks,
    int nLocalRanks) {
  resetTmpRecvIdx(args, nRanks, nLocalRanks);
}

// Prepare metadata and reset sync objects; used before exec
__global__ void ncclKernelAllToAllvDedupPrepare(
    CtranAlgoDeviceState* devState,
    ExecKernArgs args,
    PrepareConfig config,
    const int roles) {
  // TODO: move it into a generic routine for all kernels
  devState->opCount = args.opCount;
  devStateLoadToShm(devState);

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE("start kernel\n");
  }

  constexpr bool kIsExec = true;

  WorkerGroup sendG, intraFwdG, fwdG, recvG, recvOffG, recvRedG, resetG;
  assignWorkerGroup(0, 1, config.numSendIdxWorkers, sendG);
  assignWorkerGroup(sendG.end + 1, 1, config.numIntraFwdIdxWorkers, intraFwdG);
  assignWorkerGroup(intraFwdG.end + 1, 1, config.numFwdIxWorkers, fwdG);
  assignWorkerGroup(fwdG.end + 1, 1, config.numRecvIdxWorkers, recvG);
  assignWorkerGroup(recvG.end + 1, 1, config.numRecvOffsetWorkers, recvOffG);
  assignWorkerGroup(recvOffG.end + 1, 1, config.numRecvRedIdxWorkers, recvRedG);
  assignWorkerGroup(recvRedG.end + 1, 1, config.numResetSyncWorkers, resetG);

  if (sendG.contains(blockIdx.x) &&
      prepareRoleContains(roles, PrepareRole::kPrepTmpSendIdx)) {
    prepareTmpSendIdx(args, sendG);
  } else if (
      intraFwdG.contains(blockIdx.x) &&
      prepareRoleContains(roles, PrepareRole::kPrepTmpIntraFwdIdx)) {
    prepareTmpIntraFwdIdx(args, intraFwdG);
  } else if (
      fwdG.contains(blockIdx.x) &&
      prepareRoleContains(roles, PrepareRole::kPrepTmpFwdIdx)) {
    prepareTmpFwdIdx(args, fwdG);
  } else if (
      recvG.contains(blockIdx.x) &&
      prepareRoleContains(roles, PrepareRole::kPrepTmpRecvIdx)) {
    prepareTmpRecvIdx(args, recvG);
  } else if (
      recvOffG.contains(blockIdx.x) &&
      prepareRoleContains(roles, PrepareRole::kPrepTmpRecvOffsets)) {
    prepareTmpRecvOffsets(args, recvOffG);
  } else if (
      recvRedG.contains(blockIdx.x) &&
      prepareRoleContains(roles, PrepareRole::kPrepTmpRecvRedIdx)) {
    prepareTmpRecvRedIdx(args, recvRedG);
  } else if (
      resetG.contains(blockIdx.x) &&
      prepareRoleContains(roles, PrepareRole::kResetSync)) {
    resetSync(args, kIsExec);
  }

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE("exit kernel\n");
  }
}
} // namespace ctran::alltoallvdedup
