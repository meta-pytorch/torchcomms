// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>
#include <cstddef>

#include "comms/ctran/algos/AllToAllvDedup/CommonDev.h"
#include "comms/ctran/algos/AllToAllvDedup/FwdGroupSyncDev.cuh"
#include "comms/ctran/algos/AllToAllvDedup/WorkerSyncDev.cuh"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/algos/common/MPSCTbSyncDev.cuh"
#include "comms/ctran/commstate/CommStateXDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"

using namespace ctran::algos;

namespace ctran::alltoallvdedup {
__device__ inline int
bucketToLocalBucket(const ExecKernArgs& args, int bucket, int nLocalRanks) {
  return bucket & (args.pArgs.numRecvBuckets * nLocalRanks - 1);
}

__device__ inline void localRankToLocalBucketRange(
    const ExecKernArgs& args,
    int localRank,
    int& min,
    int& max) {
  min = localRank * args.pArgs.numRecvBuckets;
  max = (localRank + 1) * args.pArgs.numRecvBuckets - 1;
}

#define CHECK_VALID_STATE(idx, totalPending, numPendingBlocks) \
  do {                                                         \
    if (totalPending < 0 || numPendingBlocks[idx] < 0) {       \
      CTRAN_DEV_TRACE(                                         \
          "ERROR! %s %d < 0 or %s[%d] %ld < 0\n",              \
          #totalPending,                                       \
          totalPending,                                        \
          #numPendingBlocks,                                   \
          idx,                                                 \
          numPendingBlocks[idx]);                              \
    }                                                          \
  } while (0)

#define LOG_STATE_UPDATE_STEP(CLASS_NAME)                                                          \
  do {                                                                                             \
    CTRAN_DEV_TRACE(                                                                               \
        CLASS_NAME                                                                                 \
        ": finished steps[%d] %d numBlocks %d, updated numPendingBlocks %ld numTotalPending %d\n", \
        idx,                                                                                       \
        old,                                                                                       \
        numBlocks,                                                                                 \
        numPendingBlocks[idx],                                                                     \
        numTotalPending);                                                                          \
  } while (0)

// FIXME: allocate as resounrce managed buffer to allow arbitrary nnodes
#define MAX_NNODES_PER_GROUP 64

struct ProgressSendState {
  int64_t numPendingBlocks[MAX_NNODES_PER_GROUP];
  int64_t totalToSend[MAX_NNODES_PER_GROUP];
  int64_t lastBlockIdx[MAX_NNODES_PER_GROUP];
  int steps[MAX_NNODES_PER_GROUP];
  int numTotalPending;

  __device__ __inline__ void
  updateStep(const int idx, const int b, const int numBlocks) {
    const auto old = steps[idx];
    steps[idx]++;
    numPendingBlocks[idx] -= numBlocks;
    lastBlockIdx[idx] = b;
    if (numPendingBlocks[idx] == 0) {
      numTotalPending--;
    }
    if (threadIdx.x == 0) {
      LOG_STATE_UPDATE_STEP("progressSend");
      // TODO: add control to skip out-of-range check
      CHECK_VALID_STATE(idx, numTotalPending, numPendingBlocks);
    }
  }
};

struct ProgressFwdState {
  // number of pending blocks to be forwarded for other nodes
  int64_t numPendingBlocks[MAX_NNODES_PER_GROUP];
  int steps[MAX_NNODES_PER_GROUP];
  // track numBlocks forwarded to each local rank for debugging purpose, can be
  // disabled to save registers
  int64_t numFwdBlocksToRank[CTRAN_MAX_NVL_PEERS];
  int numTotalPending;

  __device__ __inline__ void updateStep(const int idx, const int numBlocks) {
    const auto old = steps[idx];
    steps[idx]++;
    numPendingBlocks[idx] -= numBlocks;
    if (numPendingBlocks[idx] == 0) {
      numTotalPending--;
    }
    if (threadIdx.x == 0) {
      LOG_STATE_UPDATE_STEP("progressFwd");
      // TODO: add control to skip out-of-range check
      CHECK_VALID_STATE(idx, numTotalPending, numPendingBlocks);
    }
  }
};

struct ProgressRecvState {
  // number of pending blocks to be received from local forwarding ranks
  // Use signed int to check potential out-of-range numBlocks in the pipeline
  int64_t numPendingBlocks[CTRAN_MAX_NVL_PEERS];
  int steps[CTRAN_MAX_NVL_PEERS];
  int numTotalPending;

  __device__ __inline__ void updateStep(const int idx, const int numBlocks) {
    const auto old = steps[idx];
    steps[idx]++;
    numPendingBlocks[idx] -= numBlocks;
    if (numPendingBlocks[idx] == 0) {
      numTotalPending--;
    }
    if (threadIdx.x == 0) {
      LOG_STATE_UPDATE_STEP("progressRecv");

      // TODO: add control to skip out-of-range check
      CHECK_VALID_STATE(idx, numTotalPending, numPendingBlocks);
    }
  }
};

struct ProgressIntraFwdState {
  // number of pending blocks to be forwarded to local receiving ranks
  int64_t numPendingBlocks[CTRAN_MAX_NVL_PEERS];
  int steps[CTRAN_MAX_NVL_PEERS];
  int numTotalPending;

  __device__ __inline__ void updateStep(int idx, const int numBlocks) {
    const auto old = steps[idx];
    steps[idx]++;
    numPendingBlocks[idx] -= numBlocks;
    if (numPendingBlocks[idx] == 0) {
      numTotalPending--;
    }
    if (threadIdx.x == 0) {
      CTRAN_DEV_TRACE(
          "progressIntraFwd: finished step %d numBlock %d to localRecvRank %d, updated numPendingBlocks %ld numTotalPending %d\n",
          old,
          numBlocks,
          idx,
          numPendingBlocks[idx],
          numTotalPending);

      // TODO: add control to skip out-of-range check
      CHECK_VALID_STATE(idx, numTotalPending, numPendingBlocks);
    }
  }
};

struct GroupLocalRanks {
  int start;
  int end;
  int numRanks;
  __device__ inline bool contains(const int r) const {
    return r >= start && r <= end;
  }
};

struct GroupNodes {
  int start;
  int end;
  int numNodes;
};

__device__ inline void getGroupRange(
    const int totalNum,
    const int numGroups,
    const int groupId,
    int& myStart,
    int& myEnd,
    int& myNum) {
  const int numPerGroup = totalNum / numGroups;
  myStart = groupId * numPerGroup;
  myEnd = (groupId == numGroups - 1) ? totalNum - 1 : myStart + numPerGroup - 1;
  myNum = myEnd - myStart + 1;
}

__device__ inline GroupLocalRanks assignGroupLocalRanks(
    ExecKernArgs& args,
    const int groupId,
    const int numGroups) {
  GroupLocalRanks range = {-1, -1, -1};
  const auto nLocalRanks = statex->nLocalRanks();
  getGroupRange(
      nLocalRanks, numGroups, groupId, range.start, range.end, range.numRanks);
  return range;
}

__device__ inline GroupNodes
assignGroupNodes(ExecKernArgs& args, const int groupId, const int numGroups) {
  GroupNodes range = {-1, -1, -1};
  const auto nNodes = statex->nNodes();
  getGroupRange(
      nNodes, numGroups, groupId, range.start, range.end, range.numNodes);
  return range;
}

template <typename T>
__device__ inline void updateProgressSendState(
    ExecKernArgs& args,
    int workerId,
    int groupId,
    GroupNodes& range,
    ProgressSendState& state) {
  const auto myNode = statex->node();
  const auto nNodes = statex->nNodes();

  for (auto n = 0; n < range.numNodes; n++) {
    state.numPendingBlocks[n] = 0; // numSendBlocks for each node
    state.totalToSend[n] = 0;
    state.lastBlockIdx[n] = 0;
    state.steps[n] = 0;
  }
  const auto numWorkers = args.config.numSendWorkers;

  const auto* sendIdx = args.execArgs.sendIdx;
  auto* tmpSendIdx = reinterpret_cast<int*>(args.tmpSendIdx);

  for (auto j = threadIdx.x + blockDim.x * workerId;
       j < nNodes * args.pArgs.totalNumSendBlocks;
       j += blockDim.x * numWorkers) {
    tmpSendIdx[j] = -1;
  }

  WorkerSyncDev::sync(
      args.kSync.workerSync, groupId, numWorkers, WorkerSync::kSend, 0);
  __threadfence();

  for (int n = 0; n < nNodes; n++) {
    for (auto j = threadIdx.x + blockDim.x * workerId;
         j < args.pArgs.totalNumSendBlocks;
         j += blockDim.x * numWorkers) {
      int sendIdxj = sendIdx[n * args.pArgs.totalNumSendBlocks + j];
      if (sendIdxj != -1) {
        tmpSendIdx[n * args.pArgs.totalNumSendBlocks + sendIdxj] = j;
      }
    }
  }
  WorkerSyncDev::sync(
      args.kSync.workerSync, groupId, numWorkers, WorkerSync::kSend, 1);
  __threadfence();

  __shared__ int maxval;
  for (auto n = 0; n < range.numNodes; n++) {
    if (threadIdx.x == 0) {
      maxval = 0;
    }
    __syncthreads();
    auto recvNode = n + range.start;

    if (recvNode != myNode) {
      for (auto b = threadIdx.x; b < args.pArgs.totalNumSendBlocks;
           b += blockDim.x) {
        if (tmpSendIdx[recvNode * args.pArgs.totalNumSendBlocks + b] != -1) {
          atomicMax(&maxval, b + 1);
        }
      }
    }
    __syncthreads();
    state.numPendingBlocks[n] = maxval;
    __syncthreads();
  }

  state.numTotalPending = 0;
  for (int n = 0; n < range.numNodes; n++) {
    if (state.numPendingBlocks[n] > 0) {
      state.numTotalPending++;
      state.totalToSend[n] = state.numPendingBlocks[n];
      if (threadIdx.x == 0) {
        int recvNode = n + range.start;
        CTRAN_DEV_TRACE(
            "groupId %d loaded numPendingBlocks %ld to recvNode %d, numTotalPending %d\n",
            groupId,
            state.numPendingBlocks[n],
            recvNode,
            state.numTotalPending);
      }
    }
  }
}

template <typename T>
__device__ inline void progressSendCopy(
    ExecKernArgs& args,
    int groupId,
    int workerId,
    GroupNodes& range) {
  const auto myNode = statex->node();
  const auto rank = statex->rank();

  // Each group handles different nodeIds
  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "groupId %d/%d workerId %d/%d handles nodes start %d - end %d, %d nodes\n",
        groupId,
        args.config.numSendGroups,
        workerId,
        args.config.numSendWorkers,
        range.start,
        range.end,
        range.numNodes);
  }

  ProgressSendState state;
  updateProgressSendState<T>(args, workerId, groupId, range, state);

  // Each group iterates on different node range;
  while (state.numTotalPending > 0) {
    for (auto n = 0; n < range.numNodes; n++) {
      const auto recvNode = n + range.start;
      if (state.numPendingBlocks[n] == 0 || recvNode == myNode) {
        // Copied all blocks for this node, move to next
        // Skip blocks sending to local ranks, since they will be handled by
        // forward workers.
        continue;
      }

      // Wait host side to post tmpSendBuff ready
      auto sync = args.kSync.sendCopyGKSyncs + recvNode;

      const auto step = state.steps[n];
      bool posted = false;
      posted = ctran::algos::GpeKernelSyncDev::checkPost(sync, workerId, step);
      if (!posted) {
        // If no available chunk is posted for this node, move to next
        continue;
      }
      if (threadIdx.x == 0) {
        CTRAN_DEV_TRACE(
            "sendCopyGKSyncs[%d] %p posted step %d\n", recvNode, sync, step);
      }

      const auto numWorkers = args.config.numSendWorkers;
      const auto* tmpSendIdx = reinterpret_cast<int*>(args.tmpSendIdx);

      // Copy blocks into corresponding chunk in tmpSendBuff
      void* tmpSendBuff =
          getTmpChunkPtr(args.config, args.tmpSendBuff, step, recvNode);

      // Count total number of blocks to copy to the recvNode, so we can
      // define the range for bitmaps and data to keep contig send chunk
      // without padding
      const auto maxNumBlocksPerChunk =
          getMaxNumBlocksPerChunk<T>(args.config, args.pArgs);
      const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;
      auto numToCopy = state.numPendingBlocks[n] < maxNumBlocksPerChunk
          ? state.numPendingBlocks[n]
          : maxNumBlocksPerChunk;

      FwdChkHdr* hdr = getTmpChunkHdr(tmpSendBuff);
      const auto headerLen = getChunkHeaderLen(maxNumBlocksPerChunk);
      int* blockIds = getTmpChunkBlockIds(tmpSendBuff, maxNumBlocksPerChunk);
      T* tmpBlocks =
          reinterpret_cast<T*>(getTmpChunkData(headerLen, tmpSendBuff));

      auto startBlock = state.totalToSend[n] - state.numPendingBlocks[n];
      auto endBlock = startBlock + numToCopy;

      const auto blockCount = static_cast<int>(args.pArgs.blockCount);
      auto threadId = threadIdx.x + workerId * blockDim.x;
      auto numThreads = blockDim.x * numWorkers;
      for (int b = threadId; b < numToCopy; b += numThreads) {
        auto blockSendIdx =
            tmpSendIdx[recvNode * totalNumSendBlocks + b + startBlock];
        blockIds[b] = rank * totalNumSendBlocks + blockSendIdx;
      }
      const T* sData = reinterpret_cast<const T*>(args.execArgs.sendBuff);
      for (auto b = workerId; b < numToCopy; b += numWorkers) {
        auto blockSendIdx =
            tmpSendIdx[recvNode * totalNumSendBlocks + b + startBlock];
        ctranKernCopy<T>(
            sData + blockSendIdx * blockCount,
            tmpBlocks + b * blockCount,
            blockCount,
            0,
            1);
      }

      if (workerId == 0 && threadIdx.x == 0) {
        hdr->numBlocks = numToCopy;
        hdr->sendRank = rank;
      }
      ctran::algos::GpeKernelSyncDev::complete(sync, workerId, step);
      state.updateStep(
          n, tmpSendIdx[recvNode * totalNumSendBlocks + endBlock], numToCopy);
    }
  }
}

__device__ inline void syncRemRecvCopy(
    ExecKernArgs& args,
    const int workerId,
    const int recvLocalRank,
    const int sendRank,
    const int numFwdBlocks,
    const int chunkIdx) {
  const auto myLocalRank = statex->localRank();

  // Update total numBlocks for this step
  void* remTmpRecvBuff = getTmpChunkPtrByIdx(
      args.config, args.remTmpRecvBuffs[recvLocalRank], chunkIdx, myLocalRank);
  FwdChkHdr* hdr = getTmpChunkHdr(remTmpRecvBuff);
  if (workerId == 0) {
    hdr->sendRank = sendRank;
    hdr->numBlocks = numFwdBlocks;
  }

  // Notify recv rank to consume; recv rank will wait on all fwd workers
  auto remSync = getFwdRecvSync(
      args.config, args.kSync, recvLocalRank, myLocalRank, chunkIdx);
  MPSCTbSyncDev::post(remSync, workerId);
}

__device__ inline void fwdSyncRemRecvCopy(
    ExecKernArgs& args,
    const int workerId,
    const int step,
    const int recvLocalIdx,
    const int sendRank,
    const int numFwdBlocks,
    const int chunkIdx,
    ProgressFwdState& state) {
  // for cross-node forwarding, each group handles blocks for all local ranks
  // but from different node
  const auto recvLocalRank = recvLocalIdx;

  syncRemRecvCopy(
      args, workerId, recvLocalRank, sendRank, numFwdBlocks, chunkIdx);
  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "Updated recvLocalRank %d chunkIdx %d: sendRank %d numBlocks %d, numFwdBlocksToRank %ld step %d\n",
        recvLocalRank,
        chunkIdx,
        sendRank,
        numFwdBlocks,
        state.numFwdBlocksToRank[recvLocalIdx],
        step);
  }
}

template <typename T>
__device__ inline void updateProgressIntraFwdState(
    ExecKernArgs& args,
    int groupId,
    const GroupLocalRanks& range,
    ProgressIntraFwdState& state) {
  const auto* numSendBlocks = args.execArgs.numSendBlocks;
  const auto myNode = statex->node();

  state.numTotalPending = 0;
  for (auto r = range.start; r <= range.end; r++) {
    const auto recvRank = statex->localRankToRank(r, myNode);

    state.numPendingBlocks[r] = 0;
    state.steps[r] = 0;
    if (numSendBlocks[recvRank] > 0) {
      state.numPendingBlocks[r] = numSendBlocks[recvRank];
      state.numTotalPending++;
    }
    if (threadIdx.x == 0) {
      CTRAN_DEV_TRACE(
          "groupId %d loaded numSendBlocks[%d] %ld -> numPendingBlocks %ld to intra-node recvRank %d, numTotalPending %d\n",
          groupId,
          recvRank,
          numSendBlocks[recvRank],
          state.numPendingBlocks[r],
          recvRank,
          state.numTotalPending);
    }
  }
}

template <typename T>
__device__ inline void updateProgressFwdState(
    ExecKernArgs& args,
    int groupId,
    const GroupNodes& range,
    ProgressFwdState& state) {
  const auto myLocalRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto myNode = statex->node();

  const auto* numForwardBlocks = args.execArgs.numForwardBlocks;

  for (auto r = 0; r < nLocalRanks; r++) {
    state.numFwdBlocksToRank[r] = 0;
  }

  state.numTotalPending = 0;
  for (auto n = 0; n < range.numNodes; n++) {
    state.numPendingBlocks[n] = 0;
    state.steps[n] = 0;
    const auto sendNode = n + range.start;
    // Intra-node forwarding is handled separately
    if (sendNode == myNode) {
      continue;
    }
    const auto sendRank = statex->localRankToRank(myLocalRank, sendNode);
    if (numForwardBlocks[sendRank] > 0) {
      state.numPendingBlocks[n] = numForwardBlocks[sendRank];
      state.numTotalPending++;
    }
    if (threadIdx.x == 0) {
      CTRAN_DEV_TRACE(
          "groupId %d loaded numForwardBlocks[%d] %ld -> numPendingBlocks %ld from sendNode %d, numTotalPending %d\n",
          groupId,
          n,
          numForwardBlocks[sendRank],
          state.numPendingBlocks[n],
          sendNode,
          state.numTotalPending);
    }
  }
}

__device__ inline void intraFwdSyncRemRecvCopy(
    ExecKernArgs& args,
    const int workerId,
    const int recvLocalRank,
    const int numFwdBlocks,
    const int chunkIdx,
    ProgressIntraFwdState& state) {
  const auto myRank = statex->rank();

  syncRemRecvCopy(
      args, workerId, recvLocalRank, myRank, numFwdBlocks, chunkIdx);
  state.updateStep(recvLocalRank, numFwdBlocks);
}

template <typename T>
__device__ inline void progressIntraFwd(
    ExecKernArgs& args,
    int groupId,
    int workerId,
    const GroupLocalRanks& range) {
  const auto myRank = statex->rank();
  const auto myLocalRank = statex->localRank();
  const auto numWorkers = args.config.numFwdWorkers;
  const auto numGroups = args.config.numFwdGroups;

  // Each group handles different different localRanks for intra-node direct
  // forwarding.
  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "groupId %d workerId %d handles localRanks %d - %d, %d localRanks\n",
        groupId,
        workerId,
        range.start,
        range.end,
        range.numRanks);
  }

  ProgressIntraFwdState state;
  updateProgressIntraFwdState<T>(args, groupId, range, state);

  const auto* fwdIdx = args.execArgs.fwdIdx;
  const auto myNode = statex->node();
  const auto nNodes = statex->nNodes();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto& config = args.config;
  const auto blockCount = args.pArgs.blockCount;

  const auto maxNumBlocksPerChunk =
      getMaxNumBlocksPerChunk<T>(args.config, args.pArgs);
  const auto remHeaderLen = getChunkHeaderLen(maxNumBlocksPerChunk);

  // Track number of copied blocks for each local rank per step
  // FIXME: reduce registers by templating numGroups
  int numFwdBlocks[CTRAN_MAX_NVL_PEERS];
  // Query chunkIdx of tmpRecvBuff on each receive rank when starting a new
  // chunk
  int remChunkIdx[CTRAN_MAX_NVL_PEERS];
  for (auto r = range.start; r <= range.end; r++) {
    numFwdBlocks[r] = 0;
  }

  auto jointWorkerId = workerId + groupId * numWorkers;
  auto* tmpIntraFwdIdx = reinterpret_cast<int*>(args.tmpIntraFwdIdx);
  for (auto i = threadIdx.x + blockDim.x * jointWorkerId;
       i < args.pArgs.totalNumSendBlocks * nNodes * nLocalRanks;
       i += blockDim.x * numWorkers * numGroups) {
    tmpIntraFwdIdx[i] = -1;
  }

  // sync across groups and workers to initialize tmpIntraFwdIdx
  WorkerSyncDev::sync(
      args.kSync.workerSync, 0, numGroups * numWorkers, WorkerSync::kFwd, 0);

  for (auto r = range.start; r <= range.end; r++) {
    for (auto j = threadIdx.x + blockDim.x * workerId;
         j < nNodes * args.pArgs.totalNumSendBlocks;
         j += blockDim.x * numWorkers) {
      int n = j / args.pArgs.totalNumSendBlocks;
      int b = j % args.pArgs.totalNumSendBlocks;
      int blockFwdIdx = fwdIdx
          [r * nNodes * args.pArgs.totalNumSendBlocks +
           n * args.pArgs.totalNumSendBlocks + b];
      if (blockFwdIdx != -1) {
        tmpIntraFwdIdx
            [r * nNodes * args.pArgs.totalNumSendBlocks +
             n * args.pArgs.totalNumSendBlocks + blockFwdIdx] = b;
      }
    }
  }

  // sync across groups and workers to make sure tmpIntraFwdIdx is ready
  WorkerSyncDev::sync(
      args.kSync.workerSync, 0, numGroups * numWorkers, WorkerSync::kFwd, 1);
  __threadfence();

  const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;
  const auto numChunks =
      (totalNumSendBlocks + maxNumBlocksPerChunk - 1) / maxNumBlocksPerChunk;
  for (auto bchunk = 0; bchunk < numChunks; bchunk++) {
    for (auto r = range.start; r <= range.end; r++) {
      __shared__ int numToCopy;
      if (threadIdx.x == 0) {
        numToCopy = 0;
      }
      __syncthreads();
      // get number of blocks to copy for this step
      for (auto i = threadIdx.x; i < maxNumBlocksPerChunk; i += blockDim.x) {
        auto b = bchunk * maxNumBlocksPerChunk + i;
        if (b >= totalNumSendBlocks) {
          break;
        }
        int blockFwdIdx = tmpIntraFwdIdx
            [r * nNodes * totalNumSendBlocks + myNode * totalNumSendBlocks + b];
        if (blockFwdIdx != -1) {
          atomicMax(&numToCopy, i + 1);
        }
      }
      __syncthreads();
      if (numToCopy == 0) {
        continue;
      }

      const auto remStep = FwdGroupSyncDev::getNextStep(
          args.kSync.fwdGroupSync, groupId, workerId, r);
      const auto chunkIdx = remChunkIdx[r] = getTmpChunkIdx(config, remStep);

      // Wait till the remote tmpRecvBuff is ready
      auto remSync =
          getFwdRecvSync(config, args.kSync, r, myLocalRank, chunkIdx);
      MPSCTbSyncDev::waitReady(remSync, workerId);

      for (auto i = workerId; i < numToCopy; i += numWorkers) {
        auto b = bchunk * maxNumBlocksPerChunk + i;
        int blockFwdIdx = tmpIntraFwdIdx
            [r * nNodes * totalNumSendBlocks + myNode * totalNumSendBlocks + b];

        const auto chunkIdx = remChunkIdx[r];
        // Update remote bitmap
        void* remTmpRecvBuff = getTmpChunkPtrByIdx(
            config, args.remTmpRecvBuffs[r], chunkIdx, myLocalRank);
        int* remBlockIds =
            getTmpChunkBlockIds(remTmpRecvBuff, maxNumBlocksPerChunk);
        if (threadIdx.x == 0) {
          int blockId = myRank * args.pArgs.totalNumSendBlocks + blockFwdIdx;
          remBlockIds[i] = blockId;
        }

        const T* sendBlock =
            reinterpret_cast<const T*>(args.execArgs.sendBuff) +
            blockFwdIdx * args.pArgs.blockCount;

        // Record dest ptr for the peer
        T* remRecvBlocks =
            reinterpret_cast<T*>(getTmpChunkData(remHeaderLen, remTmpRecvBuff));
        ctranKernCopy<T>(
            sendBlock, remRecvBlocks + i * blockCount, blockCount, 0, 1);
      }
      numFwdBlocks[r] = numToCopy;

      // If any recvRank's chunk is full, notify remote rank to consume the
      // chunk and move to next step
      if (numFwdBlocks[r] == maxNumBlocksPerChunk) {
        intraFwdSyncRemRecvCopy(
            args, workerId, r, numFwdBlocks[r], remChunkIdx[r], state);
        numFwdBlocks[r] = 0; // reset for next step
      }
      __syncthreads();
    }
  }

  // Sent all blocks, notify any recvLocalRanks if there are not-confirmed
  // blocks bcasted to them
  for (auto r = range.start; r <= range.end; r++) {
    if (numFwdBlocks[r] > 0) {
      intraFwdSyncRemRecvCopy(
          args, workerId, r, numFwdBlocks[r], remChunkIdx[r], state);
    }
  }
  // todo: do we need to sync across groups and workers after intrafwd?
  WorkerSyncDev::sync(
      args.kSync.workerSync, 0, numGroups * numWorkers, WorkerSync::kFwd, 2);
}

__device__ inline void computeContigFwdIdx(
    ExecKernArgs& args,
    int* contigFwdIdx,
    int* minIdx,
    int& maxval,
    int numBlocks,
    int* fwdBlockIds,
    const int* fwdIdx,
    int r,
    int* numFwdBlocks,
    int step,
    int sendNode,
    size_t maxNumBlocksPerChunk,
    int workerId,
    int numWorkers) {
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nNodes = statex->nNodes();

  for (auto i = threadIdx.x;
       i < ctran::alltoallvdedup::MAX_NUM_BLOCKS_PER_CHUNK;
       i += blockDim.x) {
    contigFwdIdx[i] = -1;
  }
  __syncthreads();
  for (auto b = threadIdx.x; b < numBlocks; b += blockDim.x) {
    int blockId = fwdBlockIds[b];
    const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;
    int blockRank = blockId / totalNumSendBlocks;
    int blockNode = blockRank / nLocalRanks;
    int railBlockId =
        blockId % totalNumSendBlocks + blockNode * totalNumSendBlocks;
    int blockFwdIdx = fwdIdx[r * nNodes * totalNumSendBlocks + railBlockId];
    if (blockFwdIdx != -1) {
      auto offset = blockFwdIdx - minIdx[r];
      contigFwdIdx[offset] = b;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    maxval = -1;
  }
  __syncthreads();
  for (auto i = threadIdx.x; i < numBlocks; i += blockDim.x) {
    auto b = contigFwdIdx[i];
    if (b != -1) {
      atomicMax(&maxval, i);
    }
  }
  __syncthreads();
  numFwdBlocks[r] = maxval + 1;
}

__device__ inline void initializeFwdSharedMem(
    ExecKernArgs& args,
    int* minIdx,
    const int* fwdIdx,
    GroupLocalRanks range,
    int numBlocks,
    int* fwdBlockIds) {
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nNodes = statex->nNodes();
  for (int i = 0; i < nLocalRanks; i++) {
    minIdx[i] = INT_MAX;
  }
  __syncthreads();
  for (auto r = range.start; r <= range.end; r++) {
    for (auto b = threadIdx.x; b < numBlocks; b += blockDim.x) {
      int blockId = fwdBlockIds[b];
      const auto totalNumSendBlocks = args.pArgs.totalNumSendBlocks;
      int blockRank = blockId / totalNumSendBlocks;
      int blockNode = blockRank / nLocalRanks;
      int railBlockId =
          blockId % totalNumSendBlocks + blockNode * totalNumSendBlocks;
      int blockFwdIdx = fwdIdx[r * nNodes * totalNumSendBlocks + railBlockId];
      if (blockFwdIdx != -1) {
        atomicMin(minIdx + r, blockFwdIdx);
        break;
      }
    }
  }
  __syncthreads();
}

template <typename T>
__device__ inline void progressFwd(
    ExecKernArgs& args,
    int groupId,
    int workerId,
    const GroupNodes& range) {
  const auto nLocalRanks = statex->nLocalRanks();
  // Each group handles different nodeIds for cross-node forwarding.
  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "groupId %d/%d workerId %d/%d handles nodes %d - %d, %d nodes\n",
        groupId,
        args.config.numFwdGroups,
        workerId,
        args.config.numFwdWorkers,
        range.start,
        range.end,
        range.numNodes);
  }

  ProgressFwdState state;
  updateProgressFwdState<T>(args, groupId, range, state);

  // For forwarding bcast, each group may bcast to any local ranks
  GroupLocalRanks bcastRange = {0, nLocalRanks - 1, nLocalRanks};

  while (state.numTotalPending > 0) {
    for (auto n = 0; n < range.numNodes; n++) {
      auto sendNode = n + range.start;
      // skip if has finished all steps for this sendNode
      if (state.numPendingBlocks[n] == 0) {
        continue;
      }

      // Wait host side to post tmpFwdBuff after RDMA received
      auto sync = args.kSync.recvFwdGKSyncs + sendNode;
      const auto step = state.steps[n];

      bool posted = false;
      posted = ctran::algos::GpeKernelSyncDev::checkPost(sync, workerId, step);
      if (!posted) {
        // If no available chunk is posted for this node, move to next
        continue;
      }

      // Obtain hdr, bitmaps, and data blocks arrays from the corresponding
      // chunk in tmpFwdBuff.
      // Chunk format: FwdChkHdr + bitmapArray[numBlocks] +
      // blockArray[numBlocks]
      const auto chunkIdx = getTmpChunkIdx(args.config, step);
      void* tmpBuff =
          getTmpChunkPtrByIdx(args.config, args.tmpFwdBuff, chunkIdx, sendNode);
      FwdChkHdr* hdr = reinterpret_cast<FwdChkHdr*>(tmpBuff);
      const auto numBlocks = hdr->numBlocks;
      const auto sendRank = hdr->sendRank;

      if (threadIdx.x == 0) {
        CTRAN_DEV_TRACE(
            "Received hdr %p numBlocks %d sendRank %d step %d chunkIdx %d\n",
            hdr,
            numBlocks,
            sendRank,
            step,
            chunkIdx);
      }

      const auto maxNumBlocksPerChunk =
          getMaxNumBlocksPerChunk<T>(args.config, args.pArgs);
      const auto headerLen = getChunkHeaderLen(maxNumBlocksPerChunk);
      int* fwdBlockIds = getTmpChunkBlockIds(tmpBuff, maxNumBlocksPerChunk);
      T* fwdBlocks = reinterpret_cast<T*>(getTmpChunkData(headerLen, tmpBuff));
      const auto numWorkers = args.config.numFwdWorkers;

      const auto nLocalRanks = statex->nLocalRanks();
      const auto myLocalRank = statex->localRank();
      const auto* fwdIdx = args.execArgs.fwdIdx;
      const auto& config = args.config;
      const auto blockCount = args.pArgs.blockCount;

      // Get common partition in receive ranks' tmpRecvBuff chunk
      const auto remHeaderLen = getChunkHeaderLen(maxNumBlocksPerChunk);

      // Track number of copied blocks for each local rank per step
      // FIXME: reduce registers by templating numGroups
      int numFwdBlocks[CTRAN_MAX_NVL_PEERS];
      // Query chunkIdx of tmpRecvBuff on each receive rank when starting a
      // new chunk
      int remChunkIdx[CTRAN_MAX_NVL_PEERS];
      for (int r = 0; r < nLocalRanks; r++) {
        numFwdBlocks[r] = 0;
      }

      __shared__ int
          contigFwdIdx[ctran::alltoallvdedup::MAX_NUM_BLOCKS_PER_CHUNK];
      __shared__ int minIdx[CTRAN_MAX_NVL_PEERS];
      __shared__ int maxval;
      initializeFwdSharedMem(
          args, minIdx, fwdIdx, bcastRange, numBlocks, fwdBlockIds);

      for (auto r = bcastRange.start; r <= bcastRange.end; r++) {
        __syncthreads();
        computeContigFwdIdx(
            args,
            contigFwdIdx,
            minIdx,
            maxval,
            numBlocks,
            fwdBlockIds,
            fwdIdx,
            r,
            numFwdBlocks,
            step,
            sendNode,
            maxNumBlocksPerChunk,
            workerId,
            numWorkers);

        if (numFwdBlocks[r] == 0) {
          continue;
        }

        const auto remStep = FwdGroupSyncDev::getNextStep(
            args.kSync.fwdGroupSync, groupId, workerId, r);
        const auto chunkIdx = remChunkIdx[r] = getTmpChunkIdx(config, remStep);

        // Wait till the remote tmpRecvBuff is ready
        auto remSync =
            getFwdRecvSync(config, args.kSync, r, myLocalRank, chunkIdx);
        MPSCTbSyncDev::waitReady(remSync, workerId);

        for (auto rb = threadIdx.x; rb < numFwdBlocks[r]; rb += blockDim.x) {
          auto b = contigFwdIdx[rb];
          int blockId = fwdBlockIds[b];

          const auto chunkIdx = remChunkIdx[r];
          // Update remote bitmap
          void* remTmpRecvBuff = getTmpChunkPtrByIdx(
              config, args.remTmpRecvBuffs[r], chunkIdx, myLocalRank);
          int* remBlockIds =
              getTmpChunkBlockIds(remTmpRecvBuff, maxNumBlocksPerChunk);
          remBlockIds[rb] = blockId;
        }

        for (auto rb = workerId; rb < numFwdBlocks[r]; rb += numWorkers) {
          auto b = contigFwdIdx[rb];
          const T* sendBlock = fwdBlocks + b * args.pArgs.blockCount;
          const auto chunkIdx = remChunkIdx[r];
          // Update remote bitmap
          void* remTmpRecvBuff = getTmpChunkPtrByIdx(
              config, args.remTmpRecvBuffs[r], chunkIdx, myLocalRank);

          // Record dest ptr for the peer
          T* remRecvBlocks = reinterpret_cast<T*>(
              getTmpChunkData(remHeaderLen, remTmpRecvBuff));
          ctranKernCopy<T>(
              sendBlock, remRecvBlocks + rb * blockCount, blockCount, 0, 1);
        }
      }

      // Update total number of blocks copied by each worker and notify recv
      // rank to consume
      for (auto r = 0; r < nLocalRanks; r++) {
        if (numFwdBlocks[r] > 0) {
          state.numFwdBlocksToRank[r] += numFwdBlocks[r];
          fwdSyncRemRecvCopy(
              args,
              workerId,
              step,
              r,
              sendRank,
              numFwdBlocks[r],
              remChunkIdx[r],
              state);
        }
      }

      // Notify GPE thread that we have finished forwarding of this chunk
      ctran::algos::GpeKernelSyncDev::complete(sync, workerId, step);
      state.updateStep(n, numBlocks);
    }
  }
}

template <typename T>
__device__ inline void updateProgressRecvState(
    ExecKernArgs& args,
    int groupId,
    const GroupLocalRanks& range,
    ProgressRecvState& state) {
  const auto nNodes = statex->nNodes();

  state.numTotalPending = 0;
  const auto* numRecvBlocks = args.execArgs.numRecvBlocks;
  for (auto r = range.start; r <= range.end; r++) {
    state.numPendingBlocks[r] = 0;
    state.steps[r] = 0;

    // Count steps based on recvCount from each sendRank via the same fwdRank.
    for (auto n = 0; n < nNodes; n++) {
      auto sendRank = statex->localRankToRank(r, n);
      const auto offset = sendRank * args.pArgs.numRecvBuckets;
      for (auto i = 0; i < args.pArgs.numRecvBuckets; i++) {
        state.numPendingBlocks[r] += numRecvBlocks[offset + i];
      }
    }
    if (state.numPendingBlocks[r] > 0) {
      state.numTotalPending++;
    }
  }

  for (auto r = range.start; r <= range.end; r++) {
    if (threadIdx.x == 0) {
      CTRAN_DEV_TRACE(
          "groupId %d loaded numPendingBlocks %ld from localFwdRank %d, numTotalPending %d\n",
          groupId,
          state.numPendingBlocks[r],
          r,
          state.numTotalPending);
    }
  }
}

__device__ inline void computeContigRecvIdx(
    ExecKernArgs& args,
    int* contigRecvIdx,
    const int* recvIdx,
    int* blockIds,
    int* minIdx,
    int* numValid,
    int* effectiveTotal,
    int numBlocks,
    int nRanks,
    int recvBucket) {
  for (auto i = threadIdx.x;
       i < ctran::alltoallvdedup::MAX_NUM_BLOCKS_PER_CHUNK;
       i += blockDim.x) {
    contigRecvIdx[i] = -1;
  }
  __syncthreads();
  for (auto b = threadIdx.x; b < numBlocks; b += blockDim.x) {
    const auto curRecvIdx = recvIdx
        [recvBucket * args.pArgs.totalNumSendBlocks * nRanks + blockIds[b]];
    if (curRecvIdx != -1) {
      auto offset = curRecvIdx - minIdx[recvBucket];
      contigRecvIdx[offset] = b;
      atomicAdd(numValid + recvBucket, 1);
      atomicAdd(effectiveTotal, 1);
    }
  }
  __syncthreads();
}

__device__ inline void initializeRecvCopySharedMem(
    ExecKernArgs& args,
    int* minIdx,
    int* numValid,
    int* effectiveTotal,
    const int* recvIdx,
    int* blockIds,
    int numBlocks,
    int nRanks) {
  for (auto i = threadIdx.x; i < args.pArgs.numRecvBuckets; i += blockDim.x) {
    minIdx[i] = INT_MAX;
    numValid[i] = 0;
  }
  if (threadIdx.x == 0) {
    *effectiveTotal = 0;
  }
  __syncthreads();
  for (int i = 0; i < args.pArgs.numRecvBuckets; i++) {
    for (auto b = threadIdx.x; b < numBlocks; b += blockDim.x) {
      const auto curRecvIdx =
          recvIdx[i * args.pArgs.totalNumSendBlocks * nRanks + blockIds[b]];
      if (curRecvIdx != -1) {
        atomicMin(minIdx + i, curRecvIdx);
        break;
      }
    }
  }
  __syncthreads();
}

template <typename T>
__device__ inline void recvCopyBlocks(
    ExecKernArgs& args,
    int groupId,
    int workerId,
    int fwdLocalRank,
    const GroupLocalRanks& range,
    ProgressRecvState& state,
    int overallStep) {
  const auto step = state.steps[fwdLocalRank];
  const auto chunkIdx = getTmpChunkIdx(args.config, step);
  const auto myLocalRank = statex->localRank();

  const auto nRanks = statex->nRanks();
  const auto* recvIdx = args.execArgs.recvIdx;

  auto sync = getFwdRecvSync(
      args.config, args.kSync, myLocalRank, fwdLocalRank, chunkIdx);
  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "Waiting on FwdRecvSync %p (offset %ld) fwdLocalRank %d step %d chunkIdx %d\n",
        sync,
        getOffset(sync, args.kSync.fwdRecvSyncs),
        fwdLocalRank,
        step,
        chunkIdx);
  }

  // Wait for all fwd workers to finish
  if (workerId == 0) {
    MPSCTbSyncDev::waitPost(sync);
  }

  // Only worker 0 waits on fwd workers, all other recv workers wait on worker 0
  // here. overallStep is the global step across fwdLocalRanks that are operated
  // on. We sync on overallstep*2 because this is the first of 2 syncs in
  // recvCopyBlocks.
  WorkerSyncDev::sync(
      args.kSync.workerSync,
      groupId,
      args.config.numRecvWorkers,
      WorkerSync::kRecv,
      overallStep * 2);
  __threadfence();

  auto gksync = args.kSync.recvCopyGKSyncs + fwdLocalRank;
  ctran::algos::GpeKernelSyncDev::complete(gksync, workerId, 2 * step);

  // Copy from tmpRecvBuff to user recvBuff
  void* tmpRecvBuff = getTmpChunkPtrByIdx(
      args.config, args.tmpRecvBuff, chunkIdx, fwdLocalRank);
  FwdChkHdr* hdr = getTmpChunkHdr(tmpRecvBuff);
  const auto numBlocks = hdr->numBlocks;
  const auto sendRank = hdr->sendRank;
  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "Received hdr %p numBlocks %d sendRank %d from fwdLocalRank %d step %d chunkIdx %d\n",
        hdr,
        numBlocks,
        sendRank,
        fwdLocalRank,
        step,
        chunkIdx);
  }

  const auto maxNumBlocksPerChunk =
      getMaxNumBlocksPerChunk<T>(args.config, args.pArgs);
  const auto headerLen = getChunkHeaderLen(maxNumBlocksPerChunk);
  T* tmpBlocks = reinterpret_cast<T*>(getTmpChunkData(headerLen, tmpRecvBuff));
  int* blockIds = getTmpChunkBlockIds(tmpRecvBuff, maxNumBlocksPerChunk);
  const auto sendRankOffset = sendRank * args.pArgs.numRecvBuckets;

  const auto blockCount = args.pArgs.blockCount;
  T* recvBuff = reinterpret_cast<T*>(args.execArgs.recvBuff);

  // recvCopyBlocks will get the tokens and blocksIds from the tmpbuff.
  // BlockIds tells how which bucket to copy the block to by accessing recvIdx.
  // e.g.
  // blockids 16324 16325 16326 16332 ...
  // recvIdx i=0 2032 2033 -1 2034 ...
  // recvIdx i=1 -1 2032 2033 -1 ...
  // contigRecvIdx is computed for each bucket to get all tokens that will be
  // copied. minIdx is the index of the first token != -1 in each bucket used so
  // different workers can compute contigRecvIdx.
  // minIdx i=0: 0
  // minIdx i=1: 1
  // contigRecvIdx i=0 0 1 3 ...
  // contigRecvIdx i=1 1 2 ...
  // This way, we iterate on contigRecvIdx where the index is the offset of the
  // destination buffer and the value tells us which block to copy.
  __shared__ int contigRecvIdx[ctran::alltoallvdedup::MAX_NUM_BLOCKS_PER_CHUNK];
  __shared__ int minIdx[ctran::alltoallvdedup::MAX_NUM_RECV_BUCKETS];
  __shared__ int numValid[ctran::alltoallvdedup::MAX_NUM_RECV_BUCKETS];
  __shared__ int effectiveTotal;
  // Note: it may be cleaner to compute minIdx and numValid on fwd rank
  initializeRecvCopySharedMem(
      args,
      minIdx,
      numValid,
      &effectiveTotal,
      recvIdx,
      blockIds,
      numBlocks,
      nRanks);

  for (int i = 0; i < args.pArgs.numRecvBuckets; i++) {
    computeContigRecvIdx(
        args,
        contigRecvIdx,
        recvIdx,
        blockIds,
        minIdx,
        numValid,
        &effectiveTotal,
        numBlocks,
        nRanks,
        i);

    // All threads in each worker copy a single block, different workers copy
    // different blocks
    for (auto recvB = workerId; recvB < numValid[i];
         recvB += args.config.numRecvWorkers) {
      auto b = contigRecvIdx[recvB];
      if (b == -1) {
        break;
      }

      auto recvOffset = args.tmpRecvOffsets[sendRankOffset + i];
      const T* sData = tmpBlocks + b * blockCount;
      T* dData = recvBuff + recvOffset + recvB * blockCount;

      // FIXME: do I need check 16byte alignement here?
      ctranKernCopy<T>(sData, dData, blockCount, 0, 1);
    }
    __syncthreads();
  }

  ctran::algos::GpeKernelSyncDev::complete(gksync, workerId, 2 * step + 1);

  // Only worker 0 notifies fwd workers of complete. All workers wait here for
  // everyone to be done before worker 0 notifies. overallStep is the global
  // step across fwdLocalRanks that are operated on. We sync on overallstep*2+1
  // because this is the second of 2 syncs in recvCopyBlocks.
  WorkerSyncDev::sync(
      args.kSync.workerSync,
      groupId,
      args.config.numRecvWorkers,
      WorkerSync::kRecv,
      2 * overallStep + 1);
  __threadfence();

  // Consumed; notify fwd workers that the chunk is ready to reuse
  if (workerId == 0) {
    MPSCTbSyncDev::complete(sync);
  }

  // FIXME: next >PSCTbSyncDev::waitPost should ensure all threads in the same
  // block see the updated offset; does it apply to across workers?
  if (workerId == 0 && threadIdx.x == 0) {
    for (int i = 0; i < args.pArgs.numRecvBuckets; i++) {
      args.tmpRecvOffsets[sendRank * args.pArgs.numRecvBuckets + i] +=
          numValid[i] * blockCount;
    }
  }

  state.updateStep(fwdLocalRank, effectiveTotal);
}

template <typename T>
__device__ inline void progressRecvCopy(
    ExecKernArgs& args,
    int groupId,
    int workerId,
    const GroupLocalRanks& range) {
  // Each group handles different different localRanks for intra-node direct
  // forwarding.
  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "groupId %d workerId %d handles localRanks %d - %d, %d localRanks\n",
        groupId,
        workerId,
        range.start,
        range.end,
        range.numRanks);
  }

  // Compute how many blocks to be forwarded from each local FWD rank
  ProgressRecvState state;
  updateProgressRecvState<T>(args, groupId, range, state);

  // overallStep is the global step across r with range to help sync workers
  int overallStep = 0;
  while (state.numTotalPending > 0) {
    for (auto r = range.start; r <= range.end; r++) {
      // already finished
      if (state.numPendingBlocks[r] == 0) {
        continue;
      }
      recvCopyBlocks<T>(args, groupId, workerId, r, range, state, overallStep);
      overallStep++;
    }
  }
}

#define ERRORRETURN(flag)                                     \
  do {                                                        \
    const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (flag && gtIdx == 0) {                                 \
      ctran::device::KernelWaitGpeTerminate(flag);            \
    }                                                         \
    return;                                                   \
  } while (0)

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
  auto& config = args.config;
  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "ncclKernelAllToAllvDedup called, numSendGroups=%d, %d, numFwdGroups=%d, %d, numRecvGroups=%d, %d\n",
        config.numSendGroups,
        config.numSendWorkers,
        config.numFwdGroups,
        config.numFwdWorkers,
        config.numRecvGroups,
        config.numRecvWorkers);
  }

  auto totalNumSendWorkers = config.numSendGroups * config.numSendWorkers;
  auto totalNumFwdWorkers = config.numFwdGroups * config.numFwdWorkers;
  auto totalNumRecvWorkers = config.numRecvGroups * config.numRecvWorkers;

  if (gridDim.x <
      totalNumSendWorkers + totalNumFwdWorkers + totalNumRecvWorkers) {
    if (threadIdx.x == 0) {
      CTRAN_DEV_TRACE(
          "ERROR: Something wrong, total blocks %d < %d + %d + %d\n",
          gridDim.x,
          totalNumSendWorkers,
          totalNumFwdWorkers,
          totalNumRecvWorkers);
    }
    ERRORRETURN(flag);
  }

  // sendCopy groups
  if (blockIdx.x < totalNumSendWorkers) {
    const auto groupId = blockIdx.x / config.numSendWorkers;
    const auto workerId = blockIdx.x % config.numSendWorkers;
    GroupNodes range = assignGroupNodes(args, groupId, config.numSendGroups);
    progressSendCopy<T>(args, groupId, workerId, range);
  }
  // recvFwd groups
  else if (blockIdx.x < totalNumSendWorkers + totalNumFwdWorkers) {
    const auto groupId =
        (blockIdx.x - totalNumSendWorkers) / config.numFwdWorkers;
    const auto workerId =
        (blockIdx.x - totalNumSendWorkers) % config.numFwdWorkers;
    GroupLocalRanks intraRange =
        assignGroupLocalRanks(args, groupId, config.numFwdGroups);

    auto myNode = statex->node();
    auto sync = args.kSync.recvFwdGKSyncs + myNode;
    ctran::algos::GpeKernelSyncDev::waitPost(sync, workerId, 0);

    // Handle intra-node forwarding first, since the recvRanks should still be
    // idle and cross-node forwarding chunks are not yet arrived
    progressIntraFwd<T>(args, groupId, workerId, intraRange);

    ctran::algos::GpeKernelSyncDev::complete(sync, workerId, 1);

    GroupNodes nodeRange = assignGroupNodes(args, groupId, config.numFwdGroups);
    progressFwd<T>(args, groupId, workerId, nodeRange);
  } else if (
      blockIdx.x <
      totalNumSendWorkers + totalNumFwdWorkers + totalNumRecvWorkers) {
    const auto groupId =
        (blockIdx.x - totalNumSendWorkers - totalNumFwdWorkers) /
        config.numRecvWorkers;
    const auto workerId =
        (blockIdx.x - totalNumSendWorkers - totalNumFwdWorkers) %
        config.numRecvWorkers;

    GroupLocalRanks range =
        assignGroupLocalRanks(args, groupId, config.numRecvGroups);
    progressRecvCopy<T>(args, groupId, workerId, range);

    __syncthreads();
    for (int i = 0; i < statex->nLocalRanks(); i++) {
      auto sync = args.kSync.recvCopyGKSyncs + i;
      GpeKernelSyncDev::reset(sync, workerId);
    }
  }
  // FIXME: handle intra-node forward: after exchangeMetadata, each FWD rank
  // knows the recvOffset of blocks from itself, thus can copy to recvBuff.
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
