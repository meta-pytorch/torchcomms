// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include "comms/ctran/algos/AllToAllvDedup/CommonDev.h"
#include "comms/ctran/algos/AllToAllvDedup/FwdRecvSync.cuh"
#include "comms/ctran/algos/AllToAllvDedup/WorkerGroupDev.cuh"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"

#define VERBOSE 0
constexpr int kNumBlocksToTrace = 5;
#define DATATRACE_COND(b) (b < kNumBlocksToTrace && VERBOSE)

namespace ctran::alltoallvdedup {
using namespace ctran::algos;

__device__ __forceinline__ WorkerGroupSync* getWgSync(
    const ExecKernArgs& args,
    const WorkerGroupType type,
    const int groupId = 0) {
  const auto& kSync = args.kSync;
  const auto t = static_cast<int>(type);
  return kSync.wgSyncs + t * MAX_NUM_GROUPS_PER_ROLE + groupId;
}

// Single worker (thread block) in exec() recvCopy records the stepInfo for
// combine() to use
__device__ inline void recordRecvCopyStep(
    const ExecKernArgs& args,
    const int step,
    const int fwdLocalRank,
    const RecvCopyStepInfo& stepInfo,
    const int* blockIds,
    const bool kIsIntra) {
  const auto maxNumStepBlks = args.pArgs.maxNumStepBlks;
  const auto nLocalRanks = statex->nLocalRanks();
  const auto slotId = step * nLocalRanks + fwdLocalRank;

  int* blockIdsBase =
      kIsIntra ? args.intraRecvStepFwdBlockIds : args.recvStepFwdBlockIds;
  RecvCopyStepInfo* info =
      kIsIntra ? args.intraRecvStepInfo + slotId : args.recvStepInfo + slotId;
  int* stepBlockIds = blockIdsBase + slotId * maxNumStepBlks;
  for (auto b = threadIdx.x; b < stepInfo.numBlocks; b += blockDim.x) {
    stepBlockIds[b] = blockIds[b];
  }
  if (threadIdx.x == 0) {
    *info = stepInfo;
  }
}

__device__ inline void loadRecvCopyStep(
    const ExecKernArgs& args,
    const int step,
    const int fwdLocalRank,
    RecvCopyStepInfo& stepInfo,
    int*& stepBlockIds,
    const bool kIsIntra) {
  const auto maxNumStepBlks = args.pArgs.maxNumStepBlks;
  const auto nLocalRanks = statex->nLocalRanks();
  const auto slotId = step * nLocalRanks + fwdLocalRank;

  int* blockIdsBase =
      kIsIntra ? args.intraRecvStepFwdBlockIds : args.recvStepFwdBlockIds;
  RecvCopyStepInfo* info =
      kIsIntra ? args.intraRecvStepInfo + slotId : args.recvStepInfo + slotId;

  stepInfo = *info;
  stepBlockIds = blockIdsBase + slotId * maxNumStepBlks;

  CTRAN_DEV_TRACE_IF(
      threadIdx.x == 0,
      "step %d peerLocalRank %d: sendRank %d numBlocks %d\n",
      step,
      fwdLocalRank,
      stepInfo.sendRank,
      stepInfo.numBlocks);
}

struct FwdStepInfo {
  int numBlocks;
  int* blockIds;
};

// int numRecvrBlocks[CTRAN_MAX_NVL_PEERS];
__device__ inline void recordFwdStep(
    const ExecKernArgs& args,
    const int step,
    const int sendNode,
    const FwdStepInfo& stepInfo) {
  const auto nNodes = statex->nNodes();
  const auto slotId = step * nNodes + sendNode;

  if (threadIdx.x == 0) {
    args.fwdStepNumBlocks[slotId] = stepInfo.numBlocks;
  }

  int* stepBlockIds = args.fwdStepBlockIds + slotId * args.pArgs.maxNumStepBlks;
  for (auto b = threadIdx.x; b < stepInfo.numBlocks; b += blockDim.x) {
    stepBlockIds[b] = stepInfo.blockIds[b];
  }
}

__device__ inline void loadFwdStep(
    const ExecKernArgs& args,
    const int step,
    const int sendNode,
    FwdStepInfo& stepInfo) {
  const auto nNodes = statex->nNodes();
  const auto slotId = step * nNodes + sendNode;

  stepInfo.numBlocks = args.fwdStepNumBlocks[slotId];
  stepInfo.blockIds = args.fwdStepBlockIds + slotId * args.pArgs.maxNumStepBlks;

  CTRAN_DEV_TRACE_IF(
      threadIdx.x == 0,
      "step %d sendNode %d: numBlocks %d\n",
      step,
      sendNode,
      stepInfo.numBlocks);
}

__device__ inline void recordFwdStepRecvrNumBlocks(
    const ExecKernArgs& args,
    const int step,
    const int localRank,
    const int val) {
  const auto nLocalRanks = statex->nLocalRanks();
  const auto offset = step * nLocalRanks + localRank;
  args.fwdStepRecvrNumBlocks[offset] = val;
}

// a single worker (thread block) to load the numBlocks for all local ranks into
// shared mem
__device__ inline void loadFwdStepRecvrNumBlocks(
    const ExecKernArgs& args,
    const int step,
    int* shmNumRecvrBlocks) {
  const auto nLocalRanks = statex->nLocalRanks();
  const auto offset = step * nLocalRanks;
  for (auto r = threadIdx.x; r < nLocalRanks; r += blockDim.x) {
    shmNumRecvrBlocks[r] = args.fwdStepRecvrNumBlocks[offset + r];
  }
  __syncthreads();
}

// return the number of blocks for a step
__device__ __forceinline__ int
getStepNumBlocks(const ExecKernArgs& args, const int total, const int step) {
  const auto stepCount = args.pArgs.maxNumStepBlks;
  const auto pending = total - step * stepCount;
  return pending > stepCount ? stepCount : pending;
}

// return the number of blocks that have been totally posted till this step
__device__ __forceinline__ int
getStepNumPosted(const ExecKernArgs& args, const int total, const int step) {
  const auto stepCount = args.pArgs.maxNumStepBlks;
  const auto curr = step * stepCount;
  return curr <= total ? curr : total;
}

#define CHECK_VALID_STATE(idx, totalPending, numPendingBlocks)               \
  do {                                                                       \
    if (threadIdx.x == 0 && totalPending < 0 || numPendingBlocks[idx] < 0) { \
      CTRAN_DEV_FATAL(                                                       \
          "%s %d < 0 or %s[%d] %d < 0\n",                                    \
          #totalPending,                                                     \
          totalPending,                                                      \
          #numPendingBlocks,                                                 \
          idx,                                                               \
          numPendingBlocks[idx]);                                            \
    }                                                                        \
  } while (0)

#define LOG_STATE_UPDATE_STEP(                                                                              \
    CLASS_NAME, stepLabel, stepIdx, stepVal, pendingLabel, pendingIdx)                                      \
  do {                                                                                                      \
    CTRAN_DEV_TRACE(                                                                                        \
        CLASS_NAME                                                                                          \
        ": finished steps[%s %d] %d numBlocks %d, updated numPendingBlocks[%s %d] %d numTotalPending %d\n", \
        stepLabel,                                                                                          \
        stepIdx,                                                                                            \
        stepVal,                                                                                            \
        numBlocks,                                                                                          \
        pendingLabel,                                                                                       \
        pendingIdx,                                                                                         \
        numPendingBlocks[pendingIdx],                                                                       \
        numTotalPending);                                                                                   \
  } while (0)

struct GroupLocalRanks {
  int start;
  int end;
  int numRanks;
  __device__ __forceinline__ bool contains(const int r) const {
    return r >= start && r <= end;
  }
};

struct GroupNodes {
  int start;
  int end;
  int numNodes;
};

struct ProgressSendState {
  int numPendingBlocks[MAX_NUM_NODES];
  int steps[MAX_NUM_NODES];
  int numTotalPending;

  __device__ __forceinline__ void updateStep(
      const int idx,
      const int numBlocks) {
    steps[idx]++;
    numPendingBlocks[idx] -= numBlocks;
    if (numPendingBlocks[idx] == 0) {
      numTotalPending--;
    }
    if (threadIdx.x == 0) {
      // LOG_STATE_UPDATE_STEP("progressSend", "node", idx, old, "node", idx);
      // TODO: add control to skip out-of-range check
      CHECK_VALID_STATE(idx, numTotalPending, numPendingBlocks);
    }
  }

  __device__ __forceinline__ bool hasPending(const int idx) {
    return numPendingBlocks[idx] > 0;
  }

  __device__ __forceinline__ void setupState(
      ExecKernArgs& args,
      const GroupNodes& range,
      const bool kIsExec = true) {
    const auto myNode = statex->node();

    numTotalPending = 0;
    for (auto n = 0; n < range.numNodes; n++) {
      numPendingBlocks[n] = 0; // numSendBlocks for each node
      steps[n] = 0;

      auto recvNode = n + range.start;
      // local node send is handled by intraFwd in exec()
      if (kIsExec && recvNode == myNode) {
        continue;
      }

      // tmpNumSendIdx populated at start of exec
      const auto numSendIdx = args.tmpNumSendIdx[recvNode];
      numPendingBlocks[n] = numSendIdx;

      if (numSendIdx > 0) {
        numTotalPending++;
        if (threadIdx.x == 0) {
          int recvNode = n + range.start;
          CTRAN_DEV_TRACE(
              "loaded numPendingBlocks %d to recvNode %d, numTotalPending %d\n",
              numPendingBlocks[n],
              recvNode,
              numTotalPending);
        }
      }
    }
  }
};

struct ProgressFwdState {
  // number of pending blocks to be forwarded for other nodes
  int numPendingBlocks[MAX_NUM_NODES];
  int steps[MAX_NUM_NODES];
  // help control only log first check for a given step
  int frCheckSteps[CTRAN_MAX_NVL_PEERS];
  int postedStep[CTRAN_MAX_NVL_PEERS];
  int fwdReady[CTRAN_MAX_NVL_PEERS];
  int fwdDone[CTRAN_MAX_NVL_PEERS];
  // track numBlocks forwarded to each local rank for debugging purpose, can be
  // disabled to save registers
  int numFwdBlocksToRank[CTRAN_MAX_NVL_PEERS];
  int numTotalPending;

  // Update if a new chunk received from GPE and ready to be forwarded to the
  // peerLocalRank
  __device__ __forceinline__ void updatePostStep(const int r) {
    fwdReady[r]++;
  }

  // Update when finished a forwarding to the peerLocalRank
  __device__ __forceinline__ void updateFwdStep(const int r) {
    fwdDone[r]++;
  }

  __device__ __forceinline__ void updateStep(
      const int idx,
      const int numBlocks) {
    const auto old = steps[idx];
    steps[idx]++;
    numPendingBlocks[idx] -= numBlocks;
    if (numPendingBlocks[idx] == 0) {
      numTotalPending--;
    }
    if (threadIdx.x == 0) {
      LOG_STATE_UPDATE_STEP("progressFwd", "node", idx, old, "node", idx);
      // TODO: add control to skip out-of-range check
      CHECK_VALID_STATE(idx, numTotalPending, numPendingBlocks);
    }
  }

  __device__ __forceinline__ void setupState(
      ExecKernArgs& args,
      const GroupNodes& range) {
    const auto nLocalRanks = statex->nLocalRanks();
    const auto myNode = statex->node();
    const auto* tmpNumFwdIdx = args.tmpNumFwdIdx;

    for (auto r = 0; r < nLocalRanks; r++) {
      numFwdBlocksToRank[r] = 0;
      fwdReady[r] = 0;
      fwdDone[r] = 0;
      frCheckSteps[r] = -1;
    }

    numTotalPending = 0;
    for (auto n = 0; n < range.numNodes; n++) {
      numPendingBlocks[n] = 0;
      steps[n] = 0;
      const auto sendNode = n + range.start;
      // Intra-node forwarding is handled separately
      if (sendNode == myNode) {
        continue;
      }

      const auto count = tmpNumFwdIdx[sendNode];
      numPendingBlocks[n] = count;
      if (count > 0) {
        numTotalPending++;
      }
      if (threadIdx.x == 0) {
        CTRAN_DEV_TRACE(
            "loaded tmpNumFwdIdx[node %d] %d -> numPendingBlocks %d, numTotalPending %d\n",
            sendNode,
            tmpNumFwdIdx[sendNode],
            numPendingBlocks[n],
            numTotalPending);
      }
    }
  }
};

struct ProgressRecvState {
  // number of pending blocks to be received from local forwarding ranks
  // Use signed int to check potential out-of-range numBlocks in the pipeline
  int numPendingBlocks[CTRAN_MAX_NVL_PEERS];
  int steps[CTRAN_MAX_NVL_PEERS];
  // help control only log first check for a given step
  int frCheckSteps[CTRAN_MAX_NVL_PEERS];
  int numTotalPending;

  __device__ __forceinline__ void updateStep(
      const int idx,
      const int numBlocks) {
    const auto old = steps[idx];
    steps[idx]++;
    numPendingBlocks[idx] -= numBlocks;
    if (numPendingBlocks[idx] == 0) {
      numTotalPending--;
    }
    if (threadIdx.x == 0) {
      LOG_STATE_UPDATE_STEP(
          "progressRecv", "localRank", idx, old, "localRank", idx);

      // TODO: add control to skip out-of-range check
      CHECK_VALID_STATE(idx, numTotalPending, numPendingBlocks);
    }
  }

  __device__ __forceinline__ void setupIntraRecvState(ExecKernArgs& args) {
    // nLocalRanks
    int* tmpNumIntraRecvIdx = args.tmpNumIntraRecvIdx;
    const auto nLocalRanks = statex->nLocalRanks();

    // Popolate local state
    numTotalPending = 0;

    // Count number of received blocks from all sendRanks on the same node
    for (auto r = 0; r < nLocalRanks; r++) {
      const int count = tmpNumIntraRecvIdx[r];
      numPendingBlocks[r] = count;
      frCheckSteps[r] = -1;
      steps[r] = 0;
      if (count > 0) {
        numTotalPending++;
      }
      if (threadIdx.x == 0) {
        CTRAN_DEV_TRACE(
            "loaded numPendingBlocks %d from sendRank %d, numTotalPending %d\n",
            numPendingBlocks[r],
            r,
            numTotalPending);
      }
    }
  }

  __device__ __forceinline__ void setupState(
      ExecKernArgs& args,
      const GroupLocalRanks& range) {
    // nLocalRanks
    int* tmpNumFwdRecvIdx = args.tmpNumFwdRecvIdx;

    // Popolate local state
    numTotalPending = 0;

    // Count number of received blocks from all sendRanks in the range
    for (auto r = range.start; r <= range.end; r++) {
      const int count = tmpNumFwdRecvIdx[r];
      numPendingBlocks[r] = count;
      steps[r] = 0;
      frCheckSteps[r] = -1;
      if (count > 0) {
        numTotalPending++;
      }
      CTRAN_DEV_TRACE_IF(
          threadIdx.x == 0,
          "loaded numPendingBlocks %d from peerLocalRank %d, numTotalPending %d\n",
          numPendingBlocks[r],
          r,
          numTotalPending);
    }
  }
};

struct ProgressIntraFwdState {
  // number of pending blocks to be forwarded to local receiving ranks
  int numPendingBlocks[CTRAN_MAX_NVL_PEERS];
  int steps[CTRAN_MAX_NVL_PEERS];
  // help control only log first check for a given step
  int frCheckSteps[CTRAN_MAX_NVL_PEERS];
  int numTotalPending;

  __device__ __forceinline__ void updateStep(int idx, const int numBlocks) {
    steps[idx]++;
    numPendingBlocks[idx] -= numBlocks;
    if (numPendingBlocks[idx] == 0) {
      numTotalPending--;
    }
    if (threadIdx.x == 0) {
      // LOG_STATE_UPDATE_STEP(
      //     "progressIntraFwd", "localRank", idx, old, "localRank", idx);

      // TODO: add control to skip out-of-range check
      CHECK_VALID_STATE(idx, numTotalPending, numPendingBlocks);
    }
  }

  __device__ __forceinline__ bool hasPending(const int idx) {
    return numPendingBlocks[idx] > 0;
  }

  __device__ __forceinline__ void setupState(
      ExecKernArgs& args,
      const GroupLocalRanks& range) {
    const auto* tmpNumIntraFwdIdx = args.tmpNumIntraFwdIdx;

    numTotalPending = 0;
    for (auto r = range.start; r <= range.end; r++) {
      const auto localRank = r + range.start;
      const auto count = tmpNumIntraFwdIdx[localRank];
      steps[r] = 0;
      numPendingBlocks[r] = count;

      if (count > 0) {
        numTotalPending++;
      }
      if (threadIdx.x == 0) {
        CTRAN_DEV_TRACE(
            "loaded tmpNumIntraFwdIdx[localRank %d] -> numPendingBlocks %d, numTotalPending %d\n",
            localRank,
            numPendingBlocks[r],
            numTotalPending);
      }
    }
  }
};

__device__ inline void assignWorkerGroups(
    const ExecKernArgs& args,
    WorkerGroup& sendG,
    WorkerGroup& fwdG,
    WorkerGroup& recvG,
    WorkerGroup& intraFwdG,
    WorkerGroup& intraRecvG,
    bool check = true) {
  const auto& config = args.config;
  const auto totalNumSendWorkers = config.numSendGroups * config.numSendWorkers;
  const auto numFwdWorkers = config.numFwdWorkers;
  const auto totalNumRecvWorkers = config.numRecvGroups * config.numRecvWorkers;
  const auto numIntraFwdWorkers = config.numIntraFwdWorkers;
  const auto numIntraRecvWorkers = config.numIntraRecvWorkers;

  if (check &&
      gridDim.x < totalNumSendWorkers + numFwdWorkers + totalNumRecvWorkers +
              numIntraFwdWorkers + numIntraRecvWorkers) {
    if (threadIdx.x == 0) {
      CTRAN_DEV_TRACE(
          "ERROR: Something wrong, total blocks %d < %d + %d + %d + %d + %d\n",
          gridDim.x,
          totalNumSendWorkers,
          numFwdWorkers,
          totalNumRecvWorkers,
          numIntraFwdWorkers,
          numIntraRecvWorkers);
    }
    trap();
  }

  assignWorkerGroup(0, config.numSendGroups, config.numSendWorkers, sendG);
  sendG.init(getWgSync(args, WorkerGroupType::kSend, sendG.groupId()));

  assignWorkerGroup(sendG.end + 1, 1, config.numFwdWorkers, fwdG);
  fwdG.init(getWgSync(args, WorkerGroupType::kFwd, fwdG.groupId()));

  assignWorkerGroup(
      fwdG.end + 1, config.numRecvGroups, config.numRecvWorkers, recvG);
  recvG.init(getWgSync(args, WorkerGroupType::kRecv, recvG.groupId()));

  assignWorkerGroup(recvG.end + 1, 1, config.numIntraFwdWorkers, intraFwdG);
  intraFwdG.init(
      getWgSync(args, WorkerGroupType::kIntraFwd, intraFwdG.groupId()));

  assignWorkerGroup(
      intraFwdG.end + 1, 1, config.numIntraRecvWorkers, intraRecvG);
  intraRecvG.init(
      getWgSync(args, WorkerGroupType::kIntraRecv, intraRecvG.groupId()));
}

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
    const WorkerGroup& group) {
  GroupLocalRanks range = {-1, -1, -1};
  const auto nLocalRanks = statex->nLocalRanks();
  const auto groupId = group.groupId();
  const auto numGroups = group.numGroups;
  getGroupRange(
      nLocalRanks, numGroups, groupId, range.start, range.end, range.numRanks);
  return range;
}

__device__ inline GroupNodes assignGroupNodes(
    ExecKernArgs& args,
    const WorkerGroup& group) {
  GroupNodes range = {-1, -1, -1};
  const auto nNodes = statex->nNodes();
  const auto groupId = group.groupId();
  const auto numGroups = group.numGroups;
  getGroupRange(
      nNodes, numGroups, groupId, range.start, range.end, range.numNodes);
  return range;
}

#define NODE_RANGE_TRACE(group, range)                                    \
  if (threadIdx.x == 0) {                                                 \
    CTRAN_DEV_TRACE(                                                      \
        "groupId %d/%d workerId %d/%d handles nodes %d - %d, %d nodes\n", \
        group.groupId(),                                                  \
        group.numGroups,                                                  \
        group.workerId(),                                                 \
        group.numWorkers,                                                 \
        range.start,                                                      \
        range.end,                                                        \
        range.numNodes);                                                  \
  }

#define LOCAL_RANK_RANGE_TRACE(group, range)                                        \
  if (threadIdx.x == 0) {                                                           \
    CTRAN_DEV_TRACE(                                                                \
        "groupId %d/%d workerId %d/%d handles localRanks %d - %d, %d localRanks\n", \
        group.groupId(),                                                            \
        group.numGroups,                                                            \
        group.workerId(),                                                           \
        group.numWorkers,                                                           \
        range.start,                                                                \
        range.end,                                                                  \
        range.numRanks);                                                            \
  }

#define FWDHDR_TRACE_CHECK(                                                                        \
    VERBOSE, args, hdr, fwdSrcName, fwdSrcRank, step, fwdBlockIds)                                 \
  if (threadIdx.x == 0 && workerId == 0) {                                                         \
    if (hdr->numBlocks == 0 || hdr->numBlocks > args.pArgs.maxNumStepBlks ||                       \
        hdr->opCount != args.opCount) {                                                            \
      CTRAN_DEV_FATAL(                                                                             \
          "Wrong chunk header received from %s %d step %d: sendRank %d numBlocks %d opCount %d\n", \
          fwdSrcName,                                                                              \
          fwdSrcRank,                                                                              \
          step,                                                                                    \
          hdr->sendRank,                                                                           \
          hdr->numBlocks,                                                                          \
          hdr->opCount);                                                                           \
    } else {                                                                                       \
      CTRAN_DEV_TRACE(                                                                             \
          "Received hdr %p numBlocks %d sendRank %d from %s %d step %d\n",                         \
          hdr,                                                                                     \
          hdr->numBlocks,                                                                          \
          hdr->sendRank,                                                                           \
          fwdSrcName,                                                                              \
          fwdSrcRank,                                                                              \
          step);                                                                                   \
    }                                                                                              \
    if (VERBOSE) {                                                                                 \
      for (auto b = 0; b < hdr->numBlocks; b++) {                                                  \
        CTRAN_DEV_TRACE_IF(                                                                        \
            b < 5, " Received remBlockIds[%d] %d\n", b, fwdBlockIds[b]);                           \
      }                                                                                            \
    }                                                                                              \
  }

#define SEND_DATACOPY_TRACE(numToCopy, node, myRank, step)          \
  if (threadIdx.x == 0) {                                           \
    CTRAN_DEV_TRACE(                                                \
        "Copying %d blocks to node %d, from sendRank %d step %d\n", \
        (int)numToCopy,                                             \
        node,                                                       \
        myRank,                                                     \
        step);                                                      \
  }

#define FWD_DATACOPY_TRACE(numToCopy, localRank, sendRank, step)         \
  if (threadIdx.x == 0) {                                                \
    CTRAN_DEV_TRACE(                                                     \
        "Copying %d blocks to localRank %d, from sendRank %d step %d\n", \
        (int)numToCopy,                                                  \
        localRank,                                                       \
        sendRank,                                                        \
        step);                                                           \
  }

#define RECVCOPY_DATACOPY_TRACE(numToCopy, bkt, sendRank, firstRecvIdx) \
  if (threadIdx.x == 0) {                                               \
    CTRAN_DEV_TRACE(                                                    \
        "Copying %d blocks to recvBuff[bkt %d][rank %d][%d:]\n",        \
        (int)numToCopy,                                                 \
        bkt,                                                            \
        sendRank,                                                       \
        firstRecvIdx);                                                  \
  }

#define DATACOPY_TRACE(cond, blockCount, sb, rb, sIdx, sData) \
  if constexpr (std::is_integral_v<T>) {                      \
    if (blockCount >= 2 && (cond) && threadIdx.x == 0) {      \
      CTRAN_DEV_TRACE(                                        \
          " copied sData[%d] -> rData[%d] %d: %d %d\n",       \
          sb,                                                 \
          rb,                                                 \
          sIdx,                                               \
          static_cast<int>(sData[0]),                         \
          static_cast<int>(sData[1]));                        \
    }                                                         \
  }

#define FWD_RECVSYNC_WAIT_TRACE(                                                                 \
    args, state, r, remStep, numToCopy, remSync, kIsExec)                                        \
  do {                                                                                           \
    if (threadIdx.x == 0 && workerId == 0 &&                                                     \
        state.frCheckSteps[r] < remStep) {                                                       \
      CTRAN_DEV_TRACE(                                                                           \
          "Waiting fwdRecvSync %p (offset %ld) ready, peerLocalRank %d step %d, numToCopy %d\n", \
          remSync,                                                                               \
          getOffset(                                                                             \
              remSync,                                                                           \
              kIsExec ? args.kSync.remFwdRecvSyncs[r]                                            \
                      : args.kSync.fwdRecvSyncs),                                                \
          r,                                                                                     \
          remStep,                                                                               \
          numToCopy);                                                                            \
    };                                                                                           \
    /* only for controlling logging at first check*/                                             \
    state.frCheckSteps[r] = remStep;                                                             \
  } while (0)

#define RECVCOPY_RECVSYNC_WAIT_TRACE(                                      \
    args, state, r, step, fwdRecvSync, kIsIntra, kIsExec)                  \
  if (threadIdx.x == 0 && workerId == 0 && state.frCheckSteps[r] < step) { \
    CTRAN_DEV_TRACE(                                                       \
        "Waiting %s %p (offset %ld) peerLocalRank %d step %d\n",           \
        kIsIntra ? "intra fwdRecvSync" : "fwdRecvSync",                    \
        fwdRecvSync,                                                       \
        getOffset(                                                         \
            fwdRecvSync,                                                   \
            kIsExec ? args.kSync.fwdRecvSyncs                              \
                    : args.kSync.remFwdRecvSyncs[r]),                      \
        r,                                                                 \
        step);                                                             \
    /* only for controlling logging at first check*/                       \
    state.frCheckSteps[r] = step;                                          \
  }

#define RECVRED_POST_TRACE(args, fwdLocalRank, step, stepInfo) \
  CTRAN_DEV_TRACE_IF(                                          \
      threadIdx.x == 0,                                        \
      "Posted step %d numBlocks %d to peerLocalRank %d\n",     \
      step,                                                    \
      stepInfo.numBlocks,                                      \
      fwdLocalRank);

#define PROFILE_RECVCOPY_STEP_START(args, step, workG, fwdLocalRank, isIntra) \
  do {                                                                        \
    if (workerId == 0) {                                                      \
      GpeKernelSync* gkSyncBase = isIntra ? args.kSync.intraRecvCopyGKSyncs   \
                                          : args.kSync.recvCopyGKSyncs;       \
      GpeKernelSyncDev::complete(gkSyncBase + fwdLocalRank, 0, 2 * step);     \
    }                                                                         \
  } while (0)

#define PROFILE_RECVCOPY_STEP_END(args, step, workG, fwdLocalRank, isIntra)   \
  do {                                                                        \
    if (workerId == 0) {                                                      \
      GpeKernelSync* gkSyncBase = isIntra ? args.kSync.intraRecvCopyGKSyncs   \
                                          : args.kSync.recvCopyGKSyncs;       \
      GpeKernelSyncDev::complete(gkSyncBase + fwdLocalRank, 0, 2 * step + 1); \
    }                                                                         \
  } while (0)

#define PROFILE_INTRAFWD_STEP_START(args, step, workG)    \
  do {                                                    \
    if (workerId == 0) {                                  \
      GpeKernelSync* gkSync = args.kSync.intraFwdGKSyncs; \
      GpeKernelSyncDev::complete(gkSync, 0, 2 * step);    \
    }                                                     \
  } while (0)

#define PROFILE_INTRAFWD_STEP_END(args, step, workG)       \
  do {                                                     \
    if (workerId == 0) {                                   \
      GpeKernelSync* gkSync = args.kSync.intraFwdGKSyncs;  \
      GpeKernelSyncDev::complete(gkSync, 0, 2 * step + 1); \
    }                                                      \
  } while (0)

} // namespace ctran::alltoallvdedup
