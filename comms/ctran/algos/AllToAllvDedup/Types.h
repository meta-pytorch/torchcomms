// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <sstream>
#include "comms/ctran/algos/AllToAllvDedup/WorkerGroup.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/algos/common/SpscP2pSync.h"
#include "comms/ctran/utils/DevAttribute.h"
#include "comms/utils/commSpecs.h" // need for ncclDataType_t

// Define types shared by both host and device code

namespace ctran::alltoallvdedup {

struct PersistConfig {
  int numThreads;

  int numSendGroups;
  int numSendWorkers;
  int numFwdWorkers;
  int numRecvGroups;
  int numRecvWorkers;
  int numIntraFwdWorkers;
  int numIntraRecvWorkers;

  int tmpChunkSize;
  int tmpNumChunks;
};

// Persistent arguments specified at init time
struct PersistArgs {
  int totalNumSendBlocks;
  int blockCount;
  int blockNumRecvBuckets;
  int numRecvBuckets;
  commDataType_t datatype;
  size_t typeSize;

  // not passed by user, but pre-calculated at init
  int maxNumSteps;
  int maxNumStepBlks;
};

struct ExecArgs {
  // Dispatch input arguments
  const void* sendBuff;

  // nNodes * totalNumSendBlocks
  const int* sendIdx;
  // nLocalRanks * nNodes * totalNumSendBlocks
  const int* fwdIdx;
  // numRecvBuckets * nRanks * totalNumSendBlocks
  const int* recvIdx;

  // Dispatch output arguments
  void* recvBuff;
  int* recvBlockIds;
};

struct alignas(16) FwdRecvSync {
  algos::SpscP2pSync spsc;
};

struct alignas(16) IntraRedSync {
  int cnt;
};

struct KernSync {
  // WorkerGroupType::kNumTypes * MAX_NUM_GROUPS_PER_ROLE sync objects. Number
  // of groups is determined at runtime, and only used groups are reset in
  // prepare() and used in exec() / combine.
  WorkerGroupSync* wgSyncs;

  // used to sync between forward groups and receive local ranks' recv groups on
  // the same node. Reset once at resource init
  FwdRecvSync* fwdRecvSyncs;
  FwdRecvSync* remFwdRecvSyncs[CTRAN_MAX_NVL_PEERS];
  FwdRecvSync* intraFwdRecvSyncs;
  FwdRecvSync* remIntraFwdRecvSyncs[CTRAN_MAX_NVL_PEERS];

  // sync between local sendRed and intraRecv workers.
  // Reset before each exec()/combine()
  IntraRedSync* intraRedSync;

  // used to sync between kernel and GPE on send rank and forward rank,
  // respectively. Do not use arguments in kElem since it is slow host-pinned
  // memory. Each element in the list is for one rail peer.
  // Reset before each exec()/combine()
  algos::GpeKernelSync* sendGKSyncs;
  algos::GpeKernelSync* recvGKSyncs;
  algos::GpeKernelSync* intraFwdGKSyncs;
  algos::GpeKernelSync* recvCopyGKSyncs;
  algos::GpeKernelSync* intraRecvCopyGKSyncs;
};

struct InitKernArgs {
  PersistArgs pArgs;
  PersistConfig config;
  KernSync kSync;
};

struct RecvCopyStepInfo {
  int numBlocks;
  int sendRank;
};

struct ExecKernArgs {
  uint64_t opCount;
  PersistArgs pArgs;
  ExecArgs execArgs;
  PersistConfig config;

  // Used by send rank to copy blocks into contig fixed size chunk for RDMA
  void* tmpSendBuff;
  // used by fwd rank to load incoming fwdArgs + data chunk
  void* tmpFwdBuff;

  // nNodes * totalNumSendBlocks
  // store blockIdx sending to each node
  int* tmpSendIdx;
  // nNodes
  // store number of blocks sending to each node
  int* tmpNumSendIdx;

  // nNodes * maxNumSteps
  // track number of pending blocks to be reduced from remote nodes at each step
  // on send rank. Used in combine
  int* tmpSendRedStepNumPending;
  // totalNumSendBlocks
  // store the number of reduce sources (remote node) for a block.
  int* tmpSendRedIdxNumSrcs;
  int* tmpSendRedIdxNumPendingSrcs;
  // totalNumSendBlocks * nNodes
  // store the reduce sources (remote node) for a block.
  int* tmpSendRedIdxSrcIds;

  // nLocalRanks * maxNumSteps
  // track number of pending blocks to be reduced from local recv ranks at each
  // step on send rank. Used in combine
  int* tmpIntraRedStepNumPending;
  // totalNumSendBlocks
  // store the number of reduce sources (local recv rank) for a block.
  int* tmpIntraRedIdxNumSrcs;
  int* tmpIntraRedIdxNumPendingSrcs;
  // totalNumSendBlocks * nLocalRanks
  // store the reduce sources (local recv rank) for a block.
  int* tmpIntraRedIdxSrcIds;

  // nRanks * totalNumSendBlocks
  // store the number of reduce buckets for a block.
  int* tmpRecvRedIdxNumSrcs;
  // nRanks * totalNumSendBlocks * numRecvBuckets
  // store the reduce buckets for a block.
  int* tmpRecvRedIdxSrcIds;

  // nLocalRanks * nNodes * totalNumSendBlocks
  // store blockIdx forwarding to each local rank from each send node
  int* tmpFwdIdx;
  // nNodes
  int* tmpNumFwdIdx;
  // nLocalRanks
  int* tmpNumIntraFwdIdx;
  // numRecvBuckets * nRanks * totalNumSendBlocks
  // store blockIdx receiving from each send rank to each local bucket
  int* tmpRecvIdx;
  // nLocalRanks, count merged recvIdx from all buckets, as number of blocks to
  // be received from each forward rank
  int* tmpNumFwdRecvIdx;
  // nLocalRanks, count merged recvIdx from all buckets, as number of blocks to
  // be received from each local send rank
  int* tmpNumIntraRecvIdx;
  // numRecvBuckets * nRanks, starting offset in number of blocks of each bucket
  // and from each send rank in recv buffer
  int* tmpRecvOffsets;

  // Used by recv rank to receive forwarded chunk from local forward ranks
  void* tmpRecvBuff;
  // IPC imported ptr to each of the local peers' tmpRecvBuff
  void* remTmpRecvBuffs[CTRAN_MAX_NVL_PEERS];

  // Similar to tmpRecvBuff, but only for intraFwd
  void* tmpIntraRecvBuff;
  void* remTmpIntraRecvBuffs[CTRAN_MAX_NVL_PEERS];

  // nNodes
  // host copy of tmpNumSendIdx, for host side to track send progress.
  int* tmpNumSendIdxH;
  // nNodes
  // host copy of tmpNumFwdIdx, for host side to track send progress.
  int* tmpNumFwdIdxH;
  // nLocalRanks
  // host copy of tmpNumIntraFwdIdx, for host side to track intraFwd profiling.
  int* tmpNumIntraFwdIdxH;
  // nLocalRanks
  // host copy of tmpNumIntraRecvIdx, for host side to track intraRecv
  // profiling.
  int* tmpNumIntraRecvIdxH;
  // nLocalRanks
  // host copy of tmpNumFwdRecvIdx, for host side to track recv profiling.
  int* tmpNumFwdRecvIdxH;

  // maxNumSteps * nLocalRanks * sizeof(RecvCopyStepInfo)
  RecvCopyStepInfo* recvStepInfo;
  // maxNumSteps * nLocalRanks * kMaxNumBlocksPerChunk
  int* recvStepFwdBlockIds;

  // maxNumSteps * nLocalRanks * sizeof(RecvCopyStepInfo)
  RecvCopyStepInfo* intraRecvStepInfo;
  // maxNumSteps * nLocalRanks * kMaxNumBlocksPerChunk
  int* intraRecvStepFwdBlockIds;

  // maxNumSteps * nLocalRanks
  int* fwdStepRecvrNumBlocks;
  // maxNumSteps * nNodes * kMaxNumBlocksPerChunk
  int* fwdStepBlockIds;
  // maxNumSteps * nNodes
  int* fwdStepNumBlocks;

  KernSync kSync;
};

// FWD metadata generated by sender side, and transfer to fwd rank's
// tmpbuf in GPU mem together with data chunk.
// Format: fwdHdr + recvLocalRanksBitMap[numBlocks] + blocks[numBlocks]
struct alignas(16) FwdChkHdr {
  int numBlocks; // number of actual blocks in the fwd chunk
  int sendRank; // sender rank of all chunks in the fwd chunk
  int opCount;
};
} // namespace ctran::alltoallvdedup
