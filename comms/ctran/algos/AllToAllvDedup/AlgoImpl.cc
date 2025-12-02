// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAllvDedup/AlgoImpl.h"
#include "comms/ctran/algos/AllToAllvDedup/CommonDev.h"

namespace ctran::alltoallvdedup {
using namespace utils;

AlgoImpl::AlgoImpl(
    CtranComm* comm,
    ncclx::CommStateX* statex,
    ICtran* ctran,
    cudaStream_t stream)
    : comm_(comm), statex_(statex), ctran_(ctran), stream_(stream) {
  resource_ = std::make_unique<ResourceImpl>(
      statex, ctran->mapper.get(), &comm->logMetaData_);
}

commResult_t AlgoImpl::initConfig() {
  config_.numThreads = NCCL_CTRAN_ALLTOALLV_DEDUP_THREAD_BLOCK_SIZE;

  // Check specified number of thread blocks not exceed the maximum
  auto numSendGroups = NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS;
  auto numSendWorkers =
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP;

  auto numFwdWorkers = NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS;
  auto numRecvGroups = NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS;
  auto numRecvWorkers =
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP;
  auto numIntraFwdWorkers =
      NCCL_CTRAN_ALLTOALLV_DEDUP_INTRA_FWD_NUM_THREAD_BLOCKS;
  auto numIntraRecvWorkers =
      NCCL_CTRAN_ALLTOALLV_DEDUP_INTRA_RECV_NUM_THREAD_BLOCKS;

  auto numThreadBlocks = numSendGroups * numSendWorkers + numFwdWorkers +
      numRecvGroups * numRecvWorkers + numIntraFwdWorkers + numIntraRecvWorkers;
  if (numThreadBlocks > CTRAN_ALGO_MAX_THREAD_BLOCKS) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Total number of thread blocks {} ("
        "(NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS {} * NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP {} + "
        "NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS {} + "
        "NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS {} * NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP {} + "
        "NCCL_CTRAN_ALLTOALLV_DEDUP_INTRA_FWD_NUM_THREAD_BLOCKS {} + NCCL_CTRAN_ALLTOALLV_DEDUP_INTRA_RECV_NUM_THREAD_BLOCKS {}) must be <= {}",
        numThreadBlocks,
        numSendGroups,
        numSendWorkers,
        numFwdWorkers,
        numRecvGroups,
        numRecvWorkers,
        numIntraFwdWorkers,
        numIntraRecvWorkers,
        CTRAN_ALGO_MAX_THREAD_BLOCKS);
  }
  if (numSendGroups > MAX_NUM_GROUPS_PER_ROLE) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS {} must be <= {}",
        numSendGroups,
        MAX_NUM_GROUPS_PER_ROLE);
  }
  if (numRecvGroups > MAX_NUM_GROUPS_PER_ROLE) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS {} must be <= {}",
        numRecvGroups,
        MAX_NUM_GROUPS_PER_ROLE);
  }

  if (NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE <= 0 ||
      NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS <= 0) {
    FB_ERRORRETURN(
        commInvalidUsage,
        "Invalid NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE {} or NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS {}. Both must be > 0\n",
        NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE,
        NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS);
  }

  config_.tmpChunkSize = NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE;
  config_.tmpNumChunks = NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS;

  pArgs.typeSize = commTypeSize(pArgs.datatype);
  pArgs.maxNumStepBlks = getMaxNumBlocksPerChunk(&config_, pArgs);
  const auto nRanks = statex_->nRanks();
  pArgs.maxNumSteps =
      pArgs.totalNumSendBlocks * nRanks / pArgs.maxNumStepBlks + 1;
  return commSuccess;
}

commResult_t AlgoImpl::initialize() {
  FB_COMMCHECK(initConfig());
  FB_COMMCHECK(resource_->initialize(pArgs, config_, stream_));
  return commSuccess;
}
} // namespace ctran::alltoallvdedup
