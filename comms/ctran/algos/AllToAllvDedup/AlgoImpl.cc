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
  auto numRecvGroups = NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS;

  // dedup doesn't use relevant structures that are statically initialized with
  // CTRAN_ALGO_MAX_THREAD_BLOCKS; thus the sum of thread blocks can go beyond.
  // FIXME: However, would algorithm hang if one thread block is spin-waiting a
  // signal from the other thread block from a different role?

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
