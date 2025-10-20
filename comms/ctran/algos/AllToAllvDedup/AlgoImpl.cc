// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAllvDedup/AlgoImpl.h"

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
  config_.numPrepareThreadBlocks =
      NCCL_CTRAN_ALLTOALLV_DEDUP_PREPARE_NUM_THREAD_BLOCK_GROUPS;
  config_.numPrepareThreads =
      (unsigned int)PrepareKernRole::kNumRoles * kWarpSize;

  // Only supports multiple workers for forward groups, since it is hard to
  // saturate NVL BW with only 1 SM
  config_.numSendGroups =
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS;
  config_.numSendWorkers =
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP;

  config_.numFwdGroups = NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCK_GROUPS;
  config_.numFwdWorkers =
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS_PER_GROUP;

  config_.numRecvGroups =
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS;
  config_.numRecvWorkers =
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP;

  // FIXME: add cvar for RECVCPY_NUM_THREAD_BLOCK_GROUPS
  config_.numThreadBlocks = config_.numSendGroups * config_.numSendWorkers +
      config_.numFwdGroups * config_.numFwdWorkers +
      config_.numRecvGroups * config_.numRecvWorkers;
  config_.numThreads = NCCL_CTRAN_ALLTOALLV_DEDUP_THREAD_BLOCK_SIZE;

  if (config_.numThreadBlocks > CTRAN_ALGO_MAX_THREAD_BLOCKS) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Total number of thread blocks {} ("
        "(NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS {} * NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP {} + "
        "NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCK_GROUPS {} * NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS_PER_GROUP {} + "
        "NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS {} * NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP {}) must be <= {}",
        config_.numThreadBlocks,
        NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS,
        NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP,
        NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCK_GROUPS,
        NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS_PER_GROUP,
        NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS,
        NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP,
        CTRAN_ALGO_MAX_THREAD_BLOCKS);
  }

  if (NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE <= 0 ||
      NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS <= 0) {
    FB_ERRORRETURN(
        commInvalidUsage,
        "Invalid NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE {} or NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS {}\n",
        NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE,
        NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS);
  }

  config_.tmpChunkSize = NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE;
  config_.tmpNumChunks = NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS;

  return commSuccess;
}

commResult_t AlgoImpl::initialize() {
  FB_COMMCHECK(initConfig());
  FB_COMMCHECK(resource_->initialize(pArgs, config_, stream_));
  return commSuccess;
}
} // namespace ctran::alltoallvdedup
