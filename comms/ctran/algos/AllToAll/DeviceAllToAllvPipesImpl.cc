// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAll/DeviceAllToAllvPipesImpl.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::device_alltoallv_pipes {

commResult_t setupKernelConfig(
    const void* sendbuff,
    void* recvbuff,
    const int64_t* sendcounts_d,
    const int64_t* recvcounts_d,
    commDataType_t datatype,
    CtranComm* comm,
    KernelConfig& config,
    ctran::device_alltoallv_pipes::KernArgs& kernArgs,
    int64_t sendcountsMultiplier,
    int64_t recvcountsMultiplier) {
  const auto statex = comm->statex_.get();

  kernArgs.sendbuff = sendbuff;
  kernArgs.recvbuff = recvbuff;
  kernArgs.elementSize = commTypeSize(datatype);
  kernArgs.myRank = statex->rank();
  kernArgs.nLocalRanks = statex->nLocalRanks();

  // Device pointers to split sizes
  kernArgs.sendcounts_d = sendcounts_d;
  kernArgs.recvcounts_d = recvcounts_d;

  // Scaling factors for multi-dimensional tensors
  kernArgs.sendcountsMultiplier = sendcountsMultiplier;
  kernArgs.recvcountsMultiplier = recvcountsMultiplier;

  // Build local rank → global rank mapping
  if (kernArgs.nLocalRanks > CTRAN_MAX_NVL_PEERS) {
    return commInternalError;
  }
  for (int lr = 0; lr < kernArgs.nLocalRanks; lr++) {
    kernArgs.localRankToGlobalRank[lr] = statex->localRankToRank(lr);
  }

  // Set transport array from MultiPeerTransport
  kernArgs.transports = comm->getMultiPeerTransportsPtr();

  // Scheduling: warp (default) vs block
  const char* blockSchedEnv = getenv("NCCL_CTRAN_DA2A_BLOCK_SCHEDULING");
  kernArgs.useBlockGroup = (blockSchedEnv && std::atoi(blockSchedEnv) == 1);

  // Grid/block config — parameterizable via env vars for tuning
  unsigned int numBlocks = std::max(1, kernArgs.nLocalRanks * 2);

  // Block scheduling requires at least 2 * nLocalRanks blocks
  // (one send + one recv block per peer)
  if (kernArgs.useBlockGroup) {
    numBlocks = std::max(
        numBlocks, static_cast<unsigned int>(kernArgs.nLocalRanks * 2));
  }
  const char* numBlocksEnv = getenv("NCCL_CTRAN_DA2A_NBLOCKS");
  if (numBlocksEnv) {
    numBlocks = static_cast<unsigned int>(std::atoi(numBlocksEnv));
  }

  unsigned int clusterSize = NCCL_CTRAN_CGA_CLUSTER_SIZE;
  if (clusterSize > 1 && numBlocks % clusterSize != 0) {
    numBlocks = ((numBlocks + clusterSize - 1) / clusterSize) * clusterSize;
  }

  // DeviceAllToAllvPipes doesn't use GPE ops (opGroup is empty) or per-block
  // sync structures (CtranAlgoDeviceSync, KernelElem). Only flag[0] is used
  // by thread 0 for start/terminate. The CTRAN_ALGO_MAX_THREAD_BLOCKS limit
  // doesn't apply here. See AllToAllvDedup (AlgoImpl.cc) for same pattern.
  config.numBlocks = numBlocks;

  unsigned int numThreads = 256;
  const char* numThreadsEnv = getenv("NCCL_CTRAN_DA2A_NTHREADS");
  if (numThreadsEnv) {
    numThreads = static_cast<unsigned int>(std::atoi(numThreadsEnv));
  } else if (NCCL_CTRAN_ALLTOALL_THREAD_BLOCK_SIZE > 0) {
    numThreads = NCCL_CTRAN_ALLTOALL_THREAD_BLOCK_SIZE;
  }
  config.numThreads = numThreads;

  // Performance tuning env vars (all read at runtime, no recompile needed):
  //   NCCL_CTRAN_DA2A_NBLOCKS  — override block count (default: nLocalRanks*2)
  //   NCCL_CTRAN_DA2A_NTHREADS — override thread count (default: 256)
  //   NCCL_CTRAN_DA2A_BLOCK_SCHEDULING=1 — use block-level scheduling
  //     (each block handles one peer) instead of warp-level (default)
  //   NCCL_CTRAN_ENALBE_CLUSTER_KERNEL_LAUNCH=1 — enable cluster launch
  //   NCCL_CTRAN_CGA_CLUSTER_SIZE — cluster size (default: 4)
  //   NCCL_CTRAN_PIPES_NVL_CHUNK_SIZE — NVL chunk size bytes (default: 512KB)

  config.args.devState_d = comm->ctran_->algo->getDevState();
  config.algoArgs = &kernArgs;

  return commSuccess;
}

} // namespace ctran::device_alltoallv_pipes
