// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_fp16.h>
#include <cstddef>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#ifdef ENABLE_META_COMPRESSION
#include "comms/ctran/algos/AllToAll/fb/compressed/CompressedAllToAllv.h"
#endif

#define RETURN_ALLTOALLV_IB_IMPL(perfconfig) \
  return ctranAllToAllvIbImpl<perfconfig>(   \
      op->alltoallv.sendbuff,                \
      op->alltoallv.sendcounts,              \
      op->alltoallv.sdispls,                 \
      op->alltoallv.recvbuff,                \
      op->alltoallv.recvcounts,              \
      op->alltoallv.rdispls,                 \
      op->alltoallv.datatype,                \
      comm,                                  \
      std::move(timestamp));

static void* alltoallvKerns[commNumTypes] = {
    (void*)ncclKernelAllToAllv<int8_t>,
    (void*)ncclKernelAllToAllv<uint8_t>,
    (void*)ncclKernelAllToAllv<int32_t>,
    (void*)ncclKernelAllToAllv<uint32_t>,
    (void*)ncclKernelAllToAllv<int64_t>,
    (void*)ncclKernelAllToAllv<uint64_t>,
    (void*)ncclKernelAllToAllv<half>,
    (void*)ncclKernelAllToAllv<float>,
    (void*)ncclKernelAllToAllv<double>,
#if defined(__CUDA_BF16_TYPES_EXIST__)
    (void*)ncclKernelAllToAllv<__nv_bfloat16>,
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
    (void*)ncclKernelAllToAllv<__nv_fp8_e4m3>,
    (void*)ncclKernelAllToAllv<__nv_fp8_e5m2>,
#endif
};

static const auto myAlgo = NCCL_ALLTOALLV_ALGO::ctran;

static inline commResult_t setupKernelConfig(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    KernelConfig& config) {
  const auto statex = comm->statex_.get();
  // Unlike alltoall, we cannot automatically detect grid size because each rank
  // may see different counts; use static gridSize for now.
  config.numThreads = NCCL_CTRAN_ALLTOALLV_THREAD_BLOCK_SIZE;
  config.numBlocks = NCCL_CTRAN_ALLTOALLV_NUM_THREAD_BLOCKS;

  // Adjust gridSize to fit alltoallv kernel algorithm:
  // 1. gridSize must be even number, because we split blocks into two sets of
  //   groups, one for sends and the other for receives, each send and receive
  //   pair must use the same number of blocks
  if (config.numBlocks % 2) {
    config.numBlocks += 1;
  }
  // 2. gridSize must be <= CTRAN_ALGO_MAX_THREAD_BLOCKS, since internal
  //   states/flags holds at most CTRAN_ALGO_MAX_THREAD_BLOCKS blocks
  if (config.numBlocks < 2 || config.numBlocks > CTRAN_ALGO_MAX_THREAD_BLOCKS) {
    config.numBlocks = CTRAN_ALGO_MAX_THREAD_BLOCKS;
  }

  config.args.devState_d = comm->ctran_->algo->getDevState();
  config.args.collective.alltoallv.sendbuff = sendbuff;
  config.args.collective.alltoallv.recvbuff = recvbuff;
  config.args.collective.alltoallv.datatype = datatype;
  config.args.collective.alltoallv.selfCount = sendcounts[statex->rank()];
  config.args.collective.alltoallv.selfSendDispl = sdispls[statex->rank()];
  config.args.collective.alltoallv.selfRecvDispl = rdispls[statex->rank()];

  // special case of ppn=1, simply set sendElemsList and recvElemsList to
  // nullptr, and numBlocks to 1 to let alltoallv kernel skip copies over NVLink
  // and only do self copy
  if (statex->nLocalRanks() == 1) {
    config.args.collective.alltoallv.sendElemsList = nullptr;
    config.args.collective.alltoallv.recvElemsList = nullptr;
    config.numBlocks = 1;
    return commSuccess;
  }

  // Pass number of thread block groups to kernel p2p elements
  // - Half blocks handle send, and the other handle receive
  // - Used in p2p elem to ensure ngroups number of inuse flags are checked when
  // reclaiming. This avoids cross-block sync in kernel
  const int ngroups = config.numBlocks / 2;
  comm->ctran_->gpe->allocKernelElems(
      statex->nLocalRanks() - 1,
      ngroups,
      &config.args.collective.alltoallv.sendElemsList);
  comm->ctran_->gpe->allocKernelElems(
      statex->nLocalRanks() - 1,
      ngroups,
      &config.args.collective.alltoallv.recvElemsList);

  // Ensure each rank sends to different peer at a time to avoid alltoone P2P
  // write congestion. For example, with localRanks = 4, the following
  // schedule is used:
  // - Round0:
  // rank0: s(1)r(3); rank1: s(2)r(0); rank2: s(3)r(1); rank3: s(0)r(2)
  // - Round1:
  // rank0: s(2)r(2); rank1: s(3)r(3); rank2: s(0)r(0); rank3: s(1)r(1)
  // - Round2:
  // rank0: s(3)r(1); rank1: s(0)r(2); rank2: s(1)r(3); rank3: s(2)r(0)
  KernelElem* sendElem = config.args.collective.alltoallv.sendElemsList;
  KernelElem* recvElem = config.args.collective.alltoallv.recvElemsList;
  for (int r = 0; r < statex->nLocalRanks() - 1; r++) {
    int sendPeer = (statex->localRank() + r + 1) % statex->nLocalRanks();
    int recvPeer = (statex->localRank() + statex->nLocalRanks() - r - 1) %
        statex->nLocalRanks();
    int sendPeerGlobal = statex->localRankToRank(sendPeer);
    int recvPeerGlobal = statex->localRankToRank(recvPeer);

    sendElem->staged.peerRank = sendPeer;
    sendElem->staged.count = sendcounts[sendPeerGlobal];
    sendElem->staged.displ = sdispls[sendPeerGlobal];
    sendElem = sendElem->next;

    recvElem->staged.peerRank = recvPeer;
    recvElem->staged.count = recvcounts[recvPeerGlobal];
    recvElem->staged.displ = rdispls[recvPeerGlobal];
    recvElem = recvElem->next;
  }

  return commSuccess;
}

static commResult_t opIbImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = opGroup.front()->comm_;

  CtranAlgoLogger logger(allToAllvAlgoName(myAlgo), op->opCount, comm);

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(allToAllvAlgoName(myAlgo)));

  if (NCCL_CTRAN_ENABLE_PRECONNECT) {
    RETURN_ALLTOALLV_IB_IMPL(LowLatencyCollConfig);
  } else {
    RETURN_ALLTOALLV_IB_IMPL(DefaultPerfCollConfig);
  }
}

static inline commResult_t setupGpeOp(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    commDataType_t datatype,
    CtranComm* comm,
    uint64_t opCount,
    std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  const auto statex = comm->statex_.get();
  // Passing op only when remote peers are present
  if (statex->nLocalRanks() < statex->nRanks()) {
    std::unique_ptr<struct OpElem> op = std::unique_ptr<struct OpElem>(
        new OpElem(OpElem::opType::ALLTOALLV, comm, opCount));
    op->alltoallv.sendbuff = sendbuff;
    op->alltoallv.recvbuff = recvbuff;
    op->alltoallv.datatype = datatype;
    op->alltoallv.sendcounts.resize(statex->nRanks(), 0);
    op->alltoallv.sdispls.resize(statex->nRanks(), 0);
    op->alltoallv.recvcounts.resize(statex->nRanks(), 0);
    op->alltoallv.rdispls.resize(statex->nRanks(), 0);

    size_t totalSendCount = 0, totalRecvCount = 0;
    const int myNode = statex->node();
    for (int i = 0; i < statex->nRanks(); i++) {
      const int peerNode = statex->node(i);
      // GPE thread handles only remote peers
      if (myNode != peerNode) {
        op->alltoallv.sendcounts[i] = sendcounts[i];
        op->alltoallv.sdispls[i] = sdispls[i];
        op->alltoallv.recvcounts[i] = recvcounts[i];
        op->alltoallv.rdispls[i] = rdispls[i];

        totalSendCount += sendcounts[i];
        totalRecvCount += recvcounts[i];
      } else {
        // data to itself (i.e., HBM copy) will be handled by
        // ncclKernelAllToAllv kernel
        op->alltoallv.sendcounts[i] = 0;
        op->alltoallv.recvcounts[i] = 0;
      }
    }
    // if contains either non-zero send or receive, pass op
    if (totalSendCount || totalRecvCount) {
      opGroup.push_back(std::move(op));
    }
  }
  return commSuccess;
}

commResult_t ctranAllToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream) {
  auto opCount = comm->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      allToAllvAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      0UL,
      datatype,
      -1,
      comm,
      stream);
  const auto statex = comm->statex_.get();
  for (int i = 0; i < statex->nRanks(); i++) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "{}: opCount {} - sendcounts[{}] {} sdispls[{}] {} recvcounts[{}] {} rdispls[{}] {}",
        allToAllvAlgoName(myAlgo),
        opCount,
        i,
        sendcounts[i],
        i,
        sdispls[i],
        i,
        recvcounts[i],
        i,
        rdispls[i]);
  }

#ifdef ENABLE_META_COMPRESSION
  if ((NCCL_ALLTOALLV_ALGO == NCCL_ALLTOALLV_ALGO::compCtran) &&
      ctranCompressedAllToAllvSupport(comm)) {
    return ctranCompressedAllToAllv(
        sendbuff,
        sendcounts,
        sdispls,
        recvbuff,
        recvcounts,
        rdispls,
        datatype,
        comm,
        stream);
  } else if (
      (NCCL_ALLTOALLV_ALGO == NCCL_ALLTOALLV_ALGO::bsCompCtran) &&
      ctranCompressedAllToAllvSupport(comm)) {
    return ctranBootstrapCompressedAllToAllv(
        sendbuff,
        sendcounts,
        sdispls,
        recvbuff,
        recvcounts,
        rdispls,
        datatype,
        comm,
        stream);
  }
#endif

  // prepare kernel config for self and NVL copies
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALLV,
      stream,
      allToAllvAlgoName(myAlgo),
      opCount);
  FB_COMMCHECK(setupKernelConfig(
      sendbuff,
      sendcounts,
      sdispls,
      recvbuff,
      recvcounts,
      rdispls,
      datatype,
      comm,
      stream,
      config));

  // prepare operation for IB path
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  FB_COMMCHECK(setupGpeOp(
      sendbuff,
      sendcounts,
      sdispls,
      recvbuff,
      recvcounts,
      rdispls,
      datatype,
      comm,
      opCount,
      opGroup));

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      opIbImpl,
      config,
      reinterpret_cast<void*>(alltoallvKerns[datatype])));

  return commSuccess;
}

bool ctranAllToAllvSupport(CtranComm* comm) {
  bool ctranSupport = false;
  const auto statex = comm->statex_.get();
  if (ctranInitialized(comm)) {
    ctranSupport = true;
    // Check if all remote peers are supported by ctran
    // For intra-node peers, ctranAlgo supports copy based path;
    // for inter-node peers, we need a mapper backend to support.
    const int myNode = statex->node();
    for (int rank = 0; rank < statex->nRanks(); rank++) {
      if (statex->node(rank) != myNode &&
          comm->ctran_->mapper->getBackend(rank) == CtranMapperBackend::UNSET) {
        ctranSupport = false;
        break;
      }
    }
  }

  return ctranSupport;
}
