// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_fp16.h>
#include <iostream>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/ExtUtils.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

CTRAN_DATATYPE_TO_FUNC_MAPPER(kernFnMap, ncclKernelAllGatherCtranDirect);

static const auto myAlgo = NCCL_ALLGATHER_ALGO::ctdirect;

static commResult_t impl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  size_t sendSize =
      op->allgather.sendcount * commTypeSize(op->allgather.datatype);
  CtranComm* comm = opGroup.front()->comm_;

  CtranAlgoLogger logger(allGatherAlgoName(myAlgo), op->opCount, comm);

  const auto statex = comm->statex_.get();
  const int rank = statex->rank();
  const int nRanks = statex->nRanks();

  void* memHdl;
  std::vector<void*> remoteRecvBuffs(nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(nRanks);
  std::vector<std::unique_ptr<CtranMapperRequest>> irecvReq(nRanks);
  std::vector<std::unique_ptr<CtranMapperRequest>> isendReq(nRanks);
  std::vector<std::unique_ptr<CtranMapperRequest>> iputReq(nRanks);
  std::vector<std::unique_ptr<CtranMapperNotify>> notifyVec(nRanks);
  std::vector<bool> irecvComplete(nRanks, false);
  std::vector<bool> isendComplete(nRanks, false);
  std::vector<bool> iputComplete(nRanks, false);
  bool localMemReg;
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(allGatherAlgoName(myAlgo)));

  CtranMapperContext context(
      allGatherAlgoName(myAlgo), sendSize, sendSize * nRanks);
  comm->ctran_->mapper->setContext(std::move(context));
  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      op->allgather.recvbuff, nRanks * sendSize, &memHdl, &localMemReg));

  // Issue control messages for send and recv operations
  for (int p = 1; p < nRanks; p++) {
    const int peer = (rank + p) % nRanks;
    CtranMapperRequest* recvReq = nullptr;
    FB_COMMCHECK(comm->ctran_->mapper->irecvCtrl(
        &remoteRecvBuffs[peer], &remoteAccessKeys[peer], peer, &recvReq));
    irecvReq[peer] = std::unique_ptr<CtranMapperRequest>(recvReq);

    CtranMapperRequest* sendReq = nullptr;
    FB_COMMCHECK(comm->ctran_->mapper->isendCtrl(
        op->allgather.recvbuff, memHdl, peer, &sendReq));
    isendReq[peer] = std::unique_ptr<CtranMapperRequest>(sendReq);
    // Initialize notify to receive notification from inter-node peers
    // NOTE: any intra-node peer without NVL backend will cause error after
    // received ctrl msg; thus skip notify for such peers here.
    if (!statex->isSameNode(rank, peer)) {
      notifyVec[peer] = std::make_unique<CtranMapperNotify>();
      FB_COMMCHECK(comm->ctran_->mapper->initNotify(
          peer, memHdl, notifyVec[peer].get()));
    }
  }

  irecvComplete[rank] = true;
  isendComplete[rank] = true;
  iputComplete[rank] = true;

  // Post intranode bcast first
  KernelElem* elem = op->allgather.bcastElem;
  const int nLocalRanks = statex->nLocalRanks();
  const int localRank = statex->localRank();
  for (int p = 0; p < nLocalRanks; p++) {
    int pRank = statex->localRankToRank(p);
    if (p == localRank) {
      elem->bcast.dsts[p] = (char*)op->allgather.recvbuff + rank * sendSize;
    } else {
      // Wait for receiving remote recv buffer from a local peer
      comm->ctran_->mapper->waitRequest(irecvReq[pRank].get());
      if (remoteAccessKeys[pRank].backend != CtranMapperBackend::NVL) {
        CLOGF(
            ERR,
            "NVLink backend not available between rank {} and {}",
            rank,
            pRank);
        return commInternalError;
      }
      elem->bcast.dsts[p] = (char*)remoteRecvBuffs[pRank] + rank * sendSize;
    }
  }

  elem->post();

  // Post remaining inter-node puts
  bool pendingRecv;
  do {
    pendingRecv = false;
    for (int p = 1; p < nRanks; p++) {
      const int peer = (rank + p) % nRanks;
      if (irecvComplete[peer] == true || statex->isSameNode(rank, peer)) {
        // Skip already issued PUT or any intra-node peer
        continue;
      }

      bool isComplete;
      FB_COMMCHECK(
          comm->ctran_->mapper->testRequest(irecvReq[peer].get(), &isComplete));
      irecvComplete[peer] = isComplete;
      if (irecvComplete[peer] == false) {
        pendingRecv = true;
        continue;
      }

      timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(peer));

      CtranMapperRequest* putReq = nullptr;
      // iput internally dispatches to either network put or NVL copy
      FB_COMMCHECK(comm->ctran_->mapper->iput(
          (char*)op->allgather.recvbuff + rank * sendSize,
          (char*)remoteRecvBuffs[peer] + rank * sendSize,
          sendSize,
          peer,
          CtranMapperConfig{
              .memHdl_ = memHdl,
              .remoteAccessKey_ = remoteAccessKeys[peer],
              .notify_ = true},
          &putReq));
      iputReq[peer] = std::unique_ptr<CtranMapperRequest>(putReq);
      timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));
    }
  } while (pendingRecv == true);

  // Wait for all PUTs to complete
  for (int p = 1; p < nRanks; p++) {
    int peer = (rank + p) % nRanks;

    if (isendComplete[peer] == false) {
      FB_COMMCHECK(comm->ctran_->mapper->waitRequest(isendReq[peer].get()));
    }
    // Wait put and notify completion only for inter-node peers
    if (statex->isSameNode(rank, peer)) {
      continue;
    }
    if (iputComplete[peer] == false) {
      FB_COMMCHECK(comm->ctran_->mapper->waitRequest(iputReq[peer].get()));
      timestamp->putComplete.push_back(CtranMapperTimestampPoint(peer));
    }
    FB_COMMCHECK(comm->ctran_->mapper->waitNotify(notifyVec[peer].get()));
  }

  // Wait for intranode bcast to complete
  elem->wait();

  if (localMemReg) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(memHdl));
  }

  comm->ctran_->mapper->timestamps.emplace_back(std::move(timestamp));
  comm->ctran_->mapper->reportProfiling();

  return commSuccess;
}

static unsigned int bestThreadBlockSize = 0;

static inline unsigned int getThreadBlockSize() {
  // If first time call, query cuda recommended blockSize
  if (bestThreadBlockSize == 0) {
    int minGridSize = 0;
    XCHECK(kernFnMap.contains(commFloat32))
        << "kernFnMap does not contain datatype";
    FB_CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        (int*)&bestThreadBlockSize,
        kernFnMap.at(commFloat32),
        0 /* dynamicSMemSize */,
        0 /* blockSizeLimit */));
  }

  return NCCL_CTRAN_NVL_ALLGATHERDIRECT_THREAD_BLOCK_SIZE == -1
      ? bestThreadBlockSize
      : NCCL_CTRAN_NVL_ALLGATHERDIRECT_THREAD_BLOCK_SIZE;
}

static inline int getNumGroups(size_t nbytes) {
  // compute needed thread blocks for given bytes
  int nGroups = nbytes / NCCL_CTRAN_NVL_ALLGATHERDIRECT_CHUNK_SIZE;
  return std::min(
      std::max(1, nGroups), // at least 1 thread block
      // not exceed max theshold
      NCCL_CTRAN_NVL_ALLGATHERDIRECT_MAX_NUM_THREAD_BLOCKS);
}

static inline commResult_t setupPlan(
    CtranComm* comm,
    std::vector<std::unique_ptr<OpElem>>& opGroup,
    KernelConfig& config) {
  int maxNumBlocks = 1;

  // Allocate a p2pElem to coordinate with kernel for each send and recv.
  // - For sends, p2pElem has putNotify where the recvbuff will be assigned and
  // the elem will be posted to kernel once GPE thread imports remote memory.
  // - For recvs, p2pElem has waitNotify where the elem will be posted once GPE
  // thread confirms the local memory registration.
  // - If an elem with a buffer not qualified for NVL backend, the elem will
  // be revoked by GPE thread, thus kernel will skip it.
  auto& op = opGroup.front();

  KernelElem* bcastElem = nullptr;
  size_t nBytes =
      op->allgather.sendcount * commTypeSize(op->allgather.datatype);
  int nGroups = getNumGroups(nBytes);
  // record the max number of thread blocks as final launching grid size
  maxNumBlocks = std::max(maxNumBlocks, nGroups);
  FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, nGroups, &bcastElem));
  // Nothing need specify from host since all info are contained in
  // CtranKernelAllGatherArgs except dsts which would need GPE thread to fill in
  op->allgather.bcastElem = bcastElem;

  config.args.collective.allgather.bcastElem = bcastElem;

  // Allow user to increase SM usage for NVL copy
  config.numBlocks = maxNumBlocks;
  config.numThreads = getThreadBlockSize();

  return commSuccess;
}

commResult_t ctranAllGatherDirect(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream) {
  CTRAN_COLL_INFO(
      allGatherAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      sendcount,
      datatype,
      -1,
      comm,
      stream);
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHER,
      stream,
      allGatherAlgoName(myAlgo),
      comm->ctran_->getOpCount());
  void* extraCopyBuff = nullptr;
  FB_COMMCHECK(prepareAllGatherArgs(
      opGroup,
      config,
      &extraCopyBuff,
      sendbuff,
      recvbuff,
      sendcount,
      datatype,
      comm,
      stream));
  FB_COMMCHECK(setupPlan(comm, opGroup, config));
  XCHECK(kernFnMap.contains(datatype))
      << "kernFnMap does not contain datatype " << datatype;
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup), impl, config, kernFnMap.at(datatype)));
  if (extraCopyBuff != nullptr) {
    FB_CUDACHECK(cudaMemcpyAsync(
        recvbuff,
        extraCopyBuff,
        sendcount * commTypeSize(datatype) * comm->statex_->nRanks(),
        cudaMemcpyDefault,
        stream));
  }
  return commSuccess;
}
