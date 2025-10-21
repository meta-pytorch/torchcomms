// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/utils/DevUtils.cuh"
#include "comms/utils/cvars/nccl_cvars.h"

static const auto myAlgo = NCCL_ALLGATHER_ALGO::ctrd;

static commResult_t impl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  size_t sendSize =
      op->allgather.sendcount * commTypeSize(op->allgather.datatype);
  CtranComm* comm = opGroup.front()->comm_;
  const auto& statex = comm->statex_;
  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  int nSteps = ctran::utils::log2i(nRanks);
  void* recvbuff = (void*)op->allgather.recvbuff;

  CtranAlgoLogger logger(allGatherAlgoName(myAlgo), op->opCount, comm);

  void* memHdl;
  std::vector<size_t> peers(nSteps);
  std::vector<size_t> dists(nSteps);
  std::vector<void*> remoteRecvBuffs(nSteps);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(nSteps);
  std::vector<std::unique_ptr<CtranMapperRequest>> irecvReq(nSteps);
  std::vector<std::unique_ptr<CtranMapperRequest>> isendReq(nSteps);
  std::vector<std::unique_ptr<CtranMapperRequest>> iputReq(nSteps);
  std::vector<std::unique_ptr<CtranMapperNotify>> notifyVec(nSteps);
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(allGatherAlgoName(myAlgo)));

  CtranMapperContext context(
      allGatherAlgoName(myAlgo), sendSize, sendSize * nRanks);
  comm->ctran_->mapper->setContext(std::move(context));

  bool localMemReg{false};

  // Calculate distance and peer per step
  for (size_t i = 0; i < nSteps; i++) {
    dists[i] = nRanks / (2 << i);
    size_t pos = (rank / dists[i]) % 2;
    peers[i] = pos == 0 ? rank + dists[i] : rank - dists[i];
  }

  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      recvbuff, nRanks * sendSize, &memHdl, &localMemReg));

  // Exchange memory handles with relevant peers
  for (size_t i = 0; i < nSteps; i++) {
    auto peer = peers[i];
    CtranMapperRequest* recvReq = nullptr;
    FB_COMMCHECK(comm->ctran_->mapper->irecvCtrl(
        &remoteRecvBuffs[i], &remoteAccessKeys[i], peer, &recvReq));
    irecvReq[i] = std::unique_ptr<CtranMapperRequest>(recvReq);

    if (!NCCL_CTRAN_AG_RD_RTR) {
      CtranMapperRequest* sendReq = nullptr;
      FB_COMMCHECK(
          comm->ctran_->mapper->isendCtrl(recvbuff, memHdl, peer, &sendReq));
      isendReq[i] = std::unique_ptr<CtranMapperRequest>(sendReq);
    }

    // Initialize notify to receive notification from peer
    notifyVec[i] = std::make_unique<CtranMapperNotify>();
    FB_COMMCHECK(
        comm->ctran_->mapper->initNotify(peer, memHdl, notifyVec[i].get()));
  }

  for (size_t i = 0; i < nSteps; i++) {
    auto peer = peers[i];

    if (NCCL_CTRAN_AG_RD_RTR) {
      CtranMapperRequest* sendReq = nullptr;
      FB_COMMCHECK(
          comm->ctran_->mapper->isendCtrl(recvbuff, memHdl, peer, &sendReq));
      isendReq[i] = std::unique_ptr<CtranMapperRequest>(sendReq);
    }

    // Block until we have handle for this peer
    FB_COMMCHECK(comm->ctran_->mapper->waitRequest(irecvReq[i].get()));
    timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(peer));

    for (size_t j = 0; j < (1 << i); j++) {
      size_t putOffset = j * (nRanks / (1 << i)) + rank % (nRanks / (1 << i));
      // Only need to block on the final put
      bool notify = j == (1 << i) - 1;

      if (notify) {
        CtranMapperRequest* putReqPtr = nullptr;
        FB_COMMCHECK(comm->ctran_->mapper->iput(
            (char*)recvbuff + putOffset * sendSize,
            (char*)remoteRecvBuffs[i] + putOffset * sendSize,
            sendSize,
            peer,
            CtranMapperConfig{
                .memHdl_ = memHdl,
                .remoteAccessKey_ = remoteAccessKeys[i],
                .notify_ = notify},
            &putReqPtr));
        iputReq[i] = std::unique_ptr<CtranMapperRequest>(putReqPtr);
      } else {
        FB_COMMCHECK(comm->ctran_->mapper->iput(
            (char*)recvbuff + putOffset * sendSize,
            (char*)remoteRecvBuffs[i] + putOffset * sendSize,
            sendSize,
            peer,
            CtranMapperConfig{
                .memHdl_ = memHdl,
                .remoteAccessKey_ = remoteAccessKeys[i],
                .notify_ = notify},
            static_cast<CtranMapperRequest*>(nullptr)));
      }
      // Capture duration started from first put
      if (j == 0) {
        timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));
      }
    }
    // Wait for signal from receives
    FB_COMMCHECK(comm->ctran_->mapper->waitRequest(iputReq[i].get()));
    // Capture duration ended at last put when it is completed
    timestamp->putComplete.push_back(CtranMapperTimestampPoint(peer));
    FB_COMMCHECK(comm->ctran_->mapper->waitNotify(notifyVec[i].get()));
  }

  for (int i = 0; i < nSteps; i++) {
    FB_COMMCHECK(comm->ctran_->mapper->waitRequest(isendReq[i].get()));
  }

  if (localMemReg) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(memHdl));
  }

  comm->ctran_->mapper->timestamps.push_back(std::move(timestamp));
  comm->ctran_->mapper->reportProfiling();

  return commSuccess;
}

commResult_t ctranAllGatherRd(
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
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      impl,
      config,
      reinterpret_cast<void*>(ncclKernelAllGatherCtranRecDbl)));
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
