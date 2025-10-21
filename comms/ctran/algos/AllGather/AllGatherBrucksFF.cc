// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/utils/DevUtils.cuh"

static const auto myAlgo = NCCL_ALLGATHER_ALGO::ctbrucks;

// # of steps is ceil(log2(p))
int ceilLog2I(int p) {
  return 1 + ctran::utils::log2i(p - 1);
}

// Who is sending to this rank at this step
int srcAtStep(int p, int rank, int step) {
  int k = ceilLog2I(p);
  int dist = 1 << (k - step - 1);
  return (rank + dist) % p;
}

// Who is this rank sending to at this step
int dstAtStep(int p, int rank, int step) {
  int k = ceilLog2I(p);
  int dist = 1 << (k - step - 1);
  return (p + rank - dist) % p;
}

// Used in computing valid state at a given step
std::vector<int> offsetsAtStep(int k, int step) {
  std::vector<int> offsets;
  int offset = 1 << (k - step);
  for (int i = offset; i < (1 << k); i += offset) {
    offsets.push_back(i);
  }
  return offsets;
}

// Compute which blocks are valid on a rank at a given step
std::vector<bool> stateAtStep(int p, int rank, int step) {
  std::vector<bool> state(p, false);
  state.at(rank) = true;
  for (const auto& offset : offsetsAtStep(ceilLog2I(p), step)) {
    state.at((rank + offset) % p) = true;
  }
  return state;
}

// Using the states at sender and receiver, compute which blocks need to be
// transmitted at a given step
std::vector<int> sendsAtStep(int p, int rank, int step) {
  auto dst = dstAtStep(p, rank, step);
  auto srcState = stateAtStep(p, rank, step);
  auto dstState = stateAtStep(p, dst, step);

  std::vector<int> sends;
  for (int i = 0; i < p; i++) {
    if (srcState[i] && !(dstState[i])) {
      sends.push_back(i);
    }
  }
  return sends;
}

static commResult_t impl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  size_t sendSize =
      op->allgather.sendcount * commTypeSize(op->allgather.datatype);
  CtranComm* comm = opGroup.front()->comm_;
  const auto& statex = comm->statex_;
  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  int nSteps = ceilLog2I(nRanks);
  void* recvbuff = (void*)op->allgather.recvbuff;

  CtranAlgoLogger logger(allGatherAlgoName(myAlgo), op->opCount, comm);

  void* memHdl;
  std::vector<void*> remoteRecvBuffs(nSteps);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(nSteps);
  std::vector<std::unique_ptr<CtranMapperRequest>> irecvReq(nSteps);
  std::vector<std::unique_ptr<CtranMapperRequest>> isendReq(nSteps);
  std::vector<std::unique_ptr<CtranMapperRequest>> iputReqA(nSteps),
      iputReqB(nSteps);
  std::vector<std::unique_ptr<CtranMapperNotify>> notifyVec(nSteps);
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(allGatherAlgoName(myAlgo)));

  CtranMapperContext context(
      allGatherAlgoName(myAlgo), sendSize, sendSize * nRanks);
  comm->ctran_->mapper->setContext(std::move(context));

  bool localMemReg{false};
  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      recvbuff, nRanks * sendSize, &memHdl, &localMemReg));

  // Post receives for memory handles
  for (size_t i = 0; i < nSteps; i++) {
    auto srcPeer = srcAtStep(nRanks, rank, i);
    auto dstPeer = dstAtStep(nRanks, rank, i);

    // Receive destination memory address from peer we're sending to
    CtranMapperRequest* recvReq = nullptr;
    FB_COMMCHECK(comm->ctran_->mapper->irecvCtrl(
        &remoteRecvBuffs[i], &remoteAccessKeys[i], dstPeer, &recvReq));
    irecvReq[i] = std::unique_ptr<CtranMapperRequest>(recvReq);

    // Notify sender that we're ready to receive
    CtranMapperRequest* sendReq = nullptr;
    FB_COMMCHECK(
        comm->ctran_->mapper->isendCtrl(recvbuff, memHdl, srcPeer, &sendReq));
    isendReq[i] = std::unique_ptr<CtranMapperRequest>(sendReq);

    // Initialize notify to receive notification from peer that's sending to us
    notifyVec[i] = std::make_unique<CtranMapperNotify>();
    FB_COMMCHECK(comm->ctran_->mapper->initNotify(
        srcPeer, memHdl, notifyVec.at(i).get()));
  }

  uint64_t firstSend = sendSize / 2;
  uint64_t secondSend = sendSize - firstSend;

  // Wait for first dstPeer
  auto dstPeer = dstAtStep(nRanks, rank, 0);
  FB_COMMCHECK(comm->ctran_->mapper->waitRequest(irecvReq[0].get()));
  timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(dstPeer));

  // Post first half of pipelined message
  CtranMapperRequest* putReqPtr = nullptr;
  FB_COMMCHECK(comm->ctran_->mapper->iput(
      (char*)recvbuff + rank * sendSize,
      (char*)remoteRecvBuffs[0] + rank * sendSize,
      firstSend,
      dstPeer,
      CtranMapperConfig{
          .memHdl_ = memHdl,
          .remoteAccessKey_ = remoteAccessKeys[0],
          .notify_ = true},
      &putReqPtr));
  iputReqA[0] = std::unique_ptr<CtranMapperRequest>(putReqPtr);
  timestamp->putIssued.push_back(CtranMapperTimestampPoint(dstPeer));

  for (size_t i = 0; i < nSteps; i++) {
    auto dstPeer = dstAtStep(nRanks, rank, i);
    auto sends = sendsAtStep(nRanks, rank, i);

    // Post second half of messages for this step
    for (int j = 0; j < sends.size(); j++) {
      size_t putOffset = sends.at(j);
      bool notify = j == sends.size() - 1;
      CtranMapperRequest* putReqPtr = nullptr;

      if (notify) {
        FB_COMMCHECK(comm->ctran_->mapper->iput(
            (char*)recvbuff + putOffset * sendSize + firstSend,
            (char*)remoteRecvBuffs[i] + putOffset * sendSize + firstSend,
            secondSend,
            dstPeer,
            CtranMapperConfig{
                .memHdl_ = memHdl,
                .remoteAccessKey_ = remoteAccessKeys[i],
                .notify_ = notify},
            &putReqPtr));
        iputReqB[i] = std::unique_ptr<CtranMapperRequest>(putReqPtr);
      } else {
        FB_COMMCHECK(comm->ctran_->mapper->iput(
            (char*)recvbuff + putOffset * sendSize + firstSend,
            (char*)remoteRecvBuffs[i] + putOffset * sendSize + firstSend,
            secondSend,
            dstPeer,
            CtranMapperConfig{
                .memHdl_ = memHdl,
                .remoteAccessKey_ = remoteAccessKeys[i],
                .notify_ = notify},
            static_cast<CtranMapperRequest*>(nullptr)));
      }
    }

    // Wait for first step notification from srcPeer, at which point we have
    // first half of all blocks for next step
    FB_COMMCHECK(comm->ctran_->mapper->waitNotify(notifyVec[i].get()));

    // Post first half of messages for next step, if there is a next step
    if (i < (nSteps - 1)) {
      dstPeer = dstAtStep(nRanks, rank, i + 1);
      sends = sendsAtStep(nRanks, rank, i + 1);

      // Block until we have handle for next dstPeer
      FB_COMMCHECK(comm->ctran_->mapper->waitRequest(irecvReq[i + 1].get()));
      timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(dstPeer));

      for (int j = 0; j < sends.size(); j++) {
        size_t putOffset = sends.at(j);
        bool notify = j == sends.size() - 1;
        CtranMapperRequest* putReqPtr = nullptr;

        if (notify) {
          FB_COMMCHECK(comm->ctran_->mapper->iput(
              (char*)recvbuff + putOffset * sendSize,
              (char*)remoteRecvBuffs[i + 1] + putOffset * sendSize,
              firstSend,
              dstPeer,
              CtranMapperConfig{
                  .memHdl_ = memHdl,
                  .remoteAccessKey_ = remoteAccessKeys[i + 1],
                  .notify_ = notify},
              &putReqPtr));
          iputReqA[i + 1] = std::unique_ptr<CtranMapperRequest>(putReqPtr);
        } else {
          FB_COMMCHECK(comm->ctran_->mapper->iput(
              (char*)recvbuff + putOffset * sendSize,
              (char*)remoteRecvBuffs[i + 1] + putOffset * sendSize,
              firstSend,
              dstPeer,
              CtranMapperConfig{
                  .memHdl_ = memHdl,
                  .remoteAccessKey_ = remoteAccessKeys[i + 1],
                  .notify_ = notify},
              static_cast<CtranMapperRequest*>(nullptr)));
        }
        // Capture duration started from first put
        if (j == 0) {
          timestamp->putIssued.push_back(CtranMapperTimestampPoint(dstPeer));
        }
      }
    }

    // Wait for second step notification from srcPeer
    FB_COMMCHECK(comm->ctran_->mapper->waitNotify(notifyVec.at(i).get()));
  }

  // // Wait for signal from all receives
  for (int i = 0; i < nSteps; i++) {
    dstPeer = dstAtStep(nRanks, rank, i);
    FB_COMMCHECK(comm->ctran_->mapper->waitRequest(iputReqA[i].get()));
    FB_COMMCHECK(comm->ctran_->mapper->waitRequest(iputReqB[i].get()));
    timestamp->putComplete.push_back(CtranMapperTimestampPoint(dstPeer));
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

commResult_t ctranAllGatherBrucksFF(
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
