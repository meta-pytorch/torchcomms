// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::sendrecv {
KernelConfig::KernelType
getKernelType(bool hasSend, bool hasRecv, bool hasTcpDmRecv);

commResult_t setupGpeOp(
    CtranComm* comm,
    std::vector<OpElem*>& allOps,
    std::vector<OpElem*>& nvlOps,
    std::vector<OpElem*>& sendNvlOps,
    std::vector<OpElem*>& ibOps,
    std::vector<std::unique_ptr<OpElem>>& gpeOpGroup);

commResult_t setupKernelConfig(
    CtranComm* comm,
    const std::vector<OpElem*>& opGroup,
    KernelConfig& config);
} // namespace ctran::sendrecv

inline const std::string sendRecvAlgoName(
    enum NCCL_SENDRECV_ALGO algo,
    const std::vector<OpElem*>& opGroup) {
  if (algo == NCCL_SENDRECV_ALGO::ctran) {
    const auto& firstOp = opGroup.front();
    // Special case for single send/recv
    if (opGroup.size() == 1) {
      if (firstOp->type == OpElem::opType::SEND) {
        return "CtranSend";
      } else if (firstOp->type == OpElem::opType::RECV) {
        return "CtranRecv";
      }
    } else {
      // Leave it as sendrecv if more than one p2p
      return "CtranSendRecv";
    }
  }
  return "Unknown";
}

inline commResult_t selfSendRecvImpl(
    std::vector<OpElem*>& selfSends,
    std::vector<OpElem*>& selfRecvs,
    CtranComm* comm) {
  const auto statex = comm->statex_.get();
  if (selfSends.size() != selfRecvs.size()) {
    CLOGF(
        ERR,
        "Invalid usage: number of self ncclSend ({}) and ncclRecv ({}) does not match on rank {}",
        selfSends.size(),
        selfRecvs.size(),
        statex->rank());
    return commInvalidUsage;
  }

  for (int i = 0; i < selfSends.size(); i++) {
    // cudaMemcpyAsync data from local send buffer to recv buffer.
    // No need track completion in CTran, as it will finish when user
    // synchronizes the stream
    if (selfSends[i]->send.sendbuff != selfRecvs[i]->recv.recvbuff) {
      FB_COMMCHECK(comm->ctran_->mapper->icopy(
          selfRecvs[i]->recv.recvbuff,
          selfSends[i]->send.sendbuff,
          selfSends[i]->send.count * commTypeSize(selfSends[i]->send.datatype),
          selfSends[i]->stream));
    }
  }

  return commSuccess;
}

// Function submitted to GPE thread
inline commResult_t sendRecvImpl(
    const std::vector<std::unique_ptr<OpElem>>& opGroup) {
  std::vector<OpElem*> sendOpGroup, recvOpGroup, allOpGroup;

  auto& firstOp = opGroup.front();
  const auto opCount = firstOp->opCount;
  const auto comm = firstOp->comm_;

  for (auto& op : opGroup) {
    allOpGroup.push_back(op.get());
    if (op->type == OpElem::opType::SEND) {
      sendOpGroup.push_back(op.get());
    } else {
      recvOpGroup.push_back(op.get());
    }
  }

  static const auto myAlgo = NCCL_SENDRECV_ALGO::ctran;
  const std::string algoName = sendRecvAlgoName(myAlgo, allOpGroup);
  CtranAlgoLogger logger(algoName, opCount, comm);

  auto& mapper = comm->ctran_->mapper;
  std::vector<void*> sendMemHdl(sendOpGroup.size());
  std::vector<void*> remoteRecvBuff(sendOpGroup.size());
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKey(
      sendOpGroup.size());
  std::vector<std::unique_ptr<CtranMapperRequest>> sendCtrlReqs(
      sendOpGroup.size());
  std::unordered_map<int, std::unique_ptr<CtranMapperRequest>> putReqs;

  std::vector<void*> recvMemHdl(recvOpGroup.size());
  std::vector<std::unique_ptr<CtranMapperRequest>> recvCtrlReqs(
      recvOpGroup.size());
  std::vector<std::unique_ptr<CtranMapperNotify>> notifyVec(recvOpGroup.size());
  std::vector<int> recvPeerRanks(recvOpGroup.size());
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::make_unique<CtranMapperTimestamp>(algoName);

  std::vector<void*> tmpRegHdls;
  ctran::Profiler* profiler = comm->ctran_->profiler.get();
  if (profiler) {
    profiler->initForEachColl(
        opCount, NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT);
  }

  if (sendOpGroup.size() > 0 || recvOpGroup.size() > 0) {
    std::vector<size_t> sendSizes(sendOpGroup.size(), 0);
    uint64_t peerRank = 0;
    for (auto i = 0; i < sendOpGroup.size(); i++) {
      auto op = sendOpGroup[i];
      size_t sendSize = op->send.count * commTypeSize(op->send.datatype);
      sendSizes[i] = sendSize;
      peerRank = op->send.peerRank;
    }
    std::vector<size_t> recvSizes(recvOpGroup.size(), 0);
    for (auto i = 0; i < recvOpGroup.size(); i++) {
      auto op = recvOpGroup[i];
      size_t recvSize = op->recv.count * commTypeSize(op->recv.datatype);
      recvSizes[i] = recvSize;
      peerRank = op->recv.peerRank;
    }
    CtranMapperContext context(algoName, sendSizes, recvSizes);
    context.unpackPoolId = opGroup.front()->unpackPoolId;
    comm->ctran_->mapper->setContext(std::move(context));

    CTRAN_PROFILER_IF(profiler, {
      auto& algoContext = profiler->algoContext;
      algoContext.peerRank = peerRank;
      algoContext.algorithmName = algoName;
      algoContext.sendContext.messageSizes = folly::join(',', sendSizes);
      algoContext.recvContext.messageSizes = folly::join(',', recvSizes);
    });
  }

  // Issue control messages for send operations
  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::BUF_REG));
  for (auto i = 0; i < sendOpGroup.size(); ++i) {
    auto op = sendOpGroup[i];
    size_t sendBytes = op->send.count * commTypeSize(op->send.datatype);
    bool localReg = false;

    FB_COMMCHECK(mapper->searchRegHandle(
        op->send.sendbuff, sendBytes, &sendMemHdl[i], &localReg));
    if (localReg) {
      tmpRegHdls.push_back(sendMemHdl[i]);
    }
  }
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::BUF_REG));

  CTRAN_PROFILER_CONDITION_IF(
      profiler,
      sendOpGroup.size() > 0,
      profiler->startEvent(ctran::ProfilerEvent::ALGO_CTRL));
  for (auto i = 0; i < sendOpGroup.size(); ++i) {
    auto op = sendOpGroup[i];
    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(mapper->irecvCtrl(
        &remoteRecvBuff[i], &remoteAccessKey[i], op->send.peerRank, &req));
    sendCtrlReqs[i] = std::unique_ptr<CtranMapperRequest>(req);
  }

  // Issue control messages for recv operations
  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::BUF_REG));
  for (auto i = 0; i < recvOpGroup.size(); i++) {
    auto op = recvOpGroup[i];
    size_t recvBytes = op->recv.count * commTypeSize(op->recv.datatype);
    bool localReg = false;

    FB_COMMCHECK(mapper->searchRegHandle(
        op->recv.recvbuff, recvBytes, &recvMemHdl[i], &localReg));

    if (localReg) {
      tmpRegHdls.push_back(recvMemHdl[i]);
    }
  }
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::BUF_REG));

  for (auto i = 0; i < recvOpGroup.size(); i++) {
    auto op = recvOpGroup[i];
    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(mapper->isendCtrl(
        op->recv.recvbuff, recvMemHdl[i], op->recv.peerRank, &req));
    recvCtrlReqs[i] = std::unique_ptr<CtranMapperRequest>(req);
    recvPeerRanks[i] = op->recv.peerRank;

    // Initialize notify flag to receive from peer
    notifyVec[i] = std::make_unique<CtranMapperNotify>();
    FB_COMMCHECK(mapper->initNotify(
        op->recv.peerRank, recvMemHdl[i], op->recv.kElem, notifyVec[i].get()));
  }

  // As we recv control msgs, issue PUT operations
  bool isIssuedFirst = false;
  while (putReqs.size() < sendOpGroup.size() && !comm->testAbort()) {
    for (auto i = 0; i < sendOpGroup.size(); i++) {
      // Already issued PUT
      if (putReqs.find(i) != putReqs.end()) {
        continue;
      }

      bool isComplete = false;
      FB_COMMCHECK(mapper->testRequest(sendCtrlReqs[i].get(), &isComplete));
      if (isComplete) {
        auto op = sendOpGroup[i];
        size_t sendSize = op->send.count * commTypeSize(op->send.datatype);

        timestamp->recvCtrl.push_back(
            CtranMapperTimestampPoint(op->send.peerRank));
        // iput internally dispatches to either network put or NVL copy
        CtranMapperRequest* req = nullptr;
        CTRAN_PROFILER_CONDITION_IF(profiler, !isIssuedFirst, {
          profiler->startEvent(ctran::ProfilerEvent::ALGO_DATA);
          isIssuedFirst = true;
        });

        // ALGO_CTRL records duration from first irecvCtrl to the completion of
        // last irecvCtrl, i.e., when issuing the last put
        CTRAN_PROFILER_CONDITION_IF(
            profiler,
            putReqs.size() == sendOpGroup.size() - 1,
            profiler->endEvent(ctran::ProfilerEvent::ALGO_CTRL));
        FB_COMMCHECK(mapper->iput(
            op->send.sendbuff,
            remoteRecvBuff[i],
            sendSize,
            op->send.peerRank,
            CtranMapperConfig{
                .memHdl_ = sendMemHdl[i],
                .remoteAccessKey_ = remoteAccessKey[i],
                .notify_ = true,
                .kernElem_ = op->send.kElem},
            &req));
        putReqs[i] = std::unique_ptr<CtranMapperRequest>(req);
        timestamp->putIssued.push_back(
            CtranMapperTimestampPoint(op->send.peerRank));
      }
    }
  }

  // Wait for all PUT messages to complete
  for (auto i = 0; i < sendOpGroup.size(); i++) {
    auto op = sendOpGroup[i];
    FB_COMMCHECK(mapper->waitRequest(putReqs[i].get()));
    timestamp->putComplete.push_back(
        CtranMapperTimestampPoint(op->send.peerRank));
  }
  CTRAN_PROFILER_CONDITION_IF(
      profiler,
      sendOpGroup.size() > 0,
      profiler->endEvent(ctran::ProfilerEvent::ALGO_DATA));

  // Wait for all control messages and notifications to complete
  for (auto i = 0; i < recvOpGroup.size(); i++) {
    FB_COMMCHECK(mapper->waitRequest(recvCtrlReqs[i].get()));
    FB_COMMCHECK(mapper->waitNotify(notifyVec[i].get()));
  }

  // Deregister temporary registrations
  for (auto hdl : tmpRegHdls) {
    FB_COMMCHECK(mapper->deregDynamic(hdl));
  }

  CTRAN_PROFILER_IF(profiler, { profiler->reportToScuba(); });

  mapper->timestamps.push_back(std::move(timestamp));
  mapper->reportProfiling();

  return commSuccess;
}
