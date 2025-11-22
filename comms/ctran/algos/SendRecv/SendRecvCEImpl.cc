// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include "comms/ctran/algos/SendRecv/SendRecvCEImpl.h"

#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace {

// The result of the exchange: send op's op->send.recvbuff and
// op->send.remoteAccessKey are updated to the remote buffer address and remote
// access key.
// NOTE: users should pre-register the send/recv buffers before calling this
// function
commResult_t exchangeSendRecvHandles(
    const std::vector<std::unique_ptr<OpElem>>& opGroup) {
  if (opGroup.empty()) {
    return commSuccess;
  }
  auto& firstOp = opGroup.front();
  const auto comm = firstOp->comm_;
  std::vector<OpElem*> sendOpGroup, recvOpGroup;

  for (auto& op : opGroup) {
    if (op->type == OpElem::opType::SEND) {
      sendOpGroup.push_back(op.get());
    } else {
      recvOpGroup.push_back(op.get());
    }
  }

  auto& mapper = comm->ctran_->mapper;
  std::vector<void*> remoteRecvBuff(sendOpGroup.size());
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKey(
      sendOpGroup.size());
  std::vector<std::unique_ptr<CtranMapperRequest>> sendCtrlReqs(
      sendOpGroup.size());

  std::vector<void*> recvMemHdl(recvOpGroup.size());
  std::vector<std::unique_ptr<CtranMapperRequest>> recvCtrlReqs(
      recvOpGroup.size());

  std::vector<void*> tmpRegHdls;

  for (auto i = 0; i < sendOpGroup.size(); i++) {
    auto& op = sendOpGroup[i];

    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(mapper->irecvCtrl(
        &remoteRecvBuff[i], &remoteAccessKey[i], op->send.peerRank, &req));
    sendCtrlReqs[i] = std::unique_ptr<CtranMapperRequest>(req);
  }

  for (auto i = 0; i < recvOpGroup.size(); i++) {
    auto& op = recvOpGroup[i];
    size_t recvSize = op->recv.count * commTypeSize(op->recv.datatype);
    bool localReg = false;

    FB_COMMCHECK(mapper->searchRegHandle(
        op->recv.recvbuff, recvSize, &recvMemHdl[i], &localReg));

    if (localReg) {
      tmpRegHdls.push_back(recvMemHdl[i]);
    }

    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(mapper->isendCtrl(
        op->recv.recvbuff, recvMemHdl[i], op->recv.peerRank, &req));
    recvCtrlReqs[i] = std::unique_ptr<CtranMapperRequest>(req);
  }

  for (auto i = 0; i < recvOpGroup.size(); i++) {
    FB_COMMCHECK(mapper->waitRequest(recvCtrlReqs[i].get()));
  }

  for (auto i = 0; i < sendOpGroup.size(); i++) {
    FB_COMMCHECK(mapper->waitRequest(sendCtrlReqs[i].get()));
    auto& op = sendOpGroup[i];
    op->send.recvbuff->store(remoteRecvBuff[i]);
    if (remoteAccessKey[i].backend == CtranMapperBackend::NVL) {
      op->send.remoteAccessKey = remoteAccessKey[i];
    } else {
      CLOGF(
          ERR,
          "Invalid usage: remote access key exchanged in exchangeSendRecvHandles is not NVL. ");
      return commInvalidUsage;
    }
  }

  if (!tmpRegHdls.empty()) {
    CLOGF(ERR, "Invalid usage: buffers are not pre-regiestoered.");
    for (auto hdl : tmpRegHdls) {
      FB_COMMCHECK(mapper->deregDynamic(hdl));
    }
    return commInvalidUsage;
  }

  return commSuccess;
}

commResult_t submitHandleExchangeToGpe(const std::vector<OpElem*>& ops) {
  if (ops.empty()) {
    return commSuccess;
  }
  std::vector<std::unique_ptr<struct OpElem>> exchangeOpGroup;
  CtranComm* comm = ops.front()->comm_;
  cudaStream_t stream = ops.front()->stream;
  auto opCount = ops.front()->opCount;
  auto mapper = comm->ctran_->mapper.get();
  for (const auto& op : ops) {
    auto exchangeOp =
        std::make_unique<OpElem>(op->type, stream, comm, op->opCount);
    // only exchange handles for NVL backend
    if (op->type == OpElem::opType::SEND &&
        mapper->getBackend(op->send.peerRank) == CtranMapperBackend::NVL) {
      exchangeOp->send.sendbuff = op->send.sendbuff;
      exchangeOp->send.count = op->send.count;
      exchangeOp->send.datatype = op->send.datatype;
      exchangeOp->send.peerRank = op->send.peerRank;
      exchangeOp->send.recvbuff = op->send.recvbuff;
      exchangeOpGroup.push_back(std::move(exchangeOp));
    } else if (
        op->type == OpElem::opType::RECV &&
        mapper->getBackend(op->recv.peerRank) == CtranMapperBackend::NVL) {
      exchangeOp->recv.recvbuff = op->recv.recvbuff;
      exchangeOp->recv.count = op->recv.count;
      exchangeOp->recv.datatype = op->recv.datatype;
      exchangeOp->recv.peerRank = op->recv.peerRank;
      exchangeOpGroup.push_back(std::move(exchangeOp));
    }
  }

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::SENDRECV,
      stream,
      "CtranSendHanldeExchange",
      opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = comm->ctran_->algo->getDevState();

  FB_COMMCHECK(comm->ctran_->gpe->submitHost(
      std::move(exchangeOpGroup),
      exchangeSendRecvHandles,
      config,
      nullptr /* exReq */));

  return commSuccess;
}

commResult_t sendRecvCopyEngineImpl(const std::vector<OpElem*>& sendNvlOps) {
  if (sendNvlOps.empty()) {
    return commSuccess;
  }

  auto& firstOp = sendNvlOps.front();
  const auto comm = firstOp->comm_;

  for (int i = 0; i < sendNvlOps.size(); i++) {
    // cudaMemcpyAsync data from send buffer to recv buffer.
    auto& op = sendNvlOps[i];
    void* recvBuff = op->send.recvbuff->load();
    FB_COMMCHECK(comm->ctran_->mapper->icopy(
        recvBuff,
        op->send.sendbuff,
        op->send.count * commTypeSize(op->send.datatype),
        op->stream));
  }

  return commSuccess;
}

} // namespace

namespace ctran::sendrecv {

commResult_t launchSendRecvCopyEngine(
    std::vector<OpElem*>& nvlOps,
    std::vector<OpElem*>& sendNvlOps,
    CtranComm* comm) {
  FB_COMMCHECK(submitHandleExchangeToGpe(nvlOps));
  for (auto& op : sendNvlOps) {
    while (!op->send.recvbuff->load()) {
      FB_COMMCHECK(comm->getAsyncResult());
    }
  }
  sendRecvCopyEngineImpl(sendNvlOps);
  for (auto& op : sendNvlOps) {
    if (op->send.recvbuff) {
      delete op->send.recvbuff;
    }
  }

  nvlOps.clear();
  sendNvlOps.clear();
  return commSuccess;
}

} // namespace ctran::sendrecv
