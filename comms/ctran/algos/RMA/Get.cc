// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>

#include "Types.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/tracing/CollTraceWrapper.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/logger/LogUtils.h"

using namespace ctran;
using ::meta::comms::colltrace::CollTraceHandleTriggerState;

extern __global__ void ncclKernelGet(int* flag, CtranAlgoDeviceState* devState);

static commResult_t getImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto comm = op->comm_;
  auto win = op->get.win;
  int peerRank = op->get.peerRank;

  CtranAlgoRMALogger logger("get", op->opCount, peerRank, win, comm);

  // IB backend must be available for current Get implementation
  if (!comm->ctran_->mapper->hasBackend(peerRank, CtranMapperBackend::IB)) {
    CLOGF(ERR, "GET only support IB backend for now, and no IB backend found");
    return commInternalError;
  }

  size_t getSize = op->get.count * commTypeSize(op->get.datatype);
  size_t targetDispNbytes = op->get.targetDisp * commTypeSize(op->get.datatype);
  void* srcPtr = reinterpret_cast<void*>(
      reinterpret_cast<size_t>(win->remWinInfo[peerRank].dataAddr) +
      targetDispNbytes);

  // Get registration handle for local send buffer
  void* localMemHdl = nullptr;
  bool localReg = false;
  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      op->get.recvbuff, getSize, &localMemHdl, &localReg));

  CLOGF_TRACE(
      COLL,
      "getImpl: dstbuf {}, srcbuf {} (base {} + offset {}), size {}",
      op->get.recvbuff,
      srcPtr,
      win->remWinInfo[peerRank].dataAddr,
      targetDispNbytes,
      getSize);

  CtranMapperRequest* req = nullptr;
  FB_COMMCHECK(comm->ctran_->mapper->iget(
      srcPtr,
      op->get.recvbuff,
      getSize,
      peerRank,
      CtranMapperConfig{
          .memHdl_ = localMemHdl,
          .remoteAccessKey_ = win->remWinInfo[peerRank].dataRkey,
      },
      &req));

  auto getReq = std::unique_ptr<CtranMapperRequest>(req);
  FB_COMMCHECK(comm->ctran_->mapper->waitRequest(getReq.get()));

  // Deregister the sendbuffer if it is automatically registered by the
  // mapper->searchRegHandle()
  if (localReg) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(localMemHdl));
  }
  return commSuccess;
}

commResult_t ctranGet(
    void* recvBuff,
    size_t targetDisp,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranWin* win,
    CtranComm* comm,
    cudaStream_t stream) {
  const auto winOpCount = win->updateOpCount(peer);
  const auto getOpCount = win->updateOpCount(peer, window::OpCountType::kGet);
  auto statex = comm->statex_.get();
  CTRAN_RMA_INFO(
      "ctranGet",
      getOpCount,
      winOpCount,
      recvBuff,
      targetDisp,
      count,
      datatype,
      statex->rank(),
      peer,
      win,
      comm,
      stream);

  // Check if the target displacement exceeds the window size
  size_t targetDispNbytes = targetDisp * commTypeSize(datatype);
  size_t countNbytes = count * commTypeSize(datatype);
  if ((targetDispNbytes + countNbytes) > win->dataBytes) {
    CLOGF(
        ERR,
        "Invalid target displacement from {} bytes to {} bytes exceeding the window size {}",
        targetDispNbytes,
        targetDispNbytes + countNbytes,
        win->dataBytes);
    return commInvalidArgument;
  }

  // FIXME: passing also winOpCount to colltrace
  KernelConfig config =
      KernelConfig(KernelConfig::KernelType::GET, stream, "Get", getOpCount);
  config.args.devState_d = comm->ctran_->algo->getDevState();

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.clear();
  config.args.collective.get.peerLocalRank = statex->localRank(peer);

  // Use direct copy if peer is on the same host and has been IPC mapped.
  // Otherwise, do get via network
  if (statex->node(peer) == statex->node() && win->nvlEnabled(peer)) {
    // Single-node direct cudaMemcpy
    if (count > 0) {
      void* srcPtr = reinterpret_cast<void*>(
          reinterpret_cast<size_t>(win->remWinInfo[peer].dataAddr) +
          targetDispNbytes);
      // CollTrace tracing logic for local + no signal case. In this case the
      // get will not trigger gpe->submit, so we need to record manually in the

      // algo. In other cases, this handle would be a no-op and the tracing
      // will take place in the gpe function.
      auto colltraceHandle = meta::comms::colltrace::getCollTraceHandleRMA(
          comm, opGroup, config, true);
      colltraceHandle->trigger(
          CollTraceHandleTriggerState::BeforeEnqueueKernel);

      FB_CUDACHECK(cudaMemcpyAsync(
          recvBuff, srcPtr, countNbytes, cudaMemcpyDefault, stream));

      colltraceHandle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
    }
    return commSuccess;
  } else {
    // Create an op for GPE thread to complete get
    struct OpElem* op =
        new struct OpElem(OpElem::opType::GET, comm, getOpCount);
    op->get.recvbuff = recvBuff;
    op->get.count = count;
    op->get.datatype = datatype;
    op->get.targetDisp = targetDisp;
    op->get.peerRank = peer;
    op->get.win = win;
    opGroup.push_back(std::unique_ptr<struct OpElem>(op));
  }

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      getImpl,
      config,
      reinterpret_cast<void*>(ncclKernelGet)));
  return commSuccess;
}
