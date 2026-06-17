// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/Broadcast/BroadcastPImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/window/CtranWin.h"

using ctran::broadcastp::AlgoImpl;

#define CHECK_VALID_PREQ(pReq)                                         \
  do {                                                                 \
    if (!(pReq)) {                                                     \
      FB_ERRORRETURN(                                                  \
          commInvalidArgument,                                         \
          "Null PersistentRequest passed to {}",                       \
          __func__);                                                   \
    }                                                                  \
    if (pReq->type != CtranPersistentRequest::Type::BROADCAST_P_WIN) { \
      FB_ERRORRETURN(                                                  \
          commInvalidArgument,                                         \
          "Unexpected PersistentRequest type {} called into {}",       \
          pReq->type,                                                  \
          __func__);                                                   \
    }                                                                  \
  } while (0)

namespace ctran {

commResult_t BroadcastWinInit(
    CtranWin* win,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request) {
  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();

  if (win->remWinInfo.empty() ||
      static_cast<int>(win->remWinInfo.size()) != nRanks) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Window remWinInfo not populated (size {}). "
        "Was exchange() called?",
        win->remWinInfo.size());
  }

  auto algo = std::make_unique<AlgoImpl>(comm, stream);

  algo->pArgs.recvbuff = win->winDataPtr;
  algo->pArgs.recvHdl = win->dataRegHdl;
  algo->pArgs.maxRecvCount = win->dataBytes / commTypeSize(datatype);
  algo->pArgs.datatype = datatype;
  algo->pArgs.skipCtrlMsg = true;
  algo->pArgs.remoteRecvBuffs.resize(nRanks);
  algo->pArgs.remoteAccessKeys.resize(nRanks);
  for (int r = 0; r < nRanks; r++) {
    algo->pArgs.remoteRecvBuffs[r] = win->remWinInfo[r].dataAddr;
    algo->pArgs.remoteAccessKeys[r] = win->remWinInfo[r].dataRkey;
  }

  request = new CtranPersistentRequest(
      CtranPersistentRequest::Type::BROADCAST_P_WIN, comm, stream);
  request->algo = algo.release();

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "BroadcastWinInit: rank {} initialized request {}, "
      "recvbuff {}, comm {} commHash {:x} [nranks={}] stream={}",
      statex->rank(),
      (void*)request,
      win->winDataPtr,
      (void*)comm,
      statex->commHash(),
      nRanks,
      (void*)stream);

  return commSuccess;
}

commResult_t BroadcastWinExec(
    const void* sendbuff,
    const size_t count,
    int root,
    CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);
  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  CLOGF_TRACE(
      COLL,
      "BroadcastWinExec: rank {} executing request {}, sendbuff {} recvbuff {} count {} root {}",
      request->comm_->statex_->rank(),
      (void*)request,
      sendbuff,
      (void*)algo->pArgs.recvbuff,
      count,
      root);
  return algo->exec(sendbuff, count, root);
}

commResult_t BroadcastWinDestroy(CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  delete algo;
  request->algo = nullptr;

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "BroadcastWinDestroy: rank {} destroyed request {}",
      request->comm_->statex_->rank(),
      (void*)request);

  return commSuccess;
}

} // namespace ctran
