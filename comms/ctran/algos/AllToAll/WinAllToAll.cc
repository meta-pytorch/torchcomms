// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/algos/AllToAll/HostTypes.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/window/CtranWin.h"

using ctran::alltoallp::AlgoImpl;

#define CHECK_VALID_PREQ(pReq)                                        \
  do {                                                                \
    if (!(pReq)) {                                                    \
      FB_ERRORRETURN(                                                 \
          commInvalidArgument,                                        \
          "Null PersistentRequest passed to {}",                      \
          __func__);                                                  \
    }                                                                 \
    if (pReq->type != CtranPersistentRequest::Type::ALLTOALL_P_WIN) { \
      FB_ERRORRETURN(                                                 \
          commInvalidArgument,                                        \
          "Unexpected PersistentRequest type {} called into {}",      \
          pReq->type,                                                 \
          __func__);                                                  \
    }                                                                 \
  } while (0)

namespace ctran {

commResult_t AllToAllWinInit(
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
      CtranPersistentRequest::Type::ALLTOALL_P_WIN, comm, stream);
  request->algo = algo.release();

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "AllToAllWinInit: rank {} initialized request {}, "
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

commResult_t AllToAllWinExec(
    const void* sendbuff,
    const size_t count,
    CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);
  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  CLOGF_TRACE(
      COLL,
      "AllToAllWinExec: rank {} started executing request {}, sendbuff {} recvbuff {} count {}",
      request->comm_->statex_->rank(),
      (void*)request,
      sendbuff,
      (void*)algo->pArgs.recvbuff,
      count);
  return algo->exec(sendbuff, count);
}

commResult_t AllToAllWinDestroy(CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  delete algo;
  request->algo = nullptr;

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "AllToAllWinDestroy: rank {} destroyed request {}",
      request->comm_->statex_->rank(),
      (void*)request);

  return commSuccess;
}

} // namespace ctran
