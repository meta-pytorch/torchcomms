// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/cvars/nccl_cvars.h"

using ctran::allgatherp::AlgoImpl;

#define CHECK_VALID_PREQ(pReq)                                         \
  do {                                                                 \
    if (!(pReq)) {                                                     \
      FB_ERRORRETURN(                                                  \
          commInvalidArgument,                                         \
          "Null PersistentRequest passed to {}",                       \
          __func__);                                                   \
    }                                                                  \
    if (pReq->type != CtranPersistentRequest::Type::ALLGATHER_P_WIN) { \
      FB_ERRORRETURN(                                                  \
          commInvalidArgument,                                         \
          "Unexpected PersistentRequest type {} called into {}",       \
          pReq->type,                                                  \
          __func__);                                                   \
    }                                                                  \
  } while (0)

namespace ctran {

commResult_t allGatherWinInit(
    CtranWin* win,
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

  // Populate pArgs from window remote info
  algo->pArgs.recvbuff = win->winDataPtr;
  algo->pArgs.recvHdl = win->dataRegHdl;
  algo->pArgs.remoteRecvBuffs.resize(nRanks);
  algo->pArgs.remoteAccessKeys.resize(nRanks);
  for (int r = 0; r < nRanks; r++) {
    algo->pArgs.remoteRecvBuffs[r] = win->remWinInfo[r].dataAddr;
    algo->pArgs.remoteAccessKeys[r] = win->remWinInfo[r].dataRkey;
  }
  // Window already exchanged remote info, mark as initialized
  algo->pArgs.initialized.store(true);

  FB_COMMCHECK(algo->initResources());

  request = new CtranPersistentRequest(
      CtranPersistentRequest::Type::ALLGATHER_P_WIN, comm, stream);
  request->algo = algo.release();

  return commSuccess;
}

commResult_t allGatherWinExec(
    const void* sendbuff,
    const size_t count,
    commDataType_t datatype,
    CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);

  switch (NCCL_ALLGATHER_P_ALGO) {
    case NCCL_ALLGATHER_P_ALGO::ctdirect:
      return algo->execDirect(sendbuff, count, datatype);
    case NCCL_ALLGATHER_P_ALGO::ctpipeline:
      return algo->execPipeline(sendbuff, count, datatype);
    default:
      return ErrorStackTraceUtil::log(commInternalError);
  }
}

commResult_t allGatherWinDestroy(CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  if (!algo) {
    return commSuccess;
  }
  FB_COMMCHECK(algo->destroy());
  delete algo;
  request->algo = nullptr;

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "allGatherWinDestroy: rank {} destroyed request {}",
      request->comm_->statex_->rank(),
      (void*)request);

  return commSuccess;
}

} // namespace ctran
