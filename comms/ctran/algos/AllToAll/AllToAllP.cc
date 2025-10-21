// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/utils/logger/LogUtils.h"

using ctran::alltoallp::AlgoImpl;

#define CHECK_VALID_PREQ(pReq)                                    \
  do {                                                            \
    if (pReq->type != CtranPersistentRequest::Type::ALLTOALL_P) { \
      FB_ERRORRETURN(                                             \
          commInvalidArgument,                                    \
          "Unexpected PersistentRequest type {} called into {}",  \
          pReq->type,                                             \
          __func__);                                              \
    }                                                             \
  } while (0)

namespace ctran {
commResult_t AllToAllPInit(
    void* recvbuff,
    const size_t maxRecvCount,
    const meta::comms::Hints& hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request) {
  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();

  SetCudaDevRAII setCudaDev(statex->cudaDev());
  size_t size = maxRecvCount * commTypeSize(datatype);
  void* regHdl{nullptr};
  bool localReg = false;
  AlgoImpl* algo = new AlgoImpl(comm, stream);
  if (!algo) {
    return commSystemError;
  }

  auto guard = folly::makeGuard([algo] {
    if (algo) {
      delete algo;
    }
  });
  // TODO: Pass-in a flag searchOnly to avoid dynamic register instead of reg
  // then deregister.
  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      recvbuff, size, &regHdl, &localReg));
  if (localReg) {
    comm->ctran_->mapper->deregDynamic(regHdl);
    CLOGF(
        ERR,
        "recvbuff is not registered. Pointer: {} length: {}",
        recvbuff,
        size);
    return commInternalError;
  }

  std::string skip_ctrl_msg;
  hints.get("ncclx_alltoallp_skip_ctrl_msg_exchange", skip_ctrl_msg);
  algo->pArgs = {
      .recvbuff = recvbuff,
      .recvHdl = regHdl,
      .maxRecvCount = maxRecvCount,
      .datatype = datatype,
      .skipCtrlMsg = (skip_ctrl_msg == "true"),
  };
  FB_COMMCHECK(algo->init());
  request = new CtranPersistentRequest(
      CtranPersistentRequest::Type::ALLTOALL_P, comm, stream);
  if (!request) {
    return commSystemError;
  }
  request->algo = algo;

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "AllToAllPInit: rank {} initialized request {}, recvbuff {} recvHdl {}, comm {} commHash {:x} commDesc {} [nranks={}, localRanks={}] stream={}",
      statex->rank(),
      (void*)request,
      (void*)recvbuff,
      (void*)regHdl,
      (void*)comm,
      statex->commHash(),
      statex->commDesc(),
      nRanks,
      statex->nLocalRanks(),
      (void*)stream);

  guard.dismiss();
  return commSuccess;
}

// active buffer that user is using for this run
commResult_t AllToAllPExec(
    const void* sendbuff,
    const size_t count,
    CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);
  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  CLOGF_TRACE(
      COLL,
      "AllToAllPExec: rank {} started executing request {}, sendbuff {} recvbuff {} count {} ",
      request->comm_->statex_->rank(),
      (void*)request,
      sendbuff,
      (void*)algo->pArgs.recvbuff,
      count);
  return algo->exec(sendbuff, count);
}

commResult_t AllToAllPDestroy(CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  // No need to dereg handles now since user should call explicit commDeregister
  // before buffer is freed
  auto algo = reinterpret_cast<AlgoImpl*>(request->algo);
  delete algo;
  request->algo = nullptr;

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "AllToAllPDestroy: rank {} destroyed request {}",
      request->comm_->statex_->rank(),
      (void*)request);

  return commSuccess;
}

bool AllToAllPSupport(CtranComm* comm) {
  bool ctranSupport = false;
  const auto statex = comm->statex_.get();
  if (ctranInitialized(comm)) {
    ctranSupport = true;
    for (int rank = 0; rank < statex->nRanks(); rank++) {
      if (comm->ctran_->mapper->getBackend(rank) == CtranMapperBackend::UNSET) {
        ctranSupport = false;
        break;
      }
    }
  } else {
    return false;
  }

  return ctranSupport;
}
} // namespace ctran
