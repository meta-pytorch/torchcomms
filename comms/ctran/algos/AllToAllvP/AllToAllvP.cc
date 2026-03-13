// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAllvP/AllToAllvPImpl.h"
#include "comms/ctran/algos/AllToAllvP/Types.h"

#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/logger/LogUtils.h"

using ctran::alltoallvp::AlgoImpl;

#define CHECK_VALID_PREQ(pReq)                                     \
  do {                                                             \
    if (pReq->type != CtranPersistentRequest::Type::ALLTOALLV_P) { \
      FB_ERRORRETURN(                                              \
          commInvalidArgument,                                     \
          "Unexpected PersistentRequest type {} called into {}",   \
          pReq->type,                                              \
          __func__);                                               \
    }                                                              \
  } while (0)

namespace ctran {

bool allToAllvPSupport(CtranComm* comm) {
  bool ctranSupport = false;
  const auto statex = comm->statex_.get();
  if (ctranInitialized(comm)) {
    ctranSupport = true;
    for (auto rank = 0; rank < statex->nRanks(); rank++) {
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

commResult_t allToAllvPInit(
    void* recvbuff,
    const size_t maxRecvCount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request) {
  const auto statex = comm->statex_.get();
  SetCudaDevRAII setCudaDev(statex->cudaDev());
  AlgoImpl* algo = new AlgoImpl(comm, stream);
  if (!algo) {
    return commSystemError;
  }

  auto guard = folly::makeGuard([algo] {
    if (algo) {
      delete algo;
    }
  });

  // Search recvbuff handle
  auto mapper = comm->ctran_->mapper.get();
  size_t maxRecvSize = maxRecvCount * commTypeSize(datatype);

  void* recvHdl{nullptr};
  bool localRegRecv = false;
  FB_COMMCHECK(
      mapper->searchRegHandle(recvbuff, maxRecvSize, &recvHdl, &localRegRecv));
  if (localRegRecv) {
    FB_COMMCHECK(mapper->deregDynamic(recvHdl));
    CLOGF(
        ERR,
        "recvbuff is not registered. Pointer: {} length: {}",
        (void*)recvbuff,
        maxRecvSize);
    return commInternalError;
  }

  // Set up persistent arguments
  algo->pArgs.recvHdl = recvHdl;
  algo->pArgs.recvbuff = recvbuff;
  algo->pArgs.maxRecvCount = maxRecvCount;
  algo->pArgs.datatype = datatype;
  algo->pArgs.initialized.store(false);

  // Schedule exchangeMemHdl on GPE thread
  FB_COMMCHECK(algo->init());

  // Create persistent request
  request = new CtranPersistentRequest(
      CtranPersistentRequest::Type::ALLTOALLV_P, comm, stream);
  if (!request) {
    return commSystemError;
  }
  request->algo = algo;

  guard.dismiss();
  return commSuccess;
}

commResult_t allToAllvPExec(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    const size_t recvcounts[],
    const size_t rdispls[],
    commDataType_t datatype,
    CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  auto algo = reinterpret_cast<AlgoImpl*>(request->algo);
  const auto nRanks = request->comm_->statex_->nRanks();
  size_t totalRecvCount = 0;
  for (int i = 0; i < nRanks; i++) {
    totalRecvCount += recvcounts[i];
  }
  if (totalRecvCount * commTypeSize(datatype) >
      algo->pArgs.maxRecvCount * commTypeSize(algo->pArgs.datatype)) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "AllToAllvP invalid recvcounts {} * sizeof datatype {} exceeds maxRecvCount {} * sizeof datatype {}.",
        totalRecvCount,
        datatype,
        algo->pArgs.maxRecvCount,
        algo->pArgs.datatype);
  }

  switch (NCCL_ALLTOALLV_P_ALGO) {
    case NCCL_ALLTOALLV_P_ALGO::ctran:
      return algo->exec(
          sendbuff, sendcounts, sdispls, recvcounts, rdispls, datatype);
    default:
      return ErrorStackTraceUtil::log(commInternalError);
  }
}

commResult_t allToAllvPDestroy(CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  // No need to dereg handles now since user should call explicit
  // commDeregister before buffer is freed
  auto algo = reinterpret_cast<AlgoImpl*>(request->algo);
  delete algo;
  request->algo = nullptr;

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "allToAllvPDestroy: rank {} destroyed request {}",
      request->comm_->statex_->rank(),
      (void*)request);

  return commSuccess;
}
} // namespace ctran
