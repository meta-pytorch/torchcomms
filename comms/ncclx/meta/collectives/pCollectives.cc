// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>

#include "comm.h"
#include "nccl.h"

#include "comms/ctran/Ctran.h"

#include "meta/NcclxChecks.h"
#include "meta/wrapper/MetaFactory.h"

namespace ncclx {
__attribute__((visibility("default"))) ncclResult_t allGatherInit(
    void* recvbuff,
    const size_t maxRecvCount,
    const Hints& hints,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    void** request) {
  if (!ctran::allGatherPSupport(comm->ctranComm_.get())) {
    NCCLX_ERRORTHROW(
        commInvalidUsage,
        "Persistent AllGather is not supported. Check whether CTRAN is enabled.");
  }

  SetCudaDevRAII setCudaDev(comm->cudaDev);
  CtranPersistentRequest* pReq = nullptr;
  NCCLCHECK(metaCommToNccl(
      ctran::allGatherPInit(
          recvbuff,
          maxRecvCount,
          ncclToMetaComm(hints),
          ncclToMetaComm(datatype),
          comm->ctranComm_.get(),
          stream,
          pReq)));
  *request = reinterpret_cast<void*>(pReq);

  return ncclSuccess;
}

#define CHECK_VALID_CTRAN(comm)                                             \
  if (!ctranInitialized(comm)) {                                            \
    ERR(ncclInvalidUsage,                                                   \
        "CTRAN must be enabled and initialized for persistent collective"); \
    return ncclInvalidUsage;                                                \
  }

#define CHECK_PREQ_TYPE(pReq, type)                                \
  if (pReq->type != type) {                                        \
    ERR(ncclInvalidArgument,                                       \
        "%s requires persistent request type %d, but received %d", \
        __func__,                                                  \
        static_cast<int>(type),                                    \
        static_cast<int>(pReq->type));                             \
    return ncclInvalidArgument;                                    \
  }

#define GET_VALID_PREQ_OR_ERRRETURN(req, pReq)                    \
  do {                                                            \
    if (request == nullptr) {                                     \
      ERR(ncclInvalidArgument,                                    \
          "%s received invalid nullptr request",                  \
          __func__);                                              \
      return ncclInvalidArgument;                                 \
    }                                                             \
    *(pReq) = reinterpret_cast<CtranPersistentRequest*>(request); \
  } while (0)

__attribute__((visibility("default"))) ncclResult_t allGatherExec(
    const void* sendbuff,
    const size_t count,
    const ncclDataType_t datatype,
    void* request) {
  CtranPersistentRequest* pReq = nullptr;
  GET_VALID_PREQ_OR_ERRRETURN(request, &pReq);
  CHECK_PREQ_TYPE(pReq, CtranPersistentRequest::Type::ALLGATHER_P);
  CHECK_VALID_CTRAN(pReq->comm_);

  return metaCommToNccl(
      ::ctran::allGatherPExec(sendbuff, count, ncclToMetaComm(datatype), pReq));
}

__attribute__((visibility("default"))) ncclResult_t allToAllvDedupInit(
    const size_t totalNumSendBlocks,
    const size_t blockCount,
    const size_t blockNumRecvBuckets,
    const int numRecvBuckets,
    const ncclx::Hints& hints,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    void** request) {
  WARN("allToAllvDedupInit: experimental API moved to comms/experiments/algos");
  return ncclInvalidUsage;
}

__attribute__((visibility("default"))) ncclResult_t allToAllvDedupExec(
    const void* sendBuff,
    const int sendIdx[],
    const int fwdIdx[],
    const int recvIdx[],
    void* recvBuff,
    int recvBlockIds[],
    void* request) {
  WARN("allToAllvDedupExec: experimental API moved to comms/experiments/algos");
  return ncclInvalidUsage;
}

__attribute__((visibility("default"))) ncclResult_t pExec(void* request) {
  CtranPersistentRequest* pReq = nullptr;
  GET_VALID_PREQ_OR_ERRRETURN(request, &pReq);
  NCCLX_LOG(
      INFO,
      "Executing persistent request {} comm {}",
      (void*)pReq,
      (void*)pReq->comm_);

  if (!ctranInitialized(pReq->comm_)) {
    ERR(ncclInvalidUsage,
        "CTRAN must be enabled and initialized for persistent collective");
    return ncclInvalidUsage;
  }

  switch (pReq->type) {
    default:
      ERR(ncclInvalidArgument,
          "Persistent request %p has unknown op type %d",
          (void*)pReq,
          static_cast<int>(pReq->type));
      return ncclInvalidArgument;
  }
}

__attribute__((visibility("default"))) ncclResult_t AllToAllInit(
    void* recvbuff,
    const size_t maxRecvCount,
    const Hints& hints,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    void*& request) {
  if (!ctran::AllToAllPSupport(comm->ctranComm_.get())) {
    ERR(ncclInvalidUsage,
        "Persistent AllToAll is not supported. Check whether CTRAN is enabled.");
    return ncclInvalidUsage;
  }

  SetCudaDevRAII setCudaDev(comm->cudaDev);
  CtranPersistentRequest* pReq = nullptr;
  NCCLCHECK(metaCommToNccl(
      ctran::AllToAllPInit(
          recvbuff,
          maxRecvCount,
          ncclToMetaComm(hints),
          ncclToMetaComm(datatype),
          comm->ctranComm_.get(),
          stream,
          pReq)));
  request = reinterpret_cast<void*>(pReq);

  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclResult_t
AllToAllExec(const void* sendbuff, const size_t count, void* request) {
  if (request == nullptr) {
    ERR(ncclInvalidUsage,
        "request shouldn't be nullptr for persistent collective");
    return ncclInvalidUsage;
  }
  CtranPersistentRequest* pReq =
      reinterpret_cast<CtranPersistentRequest*>(request);

  if (!ctranInitialized(pReq->comm_)) {
    ERR(ncclInvalidUsage,
        "CTRAN must be enabled and initialized for persistent collective");
    return ncclInvalidUsage;
  }

  if (pReq->type != CtranPersistentRequest::Type::ALLTOALL_P) {
    ERR(ncclInvalidArgument,
        "Unexpected PersistentRequest type %d called into AllToAllExec",
        static_cast<int>(pReq->type));
    return ncclInvalidArgument;
  }

  return metaCommToNccl(ctran::AllToAllPExec(sendbuff, count, pReq));
}

__attribute__((visibility("default"))) ncclResult_t pFree(void* request) {
  CtranPersistentRequest* pReq = nullptr;
  GET_VALID_PREQ_OR_ERRRETURN(request, &pReq);

  // Ensure pReq is freed no matter destroy fails or not.
  auto reqGuard = folly::makeGuard([pReq] { delete pReq; });

  switch (pReq->type) {
    case CtranPersistentRequest::Type::ALLGATHER_P:
      NCCLCHECK(metaCommToNccl(ctran::allGatherPDestroy(pReq)));
      break;
    case CtranPersistentRequest::Type::ALLTOALL_P:
      NCCLCHECK(metaCommToNccl(ctran::AllToAllPDestroy(pReq)));
      break;
    case CtranPersistentRequest::Type::ALLTOALLV_DEDUP:
      WARN(
          "allToAllvDedupDestroy: experimental API moved to comms/experiments/algos");
      return ncclInvalidUsage;
    default:
      ERR(ncclInvalidArgument,
          "Persistent request %p has unknown op type %d",
          (void*)pReq,
          static_cast<int>(pReq->type));
      return ncclInvalidArgument;
  }

  return ncclSuccess;
}

} // namespace ncclx
