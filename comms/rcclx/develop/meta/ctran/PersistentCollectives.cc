// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comm.h"
#include "rccl.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/utils/Checks.h"

#include "MetaFactory.h"

namespace ncclx {
__attribute__((visibility("default"))) ncclResult_t allGatherInit(
    void* recvbuff,
    const size_t maxRecvCount,
    const Hints& hints,
    ncclDataType_t datatype,
    ncclComm_t comm,
    hipStream_t stream,
    void** request) {
  if (!ctran::allGatherPSupport(comm->ctranComm_.get())) {
    FB_ERRORTHROW(
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
    CTRAN_LOG(                                                              \
        ERR,                                                                \
        "CTRAN must be enabled and initialized for persistent collective"); \
    return ncclInvalidUsage;                                                \
  }

/* The parameter must not be named `type`: the preprocessor would also rewrite
 * the `->type` member accesses below, turning the check into a tautology. */
#define CHECK_PREQ_TYPE(pReq, expectedType)                        \
  if ((pReq)->type != (expectedType)) {                            \
    CTRAN_LOG(                                                     \
        ERR,                                                       \
        "{} requires persistent request type {}, but received {}", \
        __func__,                                                  \
        (expectedType),                                            \
        (pReq)->type);                                             \
    return ncclInvalidArgument;                                    \
  }

#define GET_VALID_PREQ_OR_ERRRETURN(req, pReq)                         \
  do {                                                                 \
    if ((req) == nullptr) {                                            \
      CTRAN_LOG(ERR, "{} received invalid nullptr request", __func__); \
      return ncclInvalidArgument;                                      \
    }                                                                  \
    *(pReq) = reinterpret_cast<CtranPersistentRequest*>(req);          \
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

__attribute__((visibility("default"))) ncclResult_t pExec(void* request) {
  CtranPersistentRequest* pReq = nullptr;
  GET_VALID_PREQ_OR_ERRRETURN(request, &pReq);
  CTRAN_LOG(
      INFO,
      "Executing persistent request {} comm {}",
      (void*)pReq,
      (void*)pReq->comm_);

  if (!ctranInitialized(pReq->comm_)) {
    CTRAN_LOG(
        ERR, "CTRAN must be enabled and initialized for persistent collective");
    return ncclInvalidUsage;
  }

  switch (pReq->type) {
    default:
      CTRAN_LOG(
          ERR,
          "Persistent request {} has unknown op type {}",
          (void*)pReq,
          pReq->type);
      return ncclInvalidArgument;
  }
}

__attribute__((visibility("default"))) ncclResult_t pFree(void* request) {
  CtranPersistentRequest* pReq = nullptr;
  GET_VALID_PREQ_OR_ERRRETURN(request, &pReq);

  switch (pReq->type) {
    case CtranPersistentRequest::Type::ALLGATHER_P:
      NCCLCHECK(metaCommToNccl(ctran::allGatherPDestroy(pReq)));
      break;
    default:
      CTRAN_LOG(
          ERR,
          "Persistent request {} has unknown op type {}",
          (void*)pReq,
          pReq->type);
      return ncclInvalidArgument;
  }
  delete pReq;

  return ncclSuccess;
}

} // namespace ncclx
