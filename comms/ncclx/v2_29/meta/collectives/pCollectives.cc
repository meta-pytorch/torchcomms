// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comm.h"
#include "nccl.h"

#include "comms/ctran/Ctran.h"
#include "comms/utils/checks.h"

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
    FB_ERRORRETURN(                                                         \
        ncclInvalidUsage,                                                   \
        "CTRAN must be enabled and initialized for persistent collective"); \
  }

#define CHECK_PREQ_TYPE(pReq, type)                                \
  if (pReq->type != type) {                                        \
    FB_ERRORRETURN(                                                \
        ncclInvalidArgument,                                       \
        "%s requires persistent request type %d, but received %d", \
        __func__,                                                  \
        type,                                                      \
        pReq->type);                                               \
  }

#define GET_VALID_PREQ_OR_ERRRETURN(req, pReq)                    \
  do {                                                            \
    if (request == nullptr) {                                     \
      FB_ERRORRETURN(                                             \
          ncclInvalidArgument,                                    \
          "%s received invalid nullptr request",                  \
          __func__);                                              \
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
    const size_t totalNumSendBlocks, // number of blocks (tokens) per batch
    const size_t blockCount, // number of elements per block (token)
    const size_t blockNumRecvBuckets, // number of receiving buckets for each
                                      // block (experts per token, topK)
    const int numRecvBuckets, // number of receiving buckets per rank (expert
                              // per rank)
    const ncclx::Hints& hints,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    void** request) {
  CHECK_VALID_CTRAN(comm->ctranComm_.get());

  SetCudaDevRAII setCudaDev(comm->cudaDev);
  CtranPersistentRequest* pReq = nullptr;

  NCCLCHECK(metaCommToNccl(
      ::ctran::allToAllvDedupInit(
          totalNumSendBlocks,
          blockCount,
          blockNumRecvBuckets,
          numRecvBuckets,
          ncclToMetaComm(hints),
          ncclToMetaComm(datatype),
          comm->ctranComm_.get(),
          stream,
          pReq)));
  *request = reinterpret_cast<void*>(pReq);
  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclResult_t allToAllvDedupExec(
    const void* sendBuff,
    const int sendIdx[],
    const int fwdIdx[],
    const int recvIdx[],
    void* recvBuff,
    int recvBlockIds[],
    void* request) {
  CtranPersistentRequest* pReq = nullptr;
  GET_VALID_PREQ_OR_ERRRETURN(request, &pReq);
  CHECK_PREQ_TYPE(pReq, CtranPersistentRequest::Type::ALLTOALLV_DEDUP);
  CHECK_VALID_CTRAN(pReq->comm_);

  return metaCommToNccl(
      ::ctran::allToAllvDedupExec(
          sendBuff, sendIdx, fwdIdx, recvIdx, recvBuff, recvBlockIds, pReq));
}

__attribute__((visibility("default"))) ncclResult_t pExec(void* request) {
  CtranPersistentRequest* pReq = nullptr;
  GET_VALID_PREQ_OR_ERRRETURN(request, &pReq);
  CLOGF(
      INFO,
      "Executing persistent request {} comm {}",
      (void*)pReq,
      (void*)pReq->comm_);

  if (!ctranInitialized(pReq->comm_)) {
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "CTRAN must be enabled and initialized for persistent collective");
  }

  switch (pReq->type) {
    default:
      FB_ERRORRETURN(
          ncclInvalidArgument,
          "Persistent request {} has unknown op type {}",
          (void*)pReq,
          (void*)pReq->type);
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
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Persistent AllToAll is not supported. Check whether CTRAN is enabled.");
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
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "request shouldn't be nullptr for persistent collective");
  }
  CtranPersistentRequest* pReq =
      reinterpret_cast<CtranPersistentRequest*>(request);

  if (!ctranInitialized(pReq->comm_)) {
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "CTRAN must be enabled and initialized for persistent collective");
  }

  if (pReq->type != CtranPersistentRequest::Type::ALLTOALL_P) {
    FB_ERRORRETURN(
        ncclInvalidArgument,
        "Unexpected PersistentRequest type %d called into AllToAllExec",
        pReq->type);
  }

  return metaCommToNccl(ctran::AllToAllPExec(sendbuff, count, pReq));
}

__attribute__((visibility("default"))) ncclResult_t pFree(void* request) {
  CtranPersistentRequest* pReq = nullptr;
  GET_VALID_PREQ_OR_ERRRETURN(request, &pReq);

  switch (pReq->type) {
    case CtranPersistentRequest::Type::ALLGATHER_P:
      NCCLCHECK(metaCommToNccl(ctran::allGatherPDestroy(pReq)));
      break;
    case CtranPersistentRequest::Type::ALLTOALL_P:
      return metaCommToNccl(ctran::AllToAllPDestroy(pReq));
    case CtranPersistentRequest::Type::ALLTOALLV_DEDUP:
      NCCLCHECK(metaCommToNccl(::ctran::allToAllvDedupDestroy(pReq)));
      break;
    default:
      FB_ERRORRETURN(
          ncclInvalidArgument,
          "Persistent request {} has unknown op type {}",
          (void*)pReq,
          pReq->type);
  }
  delete pReq;

  return ncclSuccess;
}

} // namespace ncclx
