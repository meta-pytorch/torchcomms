// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <nccl.h>
#include "comm.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/utils/Checks.h"
#include "meta/wrapper/MetaFactory.h"

#include "ncclWin.h"

namespace {

// Helper to validate window handle and get ncclWin pointer with Ctran check
ncclResult_t
getValidatedNcclWin(ncclWindow_t win, ncclWin** outWin, const char* funcName) {
  ncclWin* ncclWinPtr = ncclWinMap().find(win);
  if (!ncclWinPtr) {
    FB_ERRORRETURN(ncclInvalidUsage, "Invalid window handle in {}", funcName);
  }
  auto comm = ncclWinPtr->comm->ctranComm_.get();
  if (!ctranInitialized(comm)) {
    FB_ERRORRETURN(ncclInternalError, "{} requires Ctran support", funcName);
  }
  *outWin = ncclWinPtr;
  return ncclSuccess;
}

} // namespace

NCCL_API(
    ncclResult_t,
    ncclPutSignal,
    const void* origin_buff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t target_disp,
    ncclWindow_t win,
    cudaStream_t stream);
ncclResult_t ncclPutSignal(
    const void* origin_buff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t target_disp,
    ncclWindow_t win,
    cudaStream_t stream) {
  ncclWin* ncclWinPtr = nullptr;
  NCCLCHECK(getValidatedNcclWin(win, &ncclWinPtr, "ncclPutSignal"));
  return metaCommToNccl(ctranPutSignal(
      origin_buff,
      count,
      ncclToMetaComm(datatype),
      peer,
      target_disp,
      ncclWinPtr->ctranWindow,
      stream,
      true));
}

NCCL_API(
    ncclResult_t,
    ncclPut,
    const void* origin_buff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t target_disp,
    ncclWindow_t win,
    cudaStream_t stream);
ncclResult_t ncclPut(
    const void* origin_buff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t target_disp,
    ncclWindow_t win,
    cudaStream_t stream) {
  ncclWin* ncclWinPtr = nullptr;
  NCCLCHECK(getValidatedNcclWin(win, &ncclWinPtr, "ncclPut"));
  return metaCommToNccl(ctranPutSignal(
      origin_buff,
      count,
      ncclToMetaComm(datatype),
      peer,
      target_disp,
      ncclWinPtr->ctranWindow,
      stream,
      false));
}

NCCL_API(
    ncclResult_t,
    ncclGet,
    void* target_buff,
    size_t target_disp,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclWindow_t win,
    cudaStream_t stream);
ncclResult_t ncclGet(
    void* target_buff,
    size_t target_disp,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclWindow_t win,
    cudaStream_t stream) {
  ncclWin* ncclWinPtr = nullptr;
  NCCLCHECK(getValidatedNcclWin(win, &ncclWinPtr, "ncclGet"));
  auto comm = ncclWinPtr->comm->ctranComm_.get();
  return metaCommToNccl(ctranGet(
      target_buff,
      target_disp,
      count,
      ncclToMetaComm(datatype),
      peer,
      ncclWinPtr->ctranWindow,
      comm,
      stream));
}

NCCL_API(
    ncclResult_t,
    ncclWaitSignal,
    int peer,
    ncclWindow_t win,
    cudaStream_t stream);
ncclResult_t ncclWaitSignal(int peer, ncclWindow_t win, cudaStream_t stream) {
  ncclWin* ncclWinPtr = nullptr;
  NCCLCHECK(getValidatedNcclWin(win, &ncclWinPtr, "ncclWaitSignal"));
  return metaCommToNccl(ctranWaitSignal(peer, ncclWinPtr->ctranWindow, stream));
}

NCCL_API(
    ncclResult_t,
    ncclSignal,
    int peer,
    ncclWindow_t win,
    cudaStream_t stream);
ncclResult_t ncclSignal(int peer, ncclWindow_t win, cudaStream_t stream) {
  ncclWin* ncclWinPtr = nullptr;
  NCCLCHECK(getValidatedNcclWin(win, &ncclWinPtr, "ncclSignal"));
  return metaCommToNccl(ctranSignal(peer, ncclWinPtr->ctranWindow, stream));
}

namespace ncclx {

__attribute__((visibility("default"))) ncclResult_t
winAllGatherPSupported(ncclWindow_t win, bool* supported) {
  ncclWin* ncclWinPtr = nullptr;
  NCCLCHECK(getValidatedNcclWin(win, &ncclWinPtr, "winAllGatherPSupported"));
  *supported = ncclWinPtr->ctranWindow->allGatherPSupported();
  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclResult_t winAllGatherInit(
    ncclWindow_t win,
    ncclComm_t comm,
    cudaStream_t stream,
    void** request) {
  ncclWin* ncclWinPtr = nullptr;
  NCCLCHECK(getValidatedNcclWin(win, &ncclWinPtr, "winAllGatherInit"));

  bool supported = false;
  NCCLCHECK(winAllGatherPSupported(win, &supported));
  if (!supported) {
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Window AllGather is not supported. Check whether CTRAN is enabled.");
  }

  CtranPersistentRequest* pReq = nullptr;
  NCCLCHECK(metaCommToNccl(
      ctran::allGatherWinInit(
          ncclWinPtr->ctranWindow, comm->ctranComm_.get(), stream, pReq)));
  *request = reinterpret_cast<void*>(pReq);
  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclResult_t winAllGatherExec(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    void* request) {
  if (request == nullptr) {
    FB_ERRORRETURN(
        ncclInvalidArgument, "winAllGatherExec received null request");
  }
  auto* pReq = reinterpret_cast<CtranPersistentRequest*>(request);
  if (pReq->type != CtranPersistentRequest::Type::ALLGATHER_P_WIN) {
    FB_ERRORRETURN(
        ncclInvalidArgument,
        "winAllGatherExec requires ALLGATHER_P_WIN request type, got {}",
        pReq->type);
  }
  return metaCommToNccl(
      ctran::allGatherWinExec(sendbuff, count, ncclToMetaComm(datatype), pReq));
}

__attribute__((visibility("default"))) ncclResult_t
winAllGatherDestroy(void* request) {
  if (request == nullptr) {
    FB_ERRORRETURN(
        ncclInvalidArgument, "winAllGatherDestroy received null request");
  }
  auto* pReq = reinterpret_cast<CtranPersistentRequest*>(request);
  if (pReq->type != CtranPersistentRequest::Type::ALLGATHER_P_WIN) {
    FB_ERRORRETURN(
        ncclInvalidArgument,
        "winAllGatherDestroy requires ALLGATHER_P_WIN request type, got {}",
        pReq->type);
  }
  NCCLCHECK(metaCommToNccl(ctran::allGatherWinDestroy(pReq)));
  delete pReq;
  return ncclSuccess;
}

} // namespace ncclx
