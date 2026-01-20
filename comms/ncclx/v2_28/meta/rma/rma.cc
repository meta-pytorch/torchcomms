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
    size_t signalDisp,
    uint64_t signalVal,
    int peer,
    ncclWindow_t win,
    cudaStream_t stream);
ncclResult_t ncclSignal(
    size_t signalDisp,
    uint64_t signalVal,
    int peer,
    ncclWindow_t win,
    cudaStream_t stream) {
  ncclWin* ncclWinPtr = nullptr;
  NCCLCHECK(getValidatedNcclWin(win, &ncclWinPtr, "ncclSignal"));
  return metaCommToNccl(ctranSignal(peer, ncclWinPtr->ctranWindow, stream));
}
