// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <nccl.h>
#include "comm.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/utils/Checks.h"
#include "meta/wrapper/MetaFactory.h"

#include "ncclWin.h"

NCCL_API(
    ncclResult_t,
    ncclPutSignal_old,
    const void* origin_buff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t target_disp,
    ncclWin_t win,
    cudaStream_t stream);
ncclResult_t ncclPutSignal_old(
    const void* origin_buff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t target_disp,
    ncclWin_t win,
    cudaStream_t stream) {
  auto comm = win->comm->ctranComm_.get();
  if (!ctranInitialized(comm)) {
    FB_ERRORRETURN(
        ncclInternalError, "ncclPutSignal_old requires Ctran support");
  }
  return metaCommToNccl(ctranPutSignal(
      origin_buff,
      count,
      ncclToMetaComm(datatype),
      peer,
      target_disp,
      win->ctranWindow,
      comm,
      stream));
}

NCCL_API(
    ncclResult_t,
    ncclPut,
    const void* origin_buff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t target_disp,
    ncclWin_t win,
    cudaStream_t stream);
ncclResult_t ncclPut(
    const void* origin_buff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t target_disp,
    ncclWin_t win,
    cudaStream_t stream) {
  auto comm = win->comm->ctranComm_.get();
  if (!ctranInitialized(comm)) {
    FB_ERRORRETURN(ncclInternalError, "ncclPut requires Ctran support");
  }
  return metaCommToNccl(ctranPutSignal(
      origin_buff,
      count,
      ncclToMetaComm(datatype),
      peer,
      target_disp,
      win->ctranWindow,
      comm,
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
    ncclWin_t win,
    cudaStream_t stream);
ncclResult_t ncclGet(
    void* target_buff,
    size_t target_disp,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclWin_t win,
    cudaStream_t stream) {
  auto comm = win->comm->ctranComm_.get();
  if (!ctranInitialized(comm)) {
    FB_ERRORRETURN(ncclInternalError, "ncclGet requires Ctran support");
  }
  return metaCommToNccl(ctranGet(
      target_buff,
      target_disp,
      count,
      ncclToMetaComm(datatype),
      peer,
      win->ctranWindow,
      comm,
      stream));
}

NCCL_API(
    ncclResult_t,
    ncclWaitSignal_old,
    int peer,
    ncclWin_t win,
    cudaStream_t stream);
ncclResult_t ncclWaitSignal_old(int peer, ncclWin_t win, cudaStream_t stream) {
  auto comm = win->comm->ctranComm_.get();
  if (!ctranInitialized(comm)) {
    FB_ERRORRETURN(
        ncclInternalError, "ncclWaitSignal_old requires Ctran support");
  }
  return metaCommToNccl(ctranWaitSignal(peer, win->ctranWindow, comm, stream));
}

NCCL_API(
    ncclResult_t,
    ncclPutSignal,
    const void* origin_buff,
    size_t target_disp,
    size_t count,
    ncclDataType_t datatype,
    size_t signal_disp,
    uint64_t signal_val,
    int peer,
    ncclWin_t win,
    cudaStream_t stream);
ncclResult_t ncclPutSignal(
    const void* origin_buff,
    size_t target_disp,
    size_t count,
    ncclDataType_t datatype,
    size_t signal_disp,
    uint64_t signal_val,
    int peer,
    ncclWin_t win,
    cudaStream_t stream) {
  auto comm = win->ctranWindow->comm;
  if (!ctranInitialized(comm)) {
    FB_ERRORRETURN(ncclInternalError, "ncclPutSignal requires Ctran support");
  }
  return metaCommToNccl(ctranPutSignal_v2(
      origin_buff,
      target_disp,
      count,
      ncclToMetaComm(datatype),
      signal_disp,
      signal_val,
      peer,
      win->ctranWindow,
      stream,
      true));
}

NCCL_API(
    ncclResult_t,
    ncclWaitSignal,
    size_t signal_disp,
    uint64_t cmp_val,
    ncclCmpOp_t cmp_op,
    ncclWin_t win,
    cudaStream_t stream);
ncclResult_t ncclWaitSignal(
    size_t signal_disp,
    uint64_t cmp_val,
    ncclCmpOp_t cmp_op,
    ncclWin_t win,
    cudaStream_t stream) {
  auto comm = win->ctranWindow->comm;
  if (!ctranInitialized(comm)) {
    FB_ERRORRETURN(ncclInternalError, "ncclWaitSignal requires Ctran support");
  }
  return metaCommToNccl(ctranWaitSignal_v2(
      signal_disp, cmp_val, ncclToMetaComm(cmp_op), win->ctranWindow, stream));
}

NCCL_API(
    ncclResult_t,
    ncclSignal,
    size_t signalDisp,
    uint64_t signalVal,
    int peer,
    ncclWin_t win,
    cudaStream_t stream);
ncclResult_t ncclSignal(
    size_t signalDisp,
    uint64_t signalVal,
    int peer,
    ncclWin_t win,
    cudaStream_t stream) {
  auto comm = win->ctranWindow->comm;
  if (!ctranInitialized(comm)) {
    FB_ERRORRETURN(ncclInternalError, "ncclSignal requires Ctran support");
  }
  return metaCommToNccl(
      ctranSignal(signalDisp, signalVal, peer, win->ctranWindow, stream));
}
