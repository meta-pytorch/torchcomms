// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <nccl.h>
#include "comm.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/utils/Checks.h"
#include "meta/wrapper/MetaFactory.h"

#include "ncclWin.h"

NCCL_API(
    ncclResult_t,
    ncclPutSignal,
    const void* origin_buff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t target_disp,
    ncclWin_t win,
    cudaStream_t stream);
ncclResult_t ncclPutSignal(
    const void* origin_buff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t target_disp,
    ncclWin_t win,
    cudaStream_t stream) {
  auto comm = win->comm->ctranComm_.get();
  if (!ctranInitialized(comm)) {
    FB_ERRORRETURN(ncclInternalError, "ncclPutSignal requires Ctran support");
  }
  return metaCommToNccl(ctranPutSignal(
      origin_buff,
      count,
      ncclToMetaComm(datatype),
      peer,
      target_disp,
      win->ctranWindow,
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
    ncclWaitSignal,
    int peer,
    ncclWin_t win,
    cudaStream_t stream);
ncclResult_t ncclWaitSignal(int peer, ncclWin_t win, cudaStream_t stream) {
  auto comm = win->comm->ctranComm_.get();
  if (!ctranInitialized(comm)) {
    FB_ERRORRETURN(ncclInternalError, "ncclWaitSignal requires Ctran support");
  }
  return metaCommToNccl(ctranWaitSignal(peer, win->ctranWindow, stream));
}

NCCL_API(
    ncclResult_t,
    ncclPutSignal_v2,
    const void* origin_buff,
    size_t target_disp,
    size_t count,
    ncclDataType_t datatype,
    size_t signal_disp,
    uint64_t signal_val,
    int peer,
    ncclWin_t win,
    cudaStream_t stream);
ncclResult_t ncclPutSignal_v2(
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
  WARN(
      "ncclPutSignal_v2 is deprecated; please use ncclPutSignal instead. The arguments signal_disp={%ld} and signal_val={%ld} are ignored. The peer argument ({%d}) is now used as disp.",
      signal_disp,
      signal_val,
      peer);
  return metaCommToNccl(ctranPutSignal(
      origin_buff,
      count,
      ncclToMetaComm(datatype),
      peer,
      target_disp,
      win->ctranWindow,
      stream,
      true));
}

NCCL_API(
    ncclResult_t,
    ncclWaitSignal_v2,
    size_t signal_disp,
    uint64_t cmp_val,
    ncclCmpOp_t cmp_op,
    ncclWin_t win,
    cudaStream_t stream);
ncclResult_t ncclWaitSignal_v2(
    size_t signal_disp,
    uint64_t cmp_val,
    ncclCmpOp_t cmp_op,
    ncclWin_t win,
    cudaStream_t stream) {
  auto comm = win->ctranWindow->comm;
  if (!ctranInitialized(comm)) {
    FB_ERRORRETURN(ncclInternalError, "ncclWaitSignal requires Ctran support");
  }
  WARN(
      "ncclWaitSignal_v2 is deprecated; please use ncclWaitSignal instead. The arguments cmp_val={%ld}, cmp_op={%d} are ignored. signal_disp ={%ld} must equals to peer rank",
      cmp_val,
      cmp_op,
      signal_disp);
  assert(signal_disp < comm->statex_.get()->nRanks());
  return metaCommToNccl(ctranWaitSignal(signal_disp, win->ctranWindow, stream));
}

NCCL_API(
    ncclResult_t,
    ncclSignal,
    size_t signalDisp, // TODO: to be deprecated
    uint64_t signalVal, // TODO: to be deprecated
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
  return metaCommToNccl(ctranSignal(peer, win->ctranWindow, stream));
}
