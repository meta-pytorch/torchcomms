// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>

#include <folly/ScopeGuard.h>

#include "checks.h"
#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/CtranWin.h"
#include "meta/wrapper/MetaFactory.h"

#include "nccl.h"
#include "ncclWin.h"

ncclResult_t CheckCommAndReturn(ncclComm_t comm) {
  if (!ncclGetCuMemSysSupported()) {
    ERR(ncclInternalError, "ncclWin requires CUMEM support.");
    return ncclInternalError;
  }

  if (!ctranInitialized(comm->ctranComm_.get())) {
    ERR(ncclInternalError, "ncclWin requires Ctran support.");
    return ncclInternalError;
  }

  auto statex = comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    ERR(ncclInternalError, "Communicator does not have statex initialized.");
    return ncclInternalError;
  }
  return ncclSuccess;
}

NCCL_API(
    ncclResult_t,
    ncclWinAllocate,
    size_t size,
    ncclComm_t comm,
    void** baseptr,
    ncclWindow_t* win,
    const ncclx::Hints& hints);
ncclResult_t ncclWinAllocate(
    size_t size,
    ncclComm_t comm,
    void** baseptr,
    ncclWindow_t* win,
    const ncclx::Hints& hints) {
  NCCLCHECK(CheckCommAndReturn(comm));
  ncclWin* win_ = new ncclWin();
  win_->comm = comm;

  auto guard = folly::makeGuard([win_] { delete win_; });
  NCCLCHECK(metaCommToNccl(
      ctran::ctranWinAllocate(
          size,
          comm->ctranComm_.get(),
          baseptr,
          &win_->ctranWindow,
          ncclToMetaComm(hints))));

  ncclWindow_t handle = new NcclWinHandle();
  ncclWinMap().insert(handle, win_);
  *win = handle;
  guard.dismiss();
  return ncclSuccess;
}

NCCL_API(
    ncclResult_t,
    ncclWinRegister,
    const void* baseptr,
    const size_t size,
    ncclComm_t comm,
    ncclWindow_t* win,
    const ncclx::Hints& hints);
ncclResult_t ncclWinRegister(
    const void* baseptr,
    const size_t size,
    ncclComm_t comm,
    ncclWindow_t* win,
    const ncclx::Hints& hints) {
  NCCLCHECK(CheckCommAndReturn(comm));
  if (baseptr == nullptr) {
    ERR(ncclInvalidUsage,
        "Invalid baseptr to create shared buffer in ncclWinRegister.");
    return ncclInvalidUsage;
  }

  ncclWin* win_ = new ncclWin();
  win_->comm = comm;

  auto guard = folly::makeGuard([win_] { delete win_; });
  NCCLCHECK(metaCommToNccl(
      ctran::ctranWinRegister(
          baseptr,
          size,
          comm->ctranComm_.get(),
          &win_->ctranWindow,
          ncclToMetaComm(hints))));

  ncclWindow_t handle = new NcclWinHandle();
  ncclWinMap().insert(handle, win_);
  *win = handle;
  guard.dismiss();
  return ncclSuccess;
}

NCCL_API(
    ncclResult_t,
    ncclWinSharedQuery,
    int rank,
    ncclComm_t comm,
    ncclWindow_t win,
    void** addr);
ncclResult_t
ncclWinSharedQuery(int rank, ncclComm_t comm, ncclWindow_t win, void** addr) {
  ncclWin* ncclWinPtr = ncclWinMap().find(win);
  if (!comm || !win || !ncclWinPtr || comm != ncclWinPtr->comm) {
    ERR(ncclInvalidUsage,
        "Invalid parameter(s) to query shared buffer in ncclWinSharedQuery: comm %p, win %p",
        (void*)comm,
        (void*)win);
    return ncclInvalidUsage;
  }

  auto statex = comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    ERR(ncclInternalError, "Empty communicator statex.");
    return ncclInternalError;
  }

  NCCLCHECK(metaCommToNccl(
      ctran::ctranWinSharedQuery(rank, ncclWinPtr->ctranWindow, addr)));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclWinFree, ncclComm_t comm, ncclWindow_t win);
ncclResult_t ncclWinFree(ncclComm_t comm, ncclWindow_t win) {
  ncclWin* ncclWinPtr = ncclWinMap().find(win);
  if (!comm || !win || !ncclWinPtr || comm != ncclWinPtr->comm) {
    ERR(ncclInvalidUsage,
        "Invalid parameter(s) to free window: comm %p, win %p",
        (void*)comm,
        (void*)win);
    return ncclInvalidUsage;
  }

  auto statex = comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    ERR(ncclInternalError, "Empty communicator statex.");
    return ncclInternalError;
  }

  // Remove from map first, then cleanup resources
  ncclWinMap().erase(win);

  // Guard ensures cleanup happens on both success and failure paths
  auto guard = folly::makeGuard([win, ncclWinPtr] {
    delete ncclWinPtr;
    delete win;
  });

  NCCLCHECK(metaCommToNccl(ctran::ctranWinFree(ncclWinPtr->ctranWindow)));
  return ncclSuccess;
}

NCCL_API(
    ncclResult_t,
    ncclWinGetAttributes,
    int rank,
    ncclWindow_t win,
    ncclWinAttr_t* attr);
ncclResult_t
ncclWinGetAttributes(int rank, ncclWindow_t win, ncclWinAttr_t* attr) {
  ncclWin* ncclWinPtr = ncclWinMap().find(win);
  if (!win || !ncclWinPtr || !attr) {
    ERR(ncclInvalidUsage,
        "Invalid parameter(s) in ncclWinGetAttributes: win %p, attr %p",
        (void*)win,
        (void*)attr);
    return ncclInvalidUsage;
  }

  auto statex = ncclWinPtr->comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    ERR(ncclInternalError, "Empty communicator statex.");
    return ncclInternalError;
  }

  if (rank < 0 || rank >= statex->nRanks()) {
    ERR(ncclInvalidUsage,
        "Invalid rank %d in ncclWinGetAttributes: must be in range [0, %d)",
        rank,
        statex->nRanks());
    return ncclInvalidUsage;
  }

  auto newAttr = new ncclWinAttr();
  auto guard = folly::makeGuard([newAttr] { delete newAttr; });
  auto nvlEnabled = ncclWinPtr->ctranWindow->nvlEnabled(rank);
  if (nvlEnabled) {
    newAttr->accessType = ncclWinAccessType::ncclWinAccessUnified;
  } else {
    newAttr->accessType = ncclWinAccessType::ncclWinAccessSeparate;
  }
  *attr = newAttr;
  guard.dismiss();
  return ncclSuccess;
}

NCCL_API(
    ncclResult_t,
    ncclWinCreateDeviceWin,
    ncclWindow_t win,
    int signal_count,
    int counter_count,
    int barrier_count,
    void** outDevicePtr);
ncclResult_t ncclWinCreateDeviceWin(
    ncclWindow_t /*win*/,
    int /*signal_count*/,
    int /*counter_count*/,
    int /*barrier_count*/,
    void** /*outDevicePtr*/) {
  return ncclInvalidUsage;
}

NCCL_API(ncclResult_t, ncclWinDestroyDeviceWin, void* devicePtr);
ncclResult_t ncclWinDestroyDeviceWin(void* /*devicePtr*/) {
  return ncclInvalidUsage;
}

NCCL_API(
    ncclResult_t,
    ncclWinLocalRegisterBuffer,
    ncclComm_t comm,
    void* ptr,
    size_t size,
    ncclLkeyPerDevice* outLkeys);
ncclResult_t ncclWinLocalRegisterBuffer(
    ncclComm_t /*comm*/,
    void* /*ptr*/,
    size_t /*size*/,
    ncclLkeyPerDevice* outLkeys) {
  if (outLkeys != nullptr) {
    *outLkeys = ncclLkeyPerDevice{};
  }
  return ncclInvalidUsage;
}

NCCL_API(
    ncclResult_t,
    ncclWinLocalDeregisterBuffer,
    ncclComm_t comm,
    void* ptr);
ncclResult_t ncclWinLocalDeregisterBuffer(ncclComm_t /*comm*/, void* /*ptr*/) {
  return ncclInvalidUsage;
}
