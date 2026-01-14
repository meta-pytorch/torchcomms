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
    FB_ERRORRETURN(ncclInternalError, "ncclWin requires CUMEM support.");
  }

  if (!ctranInitialized(comm->ctranComm_.get())) {
    FB_ERRORRETURN(ncclInternalError, "ncclWin requires Ctran support.");
  }

  auto statex = comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    FB_ERRORRETURN(
        ncclInternalError, "Communicator does not have statex initialized.");
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

  ncclWindow_t handle = new ncclWindow_vidmem();
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
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Invalid baseptr to create shared buffer in ncclWinRegister.");
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

  ncclWindow_t handle = new ncclWindow_vidmem();
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
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Invalid parameter(s) to query shared buffer in ncclWinSharedQuery: comm {}, win {}",
        (void*)comm,
        (void*)win);
  }

  auto statex = comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    FB_ERRORRETURN(ncclInternalError, "Empty communicator statex.");
  }

  NCCLCHECK(metaCommToNccl(
      ctran::ctranWinSharedQuery(rank, ncclWinPtr->ctranWindow, addr)));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclWinFree, ncclComm_t comm, ncclWindow_t win);
ncclResult_t ncclWinFree(ncclComm_t comm, ncclWindow_t win) {
  ncclWin* ncclWinPtr = ncclWinMap().find(win);
  if (!comm || !win || !ncclWinPtr || comm != ncclWinPtr->comm) {
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Invalid parameter(s) to free window: comm {}, win {}",
        (void*)comm,
        (void*)win);
  }

  auto statex = comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    FB_ERRORRETURN(ncclInternalError, "Empty communicator statex.");
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
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Invalid parameter(s) in ncclWinGetAttributes: win {}, attr {}",
        (void*)win,
        (void*)attr);
  }

  auto statex = ncclWinPtr->comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    FB_ERRORRETURN(ncclInternalError, "Empty communicator statex.");
  }

  if (rank < 0 || rank >= statex->nRanks()) {
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Invalid rank {} in ncclWinGetAttributes: must be in range [0, {})",
        rank,
        statex->nRanks());
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
