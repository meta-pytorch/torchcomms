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

  // Create empty ncclWindow as handle and register mapping
  ncclWindow_t handle = new ncclWindow();
  windowMap()[handle] = win_;
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

  // Create empty ncclWindow as handle and register mapping
  ncclWindow_t handle = new ncclWindow();
  windowMap()[handle] = win_;
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
  auto it = windowMap().find(win);
  ncclWin* w = (it != windowMap().end()) ? it->second : nullptr;
  if (!comm || !win || !w || comm != w->comm) {
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Invalid parameter(s) to query shared buffere in ncclWinSharedQuery: comm {}, win {}",
        (void*)comm,
        (void*)win);
  }

  auto statex = comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    FB_ERRORRETURN(ncclInternalError, "Empty communicator statex.");
  }

  NCCLCHECK(
      metaCommToNccl(ctran::ctranWinSharedQuery(rank, w->ctranWindow, addr)));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclWinFree, ncclComm_t comm, ncclWindow_t win);
ncclResult_t ncclWinFree(ncclComm_t comm, ncclWindow_t win) {
  auto it = windowMap().find(win);
  ncclWin* w = (it != windowMap().end()) ? it->second : nullptr;
  if (!comm || !win || !w || comm != w->comm) {
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Invalid parameter(s) to query shared buffere in ncclWinSharedQuery: comm {}, win {}",
        (void*)comm,
        (void*)win);
  }

  auto statex = comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    FB_ERRORRETURN(ncclInternalError, "Empty communicator statex.");
  }

  NCCLCHECK(metaCommToNccl(ctran::ctranWinFree(w->ctranWindow)));
  windowMap().erase(win);
  delete w;
  delete win; // Delete the empty ncclWindow handle
  return ncclSuccess;
}
