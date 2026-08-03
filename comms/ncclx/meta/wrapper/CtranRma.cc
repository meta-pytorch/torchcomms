// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/wrapper/CtranRma.h"

#include <folly/ScopeGuard.h>

#include "comm.h"
#include "cudawrap.h"
#include "nccl_device/core.h"

#include "comms/ctran/hints/Hints.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/NcclxConfig.h"
#include "meta/rma/ncclWin.h"
#include "meta/wrapper/MetaFactory.h"
#include "meta/wrapper/NcclCommCtran.h"

namespace ncclx {

ncclResult_t ctranWinRegisterIfOwned(
    ncclComm* comm,
    void* buff,
    size_t size,
    ncclWindow_t* win,
    int winFlags,
    bool* handled) {
  *handled = false;

  // NCCL_WIN_DEVICE_API flag bypasses CTRAN and forces NCCL orig path. This is
  // needed for the device API (GIN support), which only exists in the orig
  // path.
  const bool forceOrigPath = (winFlags & NCCL_WIN_DEVICE_API) != 0;
  if (forceOrigPath ||
      NCCLX_CONFIG_FIELD(comm->config, rmaAlgo) == NCCL_RMA_ALGO::orig ||
      !ctranInitialized(meta::comms::ncclx::ncclCommCtran(comm).get())) {
    return ncclSuccess;
  }

  if (!ncclGetCuMemSysSupported()) {
    CERR(commInternalError, "ncclWin requires CUMEM support.");
    return ncclInternalError;
  }
  if (buff == nullptr) {
    CERR(
        commInvalidUsage,
        "Invalid baseptr to create shared buffer in ncclWinRegister.");
    return ncclInvalidUsage;
  }

  ncclWin* win_ = new ncclWin();
  win_->comm = comm;

  auto guard = folly::makeGuard([win_] { delete win_; });
  // Bridge the comm-level ncclx::win_register_ipc_only hint into the ctran
  // window hints, as there is no per-window config path from Python today.
  meta::comms::Hints winHints;
  NCCLCHECK(metaCommToNccl(winHints.set(
      "win_register_ipc_only",
      NCCLX_CONFIG_FIELD(comm->config, winRegisterIpcOnly) ? "1" : "0")));
  NCCLCHECK(metaCommToNccl(winHints.set(
      "win_register_enable_signal",
      NCCLX_CONFIG_FIELD(comm->config, winRegisterEnableSignal) ? "1" : "0")));
  NCCLCHECK(metaCommToNccl(winHints.set(
      "win_register_symmetric",
      NCCLX_CONFIG_FIELD(comm->config, winRegisterSymmetric) ? "1" : "0")));
  NCCLCHECK(metaCommToNccl(
      ctran::ctranWinRegister(
          buff,
          size,
          meta::comms::ncclx::ncclCommCtran(comm).get(),
          &win_->ctranWindow,
          winHints)));

  // Create empty ncclWindow as handle and register mapping.
  ncclWindow_t handle = new ncclWindow_vidmem();
  ncclWinMap().insert(handle, win_);
  *win = handle;
  guard.dismiss();
  *handled = true;
  return ncclSuccess;
}

ncclResult_t
ctranWinDeregisterIfOwned(ncclComm* comm, ncclWindow_t winDev, bool* handled) {
  *handled = false;

  if (NCCLX_CONFIG_FIELD(comm->config, rmaAlgo) == NCCL_RMA_ALGO::orig ||
      !ctranInitialized(meta::comms::ncclx::ncclCommCtran(comm).get())) {
    return ncclSuccess;
  }

  ncclWin* ncclWinPtr = ncclWinMap().find(winDev);
  // If the window is not in the CTRAN map (e.g. registered with
  // NCCL_WIN_DEVICE_API), leave *handled = false so the caller falls through to
  // the symmetric/orig deregistration path.
  if (ncclWinPtr == nullptr || comm != ncclWinPtr->comm) {
    return ncclSuccess;
  }

  auto statex = meta::comms::ncclx::ncclCommCtran(comm)->statex_.get();
  if (statex == nullptr) {
    CERR(commInternalError, "Empty communicator statex.");
    return ncclInternalError;
  }

  // Remove from the map first, then clean up resources.
  ncclWinMap().erase(winDev);
  auto guard = folly::makeGuard([winDev, ncclWinPtr] {
    delete ncclWinPtr;
    delete winDev;
  });

  NCCLCHECK(metaCommToNccl(ctran::ctranWinFree(ncclWinPtr->ctranWindow)));
  *handled = true;
  return ncclSuccess;
}

} // namespace ncclx
