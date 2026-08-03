// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>

#include "nccl.h"

struct ncclComm;

// CTRAN RMA-window dispatch facade.
//
// Concentrates the CTRAN side of ncclCommWindowRegister /
// ncclCommWindowDeregister so the routing decision (config rmaAlgo +
// CTRAN-initialized) and the CTRAN window calls live in NCCLX-only code rather
// than woven into the forked upstream dev_runtime.cc. Each entry point reports
// through *handled whether it serviced the call; when false the caller falls
// through to the upstream orig path.
namespace ncclx {

// If this communicator routes RMA windows through CTRAN (config rmaAlgo != orig
// and CTRAN is initialized) and winFlags do not force the NCCL device-API path,
// register the window through CTRAN and set *handled = true. Otherwise leaves
// *handled = false for the caller's orig path.
ncclResult_t ctranWinRegisterIfOwned(
    ncclComm* comm,
    void* buff,
    size_t size,
    ncclWindow_t* win,
    int winFlags,
    bool* handled);

// If this communicator routes RMA windows through CTRAN and winDev is a
// CTRAN-owned window, deregister it through CTRAN and set *handled = true.
// Otherwise leaves *handled = false for the caller's orig path.
ncclResult_t
ctranWinDeregisterIfOwned(ncclComm* comm, ncclWindow_t winDev, bool* handled);

} // namespace ncclx
