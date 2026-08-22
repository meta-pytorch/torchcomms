// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "nccl.h"

struct ncclComm;

// CTRAN integration hooks invoked from forked upstream lifecycle/group code.
//
// Keeps the CTRAN calls (and the CVARs/handles they need) in NCCLX-only code so
// the forked upstream group.cc / init.cc carry only a single seam line each.
namespace ncclx {

// Run the CTRAN group-end hook (flushes any CTRAN ops batched during the
// group).
ncclResult_t runCtranGroupEndHook();

// When CTRAN is enabled and initialized on comm, fold its async error into
// *asyncError, but only when the baseline path itself reported success or
// in-progress (so a baseline error is never masked).
void ctranUpdateAsyncError(ncclComm* comm, ncclResult_t* asyncError);

} // namespace ncclx
