// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>

#include "nccl.h"

struct ncclComm;

// CTRAN buffer-registration facade.
//
// Concentrates the CTRAN side of ncclCommRegister / ncclCommDeregister and the
// global pointer-based (de)registration APIs, so the routing decision and the
// CTRAN calls live in NCCLX-only code rather than woven into the forked
// upstream register.cc.
namespace ncclx {

// True when NCCLX is configured to route buffer registration through CTRAN.
bool ctranRegisterEnabled();

// True when this communicator's CTRAN instance owns buffer (de)registration.
bool ctranOwnsRegister(ncclComm* comm);

// Log and return the error for enabling NCCL_CTRAN_REGISTER and
// NCCL_LOCAL_REGISTER at the same time.
ncclResult_t logCtranRegisterConflict();

// Register/deregister a buffer through this communicator's CTRAN instance.
ncclResult_t
ctranCommRegister(ncclComm* comm, void* buff, size_t size, void** handle);
ncclResult_t ctranCommDeregister(ncclComm* comm, void* handle);

// Global pointer-based (de)registration (no handle, no comm required).
ncclResult_t ctranGlobalRegisterWithPtr(void* buff, size_t size);
ncclResult_t ctranGlobalDeregisterWithPtr(void* buff, size_t size);

} // namespace ncclx
