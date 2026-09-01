// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comm.h"
#include "nccl.h"

// TODO T279903668: Cleanup version check after v2_29 removal
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
#include "meta/comm/NcclxCommExt.h"
#endif

namespace meta::comms::ncclx {

// Version-gated accessor for the per-communicator `useCtran` flag (route
// supported collectives through CTRAN instead of the baseline path).
//
// In NCCL 2.30+ the flag lives on the opaque `comm->ncclxExt` handle, keeping
// the forked upstream `ncclComm` struct closer to pristine NCCL. In older forks
// (< 2.30, still the stable version) it remains a direct `ncclComm` member.
// Routing shared NCCLX code and tests through this accessor lets them compile
// against both layouts without touching the older fork.
inline bool& ncclCommUseCtran(ncclComm* comm) {
  // TODO T279903668: Cleanup version check after v2_29 removal
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
  return comm->ncclxExt->useCtran;
#else
  return comm->useCtran_;
#endif
}

} // namespace meta::comms::ncclx
