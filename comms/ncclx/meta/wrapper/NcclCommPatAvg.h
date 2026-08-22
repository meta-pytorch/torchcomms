// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comm.h"
#include "nccl.h"

// TODO T279903668: Cleanup version check after v2_29 removal
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
#include "meta/comm/NcclxCommExt.h"
#endif

namespace meta::comms::ncclx {

// Version-gated accessor for the per-communicator `usePatAvg` flag (forces the
// deterministic PAT algorithm with ncclDevPatSumPostDiv for ncclAvg
// ReduceScatter on supported datatypes).
//
// In NCCL 2.30+ the flag lives on the opaque `comm->ncclxExt` handle, keeping
// the forked upstream `ncclComm` struct closer to pristine NCCL. In older forks
// (< 2.30, still the stable version) it remains a direct `ncclComm` member.
// Routing shared NCCLX code and tests through this accessor lets them compile
// against both layouts without touching the older fork.
inline bool& ncclCommUsePatAvg(ncclComm* comm) {
  // TODO T279903668: Cleanup version check after v2_29 removal
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
  return comm->ncclxExt->usePatAvg;
#else
  return comm->usePatAvg_;
#endif
}

} // namespace meta::comms::ncclx
