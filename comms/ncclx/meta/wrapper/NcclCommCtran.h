// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cassert>
#include <memory>

#include "comms/ctran/CtranComm.h"

#include "comm.h"
#include "nccl.h"

// TODO T279903668: Cleanup version check after v2_29 removal
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
#include "meta/comm/NcclxCommExt.h"
#endif

namespace meta::comms::ncclx {

// Version-gated accessor for the per-communicator CtranComm handle.
//
// In NCCL 2.30+ this member lives on the opaque `comm->ncclxExt` handle,
// keeping the forked upstream `ncclComm` struct closer to pristine NCCL. In
// older forks
// (< 2.30) it remains a direct `ncclComm` member. Routing every access through
// this accessor lets the shared NCCLX code compile against both layouts without
// touching the older fork. A reference is returned so callers can read
// (`.get()`, `->`, `operator bool`) and manage lifetime (`= make_unique`,
// `.reset()`) exactly as they did with the raw member.
inline std::unique_ptr<CtranComm>& ncclCommCtran(ncclComm* comm) {
  // TODO T279903668: Cleanup version check after v2_29 removal
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
  // ncclxExt spans the communicator's whole lifetime (created right after comm
  // allocation, freed in the NCCLX comm-free hook); assert that invariant
  // before dereferencing rather than fault on a stale/half-built comm.
  assert(comm->ncclxExt != nullptr);
  return comm->ncclxExt->ctranComm;
#else
  return comm->ctranComm_;
#endif
}

} // namespace meta::comms::ncclx
