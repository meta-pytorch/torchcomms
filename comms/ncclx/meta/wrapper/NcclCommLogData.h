// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comm.h" // ncclComm (version-specific)
#include "nccl.h" // NCCL_VERSION_CODE / NCCL_VERSION

#include "comms/utils/commSpecs.h" // CommLogData

// Version-agnostic accessor for a communicator's NCCLX logging/Scuba metadata.
//
// This header is compiled into every NCCLX version's library (the meta/ tree is
// shared), but where `logMetaData` lives differs by version: NCCL >= 2.30 keeps
// it inside the opaque `ncclxExt` handle (see meta/comm/NcclxCommExt.h), while
// older forks still carry it inline on `ncclComm`. Routing all shared meta/
// (and forked src) reads through this accessor lets one source compile against
// both.
// TODO T279903668: Cleanup version check after v2_29 removal
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)

#include "meta/comm/NcclxCommExt.h"

// v2_30+: logMetaData lives on the opaque ncclxExt handle.
inline CommLogData& ncclCommLogData(ncclComm* comm) {
  return comm->ncclxExt->logMetaData;
}
inline const CommLogData& ncclCommLogData(const ncclComm* comm) {
  return comm->ncclxExt->logMetaData;
}

#else

// pre-2.30 (v2_29, current stable): keep the inline ncclComm::logMetaData
// member.
inline CommLogData& ncclCommLogData(ncclComm* comm) {
  return comm->logMetaData;
}
inline const CommLogData& ncclCommLogData(const ncclComm* comm) {
  return comm->logMetaData;
}

#endif
