// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comm.h"
#include "nccl.h"

// TODO T279903668: Cleanup version check after v2_29 removal
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
#include "meta/comm/NcclxCommExt.h"
#endif

namespace meta::comms::ncclx {

// Version-gated accessor for the per-communicator `channelMetadataOnHost` flag
// (NCCL_CHANNEL_METADATA_LOCATION: keep channel metadata in pinned host memory
// instead of device memory).
//
// In NCCL 2.30+ the flag lives on the opaque `comm->ncclxExt` handle, keeping
// the forked upstream `ncclComm` struct closer to pristine NCCL. In older forks
// (< 2.30, still the stable version) it remains a direct `ncclComm` member.
// Routing shared NCCLX code through this accessor lets it compile against both
// layouts without touching the older fork.
inline bool& ncclCommChannelMetadataOnHost(ncclComm* comm) {
  // TODO T279903668: Cleanup version check after v2_29 removal
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
  return comm->ncclxExt->channelMetadataOnHost;
#else
  return comm->channelMetadataOnHost;
#endif
}

} // namespace meta::comms::ncclx
