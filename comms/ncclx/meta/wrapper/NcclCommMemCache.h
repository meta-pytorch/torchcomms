// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "comms/ctran/memory/memCacheAllocator.h"

#include "comm.h"
#include "nccl.h"

// TODO T279903668: Cleanup version check after v2_29 removal
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
#include "meta/comm/NcclxCommExt.h"
#endif

namespace meta::comms::ncclx {

// Version-gated accessor for the per-communicator memory-cache allocator
// (`memCache`), used by the transport-ext P2P sync-buffer path.
//
// In NCCL 2.30+ the member lives on the opaque `comm->ncclxExt` handle, keeping
// the forked upstream `ncclComm` struct closer to pristine NCCL. In older forks
// (< 2.30, still the stable version) it remains a direct `ncclComm` member.
// Routing shared NCCLX code through this accessor lets it compile against both
// layouts without touching the older fork.
inline std::shared_ptr<::ncclx::memory::memCacheAllocator>& ncclCommMemCache(
    ncclComm* comm) {
  // TODO T279903668: Cleanup version check after v2_29 removal
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
  return comm->ncclxExt->memCache;
#else
  return comm->memCache;
#endif
}

} // namespace meta::comms::ncclx
