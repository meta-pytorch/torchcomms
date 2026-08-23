// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/memory/SlabAllocator.h"
#include "comms/ctran/memory/Utils.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "meta/comm/NcclxCommExt.h"
#include "meta/wrapper/MetaFactory.h"
#include "meta/wrapper/NcclCommLogData.h"

namespace meta::comms::ncclx {

// Allocates one `initChannel` channel-metadata buffer, hoisted out of the
// forked upstream `channel.cc` so that file stays close to pristine NCCL.
//
// This encapsulates the NCCLX channel-metadata allocation strategy:
//  - `NCCL_CHANNEL_METADATA_LOCATION == host` -> pinned host memory;
//  - otherwise the per-communicator batched `SlabAllocator`
//    (when `NCCL_MEM_USE_SLAB_ALLOCATOR` is set), else the plain async device
//    allocator.
//
// `pushFree` registers a matching free for buffers the communicator owns
// individually (per-channel); shared-resource buffers, which are freed together
// in comm teardown, pass `pushFree=false`. Slab-owned device memory is released
// with the slab, so it is never individually push-freed.
template <typename T>
inline ncclResult_t allocChannelMetadata(
    ncclComm* comm,
    T** ptr,
    size_t numElems,
    cudaStream_t stream,
    const char* callsite,
    bool pushFree) {
  if (comm->channelMetadataOnHost) {
    NCCLCHECK(ncclCudaHostCalloc(ptr, numElems));
    if (pushFree) {
      ncclCommPushCudaHostFree(comm, *ptr);
    }
    return ncclSuccess;
  }
  NCCLCHECK(metaCommToNccl(
      ::ncclx::memory::cudaCallocAsync(
          ptr,
          numElems,
          stream,
          &::ncclCommLogData(comm),
          callsite,
          comm->ncclxExt->slabAllocator.get())));
  if (pushFree && !NCCL_MEM_USE_SLAB_ALLOCATOR) {
    ncclCommPushCudaFree(comm, *ptr);
  }
  return ncclSuccess;
}

} // namespace meta::comms::ncclx
