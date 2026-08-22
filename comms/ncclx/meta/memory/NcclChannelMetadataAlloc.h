// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <memory>

#include "comms/ctran/memory/SlabAllocator.h"
#include "comms/ctran/memory/Utils.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "meta/comm/NcclxCommExt.h"
#include "meta/transport/transportExt.h"
#include "meta/wrapper/MetaFactory.h"
#include "meta/wrapper/NcclCommLogData.h"

namespace meta::comms::ncclx {

// Selects the channel-metadata allocation strategy for a communicator, hoisted
// out of the forked upstream `init.cc` so that file stays close to pristine
// NCCL. Establishing the strategy here also puts it next to the
// `allocChannelMetadata` / `freeChannelMetadata` paths that act on it.
//
// `::ncclx` is qualified because unqualified it resolves to the enclosing
// `meta::comms::ncclx`.
inline void initChannelMetadataPolicy(ncclComm* comm) {
  if (NCCL_MEM_USE_SLAB_ALLOCATOR) {
    comm->ncclxExt->slabAllocator =
        std::make_unique<::ncclx::memory::SlabAllocator>();
  }
  comm->ncclxExt->channelMetadataOnHost = ::ncclx::channelMetadataOnHost();
}

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
    // Qualified: this header lives in namespace `meta`, where an unqualified
    // `size_t` resolves to range-v3's `meta::size_t` alias template in any
    // translation unit that pulls it in.
    std::size_t numElems,
    cudaStream_t stream,
    const char* callsite,
    bool pushFree) {
  if (comm->ncclxExt->channelMetadataOnHost) {
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

// Releases a shared-resource channel-metadata buffer, i.e. one allocated by
// `allocChannelMetadata` with `pushFree=false`. Per-channel buffers are not
// freed here: they register their free at allocation time and are released
// through the communicator's push-free list.
//
// This mirrors the allocation-side ownership rule, which is why it lives next
// to it: slab-owned device memory is released with the slab, so it is never
// freed individually.
template <typename T>
inline ncclResult_t freeChannelMetadata(ncclComm* comm, T* ptr) {
  if (comm->ncclxExt->channelMetadataOnHost || !NCCL_MEM_USE_SLAB_ALLOCATOR) {
    return ncclCudaFree(ptr, comm->memManager);
  }
  return ncclSuccess;
}

} // namespace meta::comms::ncclx
