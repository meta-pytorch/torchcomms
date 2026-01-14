// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "nccl.h"
#include "p2p.h"

#include "comms/ctran/memory/Utils.h"
#include "comms/ctran/memory/memCacheAllocator.h"
#include "comms/utils/commSpecs.h"
#include "meta/wrapper/MetaFactory.h"

namespace ncclx::memory {

static ncclResult_t metaAllocateShareableBuffer(
    size_t size,
    int refcount,
    ncclIpcDesc* ipcDesc,
    void** ptr,
    const char* use,
    std::shared_ptr<ncclx::memory::memCacheAllocator> memCache,
    struct CommLogData* logMetaData) {
  ncclx::memory::allocatorIpcDesc d;
  NCCLCHECK(metaCommToNccl(
      ncclx::memory::allocateShareableBuffer(
          size, refcount, &d, ptr, memCache, logMetaData, use)));
  INFO(
      NCCL_ALLOC,
      "refCnt for key %s increased, need to release when kernel finishes",
      use);
  if (d.udsMemHandle.has_value()) {
    memcpy(
        &ipcDesc->cuDesc.data,
        &d.udsMemHandle.value(),
        sizeof(d.udsMemHandle.value()));
  }
#if CUDART_VERSION >= 12040
  if (d.fabricHandle.has_value()) {
    ipcDesc->cuDesc.handle = d.fabricHandle.value();
  }
#endif
  if (d.memHandle.has_value()) {
    ipcDesc->memHandle = d.memHandle.value();
  }
  return ncclSuccess;
}

} // namespace ncclx::memory
