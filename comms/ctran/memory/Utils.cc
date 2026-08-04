// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/memory/Utils.h"

#include <folly/Format.h>
#include <sstream>

#include "comms/ctran/memory/SlabAllocator.h"
#include "comms/ctran/memory/memCacheAllocator.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"

namespace ncclx::memory {

commResult_t cudaCallocAsync(
    void** ptr,
    size_t nBytes,
    cudaStream_t stream,
    const CommLogData* logMetaData,
    const meta::comms::memtrace::MemCallsite& callsite,
    SlabAllocator* allocator) {
  if (NCCL_MEM_USE_SLAB_ALLOCATOR && allocator) {
    return allocator->cuCallocAsync(ptr, nBytes, stream, callsite, logMetaData);
  }

  return ctran::utils::commCudaCallocAsync(
      (char**)ptr, nBytes, stream, logMetaData, callsite);
}

commResult_t allocateShareableBuffer(
    size_t size,
    int refcount,
    allocatorIpcDesc* ipcDesc,
    void** ptr,
    std::shared_ptr<memCacheAllocator> memCache,
    const CommLogData* logMetaData,
    const meta::comms::memtrace::MemCallsite& callsite) {
  CUmemAllocationHandleType type = ctran::utils::getCuMemAllocHandleType();
  auto commHash = (logMetaData) ? logMetaData->commHash : 0;

  // grab memory from cache if available, otherwise always allocate new memory
  CUmemGenericAllocationHandle handle;
  if (memCache) {
    std::stringstream ss;
    ss << callsite.function << ":0x" << std::hex << commHash;
    FB_COMMCHECK(memCache->getCachedCuMemById(
        ss.str(), /*key*/
        ptr,
        &handle,
        size,
        logMetaData,
        meta::comms::memtrace::MemCallsite(callsite.scope, __func__)));
  } else {
    FB_COMMCHECK(
        ctran::utils::commCuMemAlloc(
            ptr, &handle, type, size, logMetaData, callsite));
  }

  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // Return the native cuMem handle for later Export/Import via UDS
    ipcDesc->udsMemHandle = handle;
#if CUDART_VERSION >= 12040
  } else {
    CUmemFabricHandle fabricHandle;
    FB_CUCHECK(cuMemExportToShareableHandle(&fabricHandle, handle, type, 0));
    ipcDesc->fabricHandle = fabricHandle;
#endif
  }
  if (refcount) {
    ipcDesc->memHandle = handle;
    for (int r = 0; r < refcount; ++r) {
      FB_CUCHECK(cuMemRetainAllocationHandle(&handle, *ptr));
    }
  }

  // TODO: refactor this to CLOGF_SUBSYS once it supports oneof multiple masks
  CLOGF_IF(
      INFO,
      CLOGF_ENABLED(ALLOC) || CLOGF_ENABLED(P2P),
      "commHash: {:x} Allocated shareable buffer {} size {} ipcDesc {} for {}",
      commHash,
      *ptr,
      size,
      (void*)ipcDesc,
      callsite.function);

  return commSuccess;
}

}; // namespace ncclx::memory
