// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <optional>
#include <string>

#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ncclx::memory {

class SlabAllocator;
class memCacheAllocator;

constexpr size_t kMinAlignSize = 16;

#if CUDART_VERSION >= 12010
static inline CUmulticastGranularity_flags adjustCuMulticastGran() {
  return NCCL_MEM_ENABLE_MC_ALIGNMENT ? CU_MULTICAST_GRANULARITY_RECOMMENDED
                                      : CU_MULTICAST_GRANULARITY_MINIMUM;
}
#endif

struct allocatorIpcDesc {
#if CUDART_VERSION >= 12040
  std::optional<CUmemFabricHandle> fabricHandle{std::nullopt};
#endif
  std::optional<CUmemGenericAllocationHandle> memHandle{std::nullopt};
  std::optional<CUmemGenericAllocationHandle> udsMemHandle{std::nullopt};
};

commResult_t cudaCallocAsync(
    void** ptr,
    size_t nBytes,
    cudaStream_t stream,
    const CommLogData* logMetaData,
    const char* callsite,
    SlabAllocator* allocator);

// Similar to ncclP2pAllocateShareableBuffer in p2p.cc, but customized for
// using ncclx memory cache and allocator, and only support CUMEM allocation
commResult_t allocateShareableBuffer(
    size_t size,
    int refcount,
    allocatorIpcDesc* ipcDesc,
    void** ptr,
    std::shared_ptr<memCacheAllocator> memCache,
    const CommLogData* logMetaData,
    const char* use);

// wrapper function with a communicator to use slab allocator associated
// with it, keep this function definition in header file to avoid explicit
// instantiation
template <typename T>
commResult_t cudaCallocAsync(
    T** ptr,
    size_t numElems,
    cudaStream_t stream,
    const CommLogData* logMetaData,
    const char* callsite,
    SlabAllocator* allocator) {
  return ncclx::memory::cudaCallocAsync(
      (void**)ptr,
      numElems * sizeof(T),
      stream,
      logMetaData,
      callsite,
      allocator);
}

// wrapper function without a communicator (default to nullptr), keep this
// function definition in header file to avoid explicit instantiation
template <typename T>
commResult_t cudaCallocAsync(
    T** ptr,
    size_t numElems,
    cudaStream_t stream,
    const CommLogData* logMetaData,
    const char* callsite) {
  return ncclx::memory::cudaCallocAsync(
      (void**)ptr,
      numElems * sizeof(T),
      stream,
      logMetaData,
      callsite,
      nullptr);
}

/* helper function to generate a string for memory logging, the current format
 * is "<prefix>:<channelId>/<connIndex>/<peerRank>". where prefix consists of
 * p2p, Send/Recv and callsite provided by caller, e.g.,
 * p2pSendProxySetup:0/0/1,
 */
inline std::string genKey(
    std::string_view callsite,
    bool isP2p,
    bool isSend,
    int channelId,
    int connIndex,
    int peerRank) {
  return fmt::format(
      "{}{}{}:{}/{}/{}",
      isP2p ? "p2p" : "",
      isSend ? "Send" : "Recv",
      callsite,
      channelId,
      connIndex,
      peerRank);
}

}; // namespace ncclx::memory
