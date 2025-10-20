// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/algos/common/BufManager.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"

namespace ctran::alltoallvdedup {

enum class ResourceBufName {
  // Temporary send buffer on send rank for gathering blocks for RDMA. Allocated
  // in device memory.
  kTmpSendBuff,

  kTmpSendIdx,
  kTmpIntraFwdIdx,

  // Temporary forward buffer for receiving blocks from remote peers on each
  // forward rank. Allocated in device memory, and exchanged with all rail peer
  // ranks.
  kTmpFwdBuff,

  // Temporary receive buffer for receiving blocks from forwarding rank on each
  // receive rank. Allocated in device memory, and exchanged with intra-node
  // peer ranks.
  kTmpRecvBuff,

  // used for pytorch metadata
  kLocalOutputSplits,
  kLocalOutputSplitsH,
  kRankBitmaps,
  kRankBitmapsH,

  // Temporary metadata buffer to compute numRecvBlocks. Allocated in device
  // memory, and exchanged with intra-node peer ranks.
  kTmpNumRecvBlocksBuff,

  // Copy of kTmpNumRecvBlocksBuff to be accessed by GPE thread. Used to fill
  // localOutputSplit
  kTmpNumRecvBlocksBuffH,

  // Temporary metadata buffer to compute numSendBlocks. Allocated in
  // host-pinned memory, and exchanged with rail peer ranks.
  kTmpNumSendBlocksBuffH,

  // Temporary metadata buffer to compute numForwardBlocks for GPE thread to
  // access. Allocated in host-pinned memory.
  kNumForwardBlocksH,

  // Temporary metadata buffer to copy from blockRecvBuckets (user input) for
  // GPE thread to access. Allocated in host-pinned memory.
  kBlockRecvBucketsH,

  // Temporary metadata buffer to compute recvOffsets and to be updated in
  // algorithm execution to track the completion of receive. Allocated in device
  // memory.
  kTmpRecvOffsets,

  // Synchronization objects used between GPE thread and kernel. Allocated in
  // host pinned memory.
  kGpeKernelSyncs,

  // Synchronization objects used between forward rank and receive rank in
  // kernel. Allocated in device memory, and exchanged with intra-node peer
  // ranks.
  kFwdRecvSync,

  // Synchronization objects used within forward rank kernel. Allocated in
  // device memory.
  kFwdGroupSync,

  kWorkerSync,

  // Tracks total number. Keep last.
  kNumBufsNames,
};

using ResourceBufs = ::ctran::algos::
    BufSnapshot<ResourceBufName, ResourceBufName::kNumBufsNames>;
using algos::bufmanager::RegBuf;
using algos::bufmanager::RemRegBuf;

struct ResourceRef {
  ResourceBufs bufs;

  template <typename T>
  inline bool getBufPtr(const ResourceBufName name, T*& ptr) {
    auto p = bufs.getRegBuf(name).ptr;
    if (p == nullptr) {
      return false;
    }
    ptr = reinterpret_cast<T*>(p);
    return true;
  }

  template <typename T>
  inline bool getRemBufPtrs(
      const ResourceBufName name,
      const size_t nPeers,
      std::vector<T*>& ptrs) {
    const auto& remBufs = bufs.getRemRegBufs(name);
    if (nPeers != remBufs.size()) {
      return false;
    }
    ptrs.reserve(nPeers);
    for (auto& remBuf : remBufs) {
      ptrs.push_back(reinterpret_cast<T*>(remBuf.ptr));
    }
    return true;
  }

  inline RegBuf& getBuf(const ResourceBufName name) {
    return bufs.getRegBuf(name);
  }

  inline std::vector<RemRegBuf>& getRemBufs(const ResourceBufName name) {
    return bufs.getRemRegBufs(name);
  }

  // Sync objects used by host
  algos::GpeKernelSync* prepareGKSync;
  std::vector<algos::GpeKernelSync*> sendCopyGKSyncs;
  std::vector<algos::GpeKernelSync*> recvFwdGKSyncs;
  std::vector<algos::GpeKernelSync*> recvCopyGKSyncs;

  // Sync objects used by device
  KernSync kSync;
};

#define GET_RESOURCE_BUFPTR(res, bufname, ptr)                  \
  do {                                                          \
    auto ret = (res)->getBufPtr(ResourceBufName::bufname, ptr); \
    FB_CHECKABORT(ret, "Failed to get {}", ARGTOSTR(bufname));  \
  } while (0)

#define GET_RESOURCE_REM_BUFPTRS(res, bufname, nPeers, ptrs)                 \
  do {                                                                       \
    auto ret = (res)->getRemBufPtrs(ResourceBufName::bufname, nPeers, ptrs); \
    FB_CHECKABORT(ret, "Failed to get remote {}", ARGTOSTR(bufname));        \
  } while (0)

class ResourceImpl {
 public:
  ResourceImpl(
      ncclx::CommStateX* statex,
      CtranMapper* mapper,
      CommLogData* logMetadata);
  ~ResourceImpl();

  // Initialize resources (e.g., temporary buffers) based on persistent
  // argument and configs
  commResult_t initialize(
      const PersistArgs& args,
      const PersistConfig& config,
      cudaStream_t stream);

  // Release resources
  commResult_t destroy();

  // Get a read-only snapshot of all temporary buffers and synchronization
  // objects. The internal bufs snapshot is copied from bufMngr_ in
  // initialization, and can be directly referenced by algorithm.
  ResourceRef& getRef() {
    return ref_;
  }

  friend class AlgoImpl;

 protected:
  ncclx::CommStateX* statex_{nullptr};
  CtranMapper* mapper_{nullptr};
  CommLogData* logMetaData_{nullptr};

 private:
  // Reference to persistent temporary buffers and synchronization objects
  // used in algorithm. Assigned from memory managed by bufMngr_ at
  // initialize().
  ResourceRef ref_;

  commResult_t exchange();
  // Set buffer snapshots and initialize sync objects after initialize and
  // exchange
  commResult_t setRef(const PersistConfig& config, cudaStream_t stream);

  // Base of all allocated buffers
  std::unique_ptr<::ctran::algos::BufManager<
      ResourceBufName,
      ResourceBufName::kNumBufsNames>>
      bufMngr_;
};

} // namespace ctran::alltoallvdedup
