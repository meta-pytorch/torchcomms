// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/algos/common/BufManager.h"

namespace ctran::alltoallvdedup {

enum class ResourceBufName {
  kTmpSendIdx,
  kTmpNumSendIdx,
  kTmpFwdIdx,
  kTmpNumFwdIdx,
  kTmpNumIntraFwdIdx,
  kTmpRecvIdx,
  kTmpNumFwdRecvIdx,
  kTmpNumIntraRecvIdx,

  // Number of blocks accessed by GPE thread for tracking progress on host side;
  // copied from the above device buffers at prepare()
  kTmpNumSendIdxH,
  kTmpNumFwdIdxH,
  kTmpNumIntraFwdIdxH,
  kTmpNumFwdRecvIdxH,
  kTmpNumIntraRecvIdxH,

  // Temporary state objects used for combine
  kTmpSendRedStepNumPending,
  kTmpSendRedIdxSrcIds,
  kTmpSendRedIdxNumSrcs,
  kTmpSendRedIdxNumPendingSrcs,
  kTmpIntraRedStepNumPending,
  kTmpIntraRedIdxSrcIds,
  kTmpIntraRedIdxNumSrcs,
  kTmpIntraRedIdxNumPendingSrcs,
  kTmpRecvRedIdxNumSrcs,
  kTmpRecvRedIdxSrcIds,

  // Temporary send buffer on send rank for gathering blocks for RDMA. Allocated
  // in device memory.
  kTmpSendBuff,

  // Temporary forward buffer for receiving blocks from remote peers on each
  // forward rank. Allocated in device memory, and exchanged with all rail peer
  // ranks.
  kTmpFwdBuff,

  // Temporary receive buffer for receiving blocks from forwarding rank on each
  // receive rank. Allocated in device memory, and exchanged with intra-node
  // peer ranks.
  kTmpRecvBuff,
  // Similar as kTmpRecvBuff, but exclusively used for receiving from intraFwd.
  // Explicitly separate intraFwd->intraRecv and fwd->recv pipelines, to keep
  // remote step assignment simple
  kTmpIntraRecvBuff,

  // Temporary metadata buffer to compute recvOffsets and to be updated in
  // algorithm execution to track the completion of receive. Allocated in device
  // memory.
  kTmpRecvOffsets,

  kRecvStepInfo,
  kRecvStepFwdBlockIds,
  kIntraRecvStepInfo,
  kIntraRecvStepFwdBlockIds,

  kFwdStepRecvrNumBlocks,
  kFwdStepBlocksIds,
  kFwdStepNumBlocks,

  // Synchronization objects used between GPE thread and kernel. Allocated in
  // host pinned memory.
  kSendGKSyncs,
  kRecvGKSyncs,
  kIntraFwdGKSyncs,
  kRecvCopyGKSyncs,
  kIntraRecvCopyGKSyncs,

  // Synchronization objects used between forward rank and receive rank in
  // kernel. Allocated in device memory, and exchanged with intra-node peer
  // ranks.
  kFwdRecvSync,
  kIntraFwdRecvSync,

  // Synchronization objects used between sendRed and intraRed workers in the
  // same GPU. Allocated in device memory.
  kIntraRedSync,

  // Synchronization objects used within each worker group. Allocated in device
  // and accessed by all workers in the group.
  kWorkerGroupSync,

  // Tracks total number. Keep last.
  kNumBufsNames,
};

using algos::bufmanager::MemType;
using algos::bufmanager::RegBuf;
using algos::bufmanager::RemRegBuf;
using ResourceBufs = ::ctran::algos::
    BufSnapshot<ResourceBufName, ResourceBufName::kNumBufsNames>;

struct ResourceBufMd {
  MemType memType{MemType::kDevice};
  size_t buflen{0};
  const std::string str;

 public:
  ResourceBufMd(MemType memType, size_t buflen, std::string_view str)
      : memType(memType), buflen(buflen), str(str) {}
};

struct ResourceRef {
  ResourceBufs bufs;

  template <typename T>
  inline void
  getBufPtr(const ResourceBufName name, const bool useReg, T*& ptr) {
    auto p = useReg ? bufs.getRegBuf(name).ptr : bufs.getBuf(name).ptr;
    ptr = reinterpret_cast<T*>(p);
  }

  template <typename T>
  inline void getRemBufPtrs(const ResourceBufName name, T* ptrs[]) {
    const auto& remBufs = bufs.getRemRegBufs(name);
    for (auto i = 0; i < remBufs.size(); i++) {
      ptrs[i] = static_cast<T*>(remBufs.at(i).ptr);
    }
  }

  inline RegBuf& getBuf(const ResourceBufName name) {
    return bufs.getRegBuf(name);
  }

  inline std::vector<RemRegBuf>& getRemBufs(const ResourceBufName name) {
    return bufs.getRemRegBufs(name);
  }

  // Sync objects
  KernSync kSync;
};

#define GET_RESOURCE_REGBUFPTR(res, bufname, ptr) \
  (res)->getBufPtr(ResourceBufName::bufname, true, ptr);

// Optional to get buffers without registration and exchange, specified by
// useReg. Used only for common path with prepare UT where we don't want to
// create a communicator.
#define GET_RESOURCE_BUFPTR(res, bufname, useReg, ptr) \
  (res)->getBufPtr(ResourceBufName::bufname, useReg, ptr);

#define GET_RESOURCE_REM_BUFPTRS(res, bufname, ptrs) \
  (res)->getRemBufPtrs(ResourceBufName::bufname, ptrs);

} // namespace ctran::alltoallvdedup
