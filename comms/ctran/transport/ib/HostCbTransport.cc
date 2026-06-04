// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/transport/ib/HostCbTransport.h"

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/ctran/utils/PinnedHostPool.h"

namespace ctran::transport::ib {

// ═══════════════════════════════════════════════════════════
// Constructor / Destructor
// ═══════════════════════════════════════════════════════════

HostCbTransport::HostCbTransport(
    int peerRank,
    CtranIb* ctranIb,
    GpeKernelSyncPool* pool,
    int pipelineDepth,
    size_t chunkSize,
    int myRank,
    int cudaDev,
    const CommLogData* logMetaData)
    : peerRank_(peerRank),
      myRank_(myRank),
      cudaDev_(cudaDev),
      ctranIb_(ctranIb),
      pipelineDepth_(std::min(pipelineDepth, kMaxPipelineDepth)),
      chunkSize_(chunkSize),
      logMetaData_(logMetaData),
      gpeKernelSyncPool_(pool) {
  if (ctranIb_ == nullptr) {
    throw ctran::utils::Exception(
        "HostCbTransport: ctranIb must not be null", commInternalError);
  }
  if (gpeKernelSyncPool_ == nullptr) {
    throw ctran::utils::Exception(
        "HostCbTransport: GpeKernelSyncPool* must not be null",
        commInternalError);
  }

  const size_t stagingSize = pipelineDepth_ * chunkSize_;

  // Use ctran::utils allocator wrappers so all CB allocations show up
  // in NCCL memtrace; pass logMetaData_ for callsite attribution.
  FB_COMMCHECKTHROW_EX_NOCOMM(
      ctran::utils::commCudaMalloc(
          &sendStaging_, stagingSize, logMetaData_, "HostCbTransport.send"));
  FB_COMMCHECKTHROW_EX_NOCOMM(
      ctran::utils::commCudaMalloc(
          &recvStaging_, stagingSize, logMetaData_, "HostCbTransport.recv"));
  FB_COMMCHECKTHROW_EX_NOCOMM(
      CtranIb::regMem(
          sendStaging_, stagingSize, cudaDev_, &sendStagingRegElem_));
  FB_COMMCHECKTHROW_EX_NOCOMM(
      CtranIb::regMem(
          recvStaging_, stagingSize, cudaDev_, &recvStagingRegElem_));

  // Pop pipelineDepth_ slot-syncs each for send and recv from the
  // borrowed pool. The pool is owned by CtranGpe; we return entries
  // via reset() in the destructor. We follow the same nworkers-then-
  // resetStatus() pattern as `allocGpeKernelSyncs` so that the full
  // postFlag/completeFlag arrays are initialized to kUnset (the
  // pool's onPop() resetStatus runs before nworkers is known and so
  // can leave higher indices uninitialized).
  for (int s = 0; s < pipelineDepth_; ++s) {
    auto* sendG = gpeKernelSyncPool_->pop();
    FB_CHECKABORT(
        sendG != nullptr, "HostCbTransport: GpeKernelSyncPool exhausted");
    sendG->nworkers = CTRAN_ALGO_MAX_THREAD_BLOCKS;
    sendG->resetStatus();
    sendSyncs_.push_back(sendG);

    auto* recvG = gpeKernelSyncPool_->pop();
    FB_CHECKABORT(
        recvG != nullptr, "HostCbTransport: GpeKernelSyncPool exhausted");
    recvG->nworkers = CTRAN_ALGO_MAX_THREAD_BLOCKS;
    recvG->resetStatus();
    recvSyncs_.push_back(recvG);
  }

  FB_COMMCHECKTHROW_EX_NOCOMM(
      ctran::utils::commCudaHostAlloc(
          &remoteReady_,
          pipelineDepth_,
#if defined(__HIP_PLATFORM_AMD__)
          hipHostMallocDefault,
#else
          cudaHostAllocDefault,
#endif
          logMetaData_,
          "HostCbTransport.remoteReady"));
  memset(remoteReady_, 0, pipelineDepth_ * sizeof(uint64_t));
  FB_COMMCHECKTHROW_EX_NOCOMM(
      CtranIb::regMem(
          remoteReady_,
          pipelineDepth_ * sizeof(uint64_t),
          cudaDev_,
          &remoteReadyRegElem_));

  FB_COMMCHECKTHROW_EX_NOCOMM(
      ctran::utils::commCudaHostAlloc(
          &fetchAddDiscardBuf_,
          1,
#if defined(__HIP_PLATFORM_AMD__)
          hipHostMallocDefault,
#else
          cudaHostAllocDefault,
#endif
          logMetaData_,
          "HostCbTransport.fetchAddDiscard"));
  *fetchAddDiscardBuf_ = 0;
  FB_COMMCHECKTHROW_EX_NOCOMM(
      CtranIb::regMem(
          fetchAddDiscardBuf_,
          sizeof(uint64_t),
          cudaDev_,
          &fetchAddDiscardRegElem_));

  // One-shot VC rendezvous + storage. Must run exactly once per
  // transport instance. After this, hot paths use impl::checkValidVc
  // as a defense-in-depth precondition.
  impl::connectAndPopulateVcs(ctranIb_, peerRank_, myRank_, vcs_);
}

HostCbTransport::~HostCbTransport() {
  if (devTransport_) {
    (void)ctran::utils::commCudaFree(devTransport_, logMetaData_);
    devTransport_ = nullptr;
  }
  if (fetchAddDiscardRegElem_) {
    CtranIb::deregMem(fetchAddDiscardRegElem_);
  }
  if (fetchAddDiscardBuf_) {
    (void)ctran::utils::commCudaFreeHost(fetchAddDiscardBuf_, logMetaData_);
  }
  if (remoteReadyRegElem_) {
    CtranIb::deregMem(remoteReadyRegElem_);
  }
  if (remoteReady_) {
    (void)ctran::utils::commCudaFreeHost(remoteReady_, logMetaData_);
  }
  for (auto* s : recvSyncs_) {
    if (s) {
      s->reset();
    }
  }
  recvSyncs_.clear();
  for (auto* s : sendSyncs_) {
    if (s) {
      s->reset();
    }
  }
  sendSyncs_.clear();
  if (recvStagingRegElem_) {
    CtranIb::deregMem(recvStagingRegElem_);
  }
  if (sendStagingRegElem_) {
    CtranIb::deregMem(sendStagingRegElem_);
  }
  if (recvStaging_) {
    (void)ctran::utils::commCudaFree(recvStaging_, logMetaData_);
  }
  if (sendStaging_) {
    (void)ctran::utils::commCudaFree(sendStaging_, logMetaData_);
  }
}

// ═══════════════════════════════════════════════════════════
// getDeviceTransport
// ═══════════════════════════════════════════════════════════

HostTransportDev* HostCbTransport::getDeviceTransport() {
  if (devTransport_) {
    return devTransport_;
  }

  HostTransportDev hostCopy{};
  hostCopy.pipelineDepth = pipelineDepth_;
  hostCopy.chunkSize = chunkSize_;

  for (int s = 0; s < pipelineDepth_; ++s) {
    hostCopy.sendChunks[s] = DeviceChunkDesc{
        .sync = sendSyncs_[s],
        .stagingSlot = sendStaging_ + static_cast<size_t>(s) * chunkSize_,
        .chunkSize = chunkSize_,
    };
    hostCopy.recvChunks[s] = DeviceChunkDesc{
        .sync = recvSyncs_[s],
        .stagingSlot = recvStaging_ + static_cast<size_t>(s) * chunkSize_,
        .chunkSize = chunkSize_,
    };
  }

  FB_COMMCHECKTHROW_EX_NOCOMM(
      ctran::utils::commCudaMalloc(
          &devTransport_, 1, logMetaData_, "HostCbTransport.devTransport"));
#if defined(__HIP_PLATFORM_AMD__)
  FB_CUDACHECKTHROW_EX_NOCOMM(hipMemcpy(
      devTransport_,
      &hostCopy,
      sizeof(HostTransportDev),
      hipMemcpyHostToDevice));
#else
  FB_CUDACHECKTHROW_EX_NOCOMM(cudaMemcpy(
      devTransport_,
      &hostCopy,
      sizeof(HostTransportDev),
      cudaMemcpyHostToDevice));
#endif

  return devTransport_;
}

void HostCbTransport::setKernelNumBlocks(int sendNumBlocks, int recvNumBlocks) {
  // Reset status after lowering nworkers to ensure the previously
  // active range [nworkers, prevNworkers) doesn't carry stale
  // postFlag/completeFlag values into a fresh round.
  if (sendNumBlocks > 0) {
    for (auto* s : sendSyncs_) {
      if (s) {
        s->nworkers = sendNumBlocks;
        s->resetStatus();
      }
    }
  }
  if (recvNumBlocks > 0) {
    for (auto* s : recvSyncs_) {
      if (s) {
        s->nworkers = recvNumBlocks;
        s->resetStatus();
      }
    }
  }
}

} // namespace ctran::transport::ib
