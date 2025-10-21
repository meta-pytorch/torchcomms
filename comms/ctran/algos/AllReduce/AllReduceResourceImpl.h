// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/ctran/algos/AllReduce/AllReduceNetTypes.h"
#include "comms/ctran/algos/common/BufManager.h"
#include "comms/ctran/mapper/CtranMapper.h"

namespace ctran::algos::allreduce {
enum class AllReduceResourceBufName {

  kReduceComm, // main object for the whole algo
  kLocalPeerStructures, // Local copies of peer structures for cross-GPU access
  kDevPeers, // dev peers for intra-node

  kPostFlags, // intra-node sync flags
  kCompleteFlags, // intra-node sync flags

  kTmpbuff, // temporary buffer for intra-node

  // Tracks total number. Keep last.
  kNumBufsNames,
};

// Add to ResourceImpl.cc initialization

using AllReduceResourceBufs = ::ctran::algos::BufSnapshot<
    AllReduceResourceBufName,
    AllReduceResourceBufName::kNumBufsNames>;

struct AllReduceResourceRef {
  AllReduceResourceBufs bufs;
  AllReduceComm* allReduceComms{nullptr};
  std::vector<SendResources> sendResources;
  std::vector<RecvResources> recvResources;
};

class AllReduceResourceImpl {
 public:
  AllReduceResourceImpl(
      ncclx::CommStateX* statex,
      CtranMapper* mapper,
      CommLogData* logMetadata);
  ~AllReduceResourceImpl();

  commResult_t initAllReduceDirectResourceAsync(
      int nBlocks,
      cudaStream_t stream);

  // Release resources
  commResult_t destroy();

  // Check if the resource is initialized
  bool isInitialized() const {
    return bufMngr_ && bufMngr_->isCommitted() && bufMngr_->isExchanged();
  }

  // Get a read-only snapshot of all temporary buffers and synchronization
  // objects. The internal bufs snapshot is copied from bufMngr_ in
  // initialization, and can be directly referenced by algorithm.
  AllReduceResourceRef& getRef() {
    return ref_;
  }

 protected:
  ncclx::CommStateX* statex_{nullptr};
  CtranMapper* mapper_{nullptr};
  CommLogData* logMetaData_{nullptr};

 private:
  // Reference to persistent temporary buffers and synchronization objects
  // used in algorithm. Assigned from memory managed by bufMngr_ at
  // initialize().
  AllReduceResourceRef ref_{};

  // Base of all allocated buffers
  std::unique_ptr<::ctran::algos::BufManager<
      AllReduceResourceBufName,
      AllReduceResourceBufName::kNumBufsNames>>
      bufMngr_;
};
} // namespace ctran::algos::allreduce
