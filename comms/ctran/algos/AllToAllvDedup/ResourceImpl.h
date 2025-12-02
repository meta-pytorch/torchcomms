// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>
#include "comms/ctran/algos/AllToAllvDedup/ResourceBuf.h"
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/algos/common/BufManager.h"
#include "comms/ctran/mapper/CtranMapper.h"

namespace ctran::alltoallvdedup {

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
      cudaStream_t stream,
      const bool skipRem = false);

  // Release resources
  commResult_t destroy();

  // Get a read-only snapshot of all temporary buffers and synchronization
  // objects. The internal bufs snapshot is copied from bufMngr_ in
  // initialization, and can be directly referenced by algorithm.
  ResourceRef& getRef() {
    return ref_;
  }

  // Optionally set skipRem = true if called by prepare UT to skip remote buffer
  // assignment
  void assignToKernArgs(ExecKernArgs& kernArgs, const bool skipRem = false);

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

  // exchange buffer registration
  commResult_t exchange();

  // set buffer snapshots for quick access at kernel launch
  void setRef();
  void setRegRef();
  void setRemRef(std::vector<int>& localPeers, std::vector<int>& railPeers);

  // initialize sync objects
  // skipRem: skip remote ref in prepare-only UT
  commResult_t setKSync(cudaStream_t stream, const bool skipRem = false);

  void initBufMd(
      const PersistArgs& args,
      const PersistConfig& config,
      const int nNodes,
      const int nLocalRanks);

  // Base of all allocated buffers
  std::unique_ptr<::ctran::algos::BufManager<
      ResourceBufName,
      ResourceBufName::kNumBufsNames>>
      bufMngr_;
  std::unordered_map<ResourceBufName, ResourceBufMd> bufMdMap_;
};

} // namespace ctran::alltoallvdedup
