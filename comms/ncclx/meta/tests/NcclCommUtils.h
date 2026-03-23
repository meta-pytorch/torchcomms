// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/testinfra/ITestBootstrap.h"
#include "nccl.h" // @manual

namespace ncclx::test {

// Create a NCCL communicator using ITestBootstrap for ncclUniqueId broadcast.
// Generates a unique commDesc per call to avoid stale TCPStore keys when
// fast-init mode reuses the same TCPStore across multiple test cases.
// Only overrides commDesc when the caller didn't explicitly set it.
ncclComm_t createNcclComm(
    int globalRank,
    int numRanks,
    int devId,
    meta::comms::ITestBootstrap* bootstrap,
    bool isMock = false,
    ncclConfig_t* customConfig = nullptr);

// RAII wrapper for ncclComm_t created via createNcclComm with bootstrap.
// Automatically calls ncclCommDestroy in destructor.
class NcclCommRAII {
 public:
  NcclCommRAII(
      int globalRank,
      int numRanks,
      int localRank,
      meta::comms::ITestBootstrap* bootstrap,
      bool isMock = false,
      ncclConfig_t* config = nullptr);

  ~NcclCommRAII();

  NcclCommRAII(const NcclCommRAII&) = delete;
  NcclCommRAII& operator=(const NcclCommRAII&) = delete;

  ncclComm& operator*();
  ncclComm_t operator->() const;
  ncclComm_t get() const;
  operator ncclComm_t() const;

 private:
  ncclComm_t comm_;
};

// RAII wrapper for ncclComm_t created via ncclCommSplit.
// Does not call finalizeNcclComm since the parent comm handles that.
class NcclCommSplitRAII {
 public:
  NcclCommSplitRAII(
      ncclComm_t parentComm,
      int color,
      int key,
      ncclConfig_t* config = nullptr);

  ~NcclCommSplitRAII();

  NcclCommSplitRAII(const NcclCommSplitRAII&) = delete;
  NcclCommSplitRAII& operator=(const NcclCommSplitRAII&) = delete;

  ncclComm& operator*();
  ncclComm_t operator->() const;
  ncclComm_t get() const;
  operator ncclComm_t() const;

 private:
  ncclComm_t comm_;
};

} // namespace ncclx::test
