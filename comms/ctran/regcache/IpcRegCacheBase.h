// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <folly/Synchronized.h>
#include "comms/ctran/regcache/IpcRegCacheTypes.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/utils/commSpecs.h"

namespace ctran {
namespace regcache {

struct IpcRegElem {
  // User passed addr, size at ncclCommRegister
  const void* buf{nullptr};
  const size_t len{0};
  // unique ID for tracking registrations
  const uint32_t uid{0};
  folly::Synchronized<ctran::utils::CtranIpcMem> ipcMem;

 public:
  IpcRegElem(const void* buf, const size_t len, int cudaDev, uint32_t uid)
      : buf(buf),
        len(len),
        uid(uid),
        ipcMem(ctran::utils::CtranIpcMem(cudaDev, "IPC RegElem")) {};
  ~IpcRegElem() {};

  commResult_t tryLoad(bool& supported, bool shouldSupportCudaMalloc) {
    return ipcMem.wlock()->tryLoad(
        buf, len, supported, shouldSupportCudaMalloc);
  }

  std::string toString() const {
    return fmt::format(
        "buf: {}, len: {}, uid: {}, ipcMem: {}",
        buf,
        len,
        uid,
        ipcMem.rlock()->toString());
  }
};

struct IpcRemRegElem {
  ctran::utils::CtranIpcRemMem ipcRemMem;
  // Reference count for how many communicators have imported this memory.
  // Starts at 1 on first import, incremented on subsequent cache hits.
  // Only freed when refCount reaches 0.
  std::atomic<int> refCount{1};

 public:
  IpcRemRegElem(
      const ctran::utils::CtranIpcDesc& ipcDesc,
      int cudaDev,
      const struct CommLogData* logMetaData)
      : ipcRemMem(ipcDesc, cudaDev, logMetaData, "IPC RemRegElem", {}) {};

  IpcRemRegElem(
      const ctran::utils::CtranIpcDesc& ipcDesc,
      int cudaDev,
      const struct CommLogData* logMetaData,
      const std::vector<ctran::utils::CtranIpcSegDesc>& extraSegments)
      : ipcRemMem(
            ipcDesc,
            cudaDev,
            logMetaData,
            "IPC RemRegElem",
            extraSegments) {};

  std::string toString() const {
    return fmt::format(
        "{} refCount: {}",
        ipcRemMem.toString(),
        refCount.load(std::memory_order_relaxed));
  }
};

// Forward declaration for RegElem (defined in RegCache.h)
struct RegElem;

// Abstract interface for any object that exports IPC memory and needs
// to send remReleaseMem when memory is globally freed. Implementers
// (e.g., CtranMapper) register with IpcRegCache so that globalDeregister
// can iterate all active exporters.
class IpcExportClient {
 public:
  virtual ~IpcExportClient() = default;

  // Called by IpcRegCache::releaseFromAllClients when memory is globally freed.
  // The implementer should look up the regElem in its own export cache,
  // send release to the appropriate peers, and clean up.
  virtual commResult_t remReleaseMem(RegElem* regElem) = 0;
};

} // namespace regcache
} // namespace ctran
