// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <fmt/core.h>
#include <string>

#include <folly/Synchronized.h>
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/utils/commSpecs.h"

namespace ctran {
namespace regcache {

struct IpcDesc {
  ctran::utils::CtranIpcDesc desc;
  // offset since the base of desc
  size_t offset{0};

  std::string toString() const {
    return fmt::format(
        "[IPC_MEM_DESC] offset: 0x{:x} {}", offset, desc.toString());
  }
};

struct IpcRelease {
  void* base{nullptr};

  std::string toString() const {
    std::stringstream ss;
    ss << "[IPC_RELEASE_MEM] base: " << base;
    return ss.str();
  }
};

struct IpcRegElem {
  // User passed addr, size at ncclCommRegister
  const void* buf{nullptr};
  const size_t len{0};
  folly::Synchronized<ctran::utils::CtranIpcMem> ipcMem;

 public:
  IpcRegElem(const void* buf, const size_t len, int cudaDev)
      : buf(buf),
        len(len),
        ipcMem(ctran::utils::CtranIpcMem(cudaDev, "IPC RegElem")) {};
  ~IpcRegElem() {};

  commResult_t tryLoad(bool& supported, bool shouldSupportCudaMalloc) {
    return ipcMem.wlock()->tryLoad(
        buf, len, supported, shouldSupportCudaMalloc);
  }

  std::string toString() const {
    return fmt::format(
        "buf: {}, len: {}, ipcMem: {}", buf, len, ipcMem.rlock()->toString());
  }
};

struct IpcRemRegElem {
  ctran::utils::CtranIpcRemMem ipcRemMem;

 public:
  IpcRemRegElem(
      const ctran::utils::CtranIpcDesc& ipcDesc,
      int cudaDev,
      const struct CommLogData* logMetaData)
      : ipcRemMem(ipcDesc, cudaDev, logMetaData, "IPC RemRegElem") {};

  std::string toString() const {
    return ipcRemMem.toString();
  }
};

struct IpcRemHandle {
  // use peerId and basePtr on peer to lookup the imported memory handle
  // in local cache
  std::string peerId;
  void* basePtr;

  std::string toString() const {
    return fmt::format("peerId: {}, basePtr: {}", peerId, basePtr);
  }
};

} // namespace regcache
} // namespace ctran
