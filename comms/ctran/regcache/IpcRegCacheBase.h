// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <fmt/core.h>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <folly/Synchronized.h>
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/utils/commSpecs.h"

namespace ctran {
namespace regcache {

struct IpcDesc {
  ctran::utils::CtranIpcDesc desc;
  // offset since the base of desc
  size_t offset{0};
  // unique ID for tracking registrations
  uint32_t uid{0};

  std::string toString() const {
    return fmt::format(
        "[IPC_MEM_DESC] offset: 0x{:x} uid: {} {}",
        offset,
        uid,
        desc.toString());
  }
};

struct IpcRelease {
  void* base{nullptr};
  // unique ID for tracking registrations
  uint32_t uid{0};

  std::string toString() const {
    std::stringstream ss;
    ss << "[IPC_RELEASE_MEM] base: " << base << " uid: " << uid;
    return ss.str();
  }
};

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
  // use peerId, basePtr and uid on peer to lookup the imported memory handle
  // in local cache
  std::string peerId;
  void* basePtr;
  uint32_t uid;

  std::string toString() const {
    return fmt::format(
        "peerId: {}, basePtr: {}, uid: {}", peerId, basePtr, uid);
  }
};

// Type of IPC request
enum class IpcReqType : uint8_t {
  kDesc = 0, // Memory descriptor for export
  kRelease = 1, // Release notification
};

// Maximum length for peer ID string (including null terminator)
// Format: "hostname:pid" - hostname can be up to 255 chars (DNS limit)
constexpr size_t kMaxPeerIdLen = 272;

// Unified IPC request structure sent over the network.
// Used for both memory export (IpcDesc) and release (IpcRelease) requests.
// The peer checks IpcReqType to determine which callback to invoke.
struct IpcReq {
  IpcReqType type{IpcReqType::kRelease};
  char peerId[kMaxPeerIdLen]{};
  union {
    IpcDesc desc;
    IpcRelease release;
  };

  IpcReq() : release() {}

  explicit IpcReq(IpcReqType t, const std::string& id) : type(t) {
    // Copy peerId with bounds checking
    std::strncpy(peerId, id.c_str(), kMaxPeerIdLen - 1);
    peerId[kMaxPeerIdLen - 1] = '\0';

    if (t == IpcReqType::kDesc) {
      new (&desc) IpcDesc();
    } else {
      new (&release) IpcRelease();
    }
  }

  ~IpcReq() {}

  std::string getPeerId() const {
    return std::string(peerId);
  }

  std::string toString() const {
    if (type == IpcReqType::kDesc) {
      return fmt::format(
          "[IpcReq] type: DESC, peerId: {}, {}", peerId, desc.toString());
    } else {
      return fmt::format(
          "[IpcReq] type: RELEASE, peerId: {}, {}", peerId, release.toString());
    }
  }
};

// Callback tracking structure for async IPC requests.
// Used on the sender side to track whether the request send has completed.
struct IpcReqCb {
  IpcReq req;
  std::atomic<bool> completed{false};

  IpcReqCb() = default;
  explicit IpcReqCb(IpcReqType t, const std::string& id) : req(t, id) {}
};

// Class to cache exported IpcRegElems and track which peers they are exported
// to. Moved from mapper to centralize IPC export tracking.
class IpcExportCache {
 private:
  std::unordered_map<IpcRegElem*, std::unordered_set<std::string>> map_;

 public:
  // Record the ipcRegElem and the peerId to export to.
  inline void record(IpcRegElem* ipcRegElem, const std::string& peerId) {
    map_[ipcRegElem].insert(peerId);
  }

  // Remove the specified ipcRegElem from cache. Return the exported peerIds set
  // if the ipcRegElem has been exported. Otherwise, empty set is returned.
  std::unordered_set<std::string> remove(IpcRegElem* ipcRegElem);

  // Dump a full copy of the cache map; used for testing only
  std::unordered_map<IpcRegElem*, std::unordered_set<std::string>> dump() const;
};

} // namespace regcache
} // namespace ctran
