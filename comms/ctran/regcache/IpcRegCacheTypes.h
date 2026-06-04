// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <fmt/core.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

#include "comms/ctran/utils/CtranIpcTypes.h"

// FIXME(alvinyc): move this IB constant to CtranIbBase.h once CtranIb doesn't
// depend on CtranCtrl and CtranCtrl is removed
constexpr int CTRAN_MAX_IB_DEVICES_PER_RANK{2};

namespace ctran::regcache {

struct IBDesc {
  uint64_t remoteAddr{0};
  std::array<uint32_t, CTRAN_MAX_IB_DEVICES_PER_RANK> rkeys{};
  int nKeys{0};

  std::string toString() const {
    std::string s =
        fmt::format("[IB_EXPORT_MEM] remoteAddr: 0x{:x}", remoteAddr);
    for (int i = 0; i < nKeys; i++) {
      s += fmt::format(", rkeys[{}]: {}", i, rkeys[i]);
    }
    return s;
  }
};

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
  // Number of times this buffer was exported to the peer. The import side
  // should decrement its refcount by this amount.
  int32_t exportCount{1};

  std::string toString() const {
    std::stringstream ss;
    ss << "[IPC_RELEASE_MEM] base: " << base << " uid: " << uid
       << " exportCount: " << exportCount;
    return ss.str();
  }
};

// Maximum length for peer ID string (including null terminator)
// Format: "hostname:pid" - hostname can be up to 255 chars (DNS limit)
constexpr size_t kMaxPeerIdLen = 272;

struct IpcRemHandle {
  // use peerId, basePtr and uid on peer to lookup the imported memory handle
  // in local cache.
  char peerId[kMaxPeerIdLen]{};
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

// Unified IPC request structure sent over the network.
// Used for both memory export (IpcDesc) and release (IpcRelease) requests.
// The peer checks IpcReqType to determine which callback to invoke.
struct IpcReq {
  IpcReqType type{IpcReqType::kRelease};
  char peerId[kMaxPeerIdLen]{};
  IpcDesc desc{};
  IpcRelease release{};

  IpcReq() = default;

  explicit IpcReq(IpcReqType t, const std::string& id) : type(t) {
    // Copy peerId with bounds checking
    std::strncpy(peerId, id.c_str(), kMaxPeerIdLen - 1);
    peerId[kMaxPeerIdLen - 1] = '\0';
  }

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

} // namespace ctran::regcache
