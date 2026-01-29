// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_CTRL_H
#define CTRAN_CTRL_H

#include <comms/utils/cvars/nccl_cvars.h>
#include <fmt/format.h>
#include <cstddef>
#include <functional>
#include <optional>
#include <sstream>
#include <unordered_map>

#include "comms/ctran/backends/CtranAux.h"
#include "comms/ctran/regcache/IpcRegCacheBase.h"
#include "comms/ctran/utils/CtranIpc.h"

constexpr int CTRAN_MAX_IB_DEVICES_PER_RANK{2};

/**
 * Define all control message types used in CTran backends.
 *
 * Support two protocols to transfer control message:
 * 1. Explicitly exchanged bewteen two sides via a control message channel's
 *    send/recv API. This can be used when need explicit synchronization (e.g.,
 *    handshake in zero-copy algorithms)
 * 2. Sent from one side via the send API, and the remote side handles it
 *    implicitly via pre-registered callback. The callback will be called by the
 *    control message channel whenever progressed. It is useful for asynchronous
 *    request where the sender doesn't need blockingly wait for
 *    the ack from receiver. The callback must be pre-registered to
 *    CtranCtrlManager.
 *
 * All control message types and packet format must be defined in this header
 * file for centralized management.
 */
enum ControlMsgType {
  NVL_EXPORT_MEM = 1,
  NVL_RELEASE_MEM = 2,
  IB_EXPORT_MEM = 3,
  SYNC = 4,
  UNSPECIFIED /* for receiving any type */
};

constexpr const char* ControlMsgTypeToString(ControlMsgType type) {
  switch (type) {
    case NVL_EXPORT_MEM:
      return "NVL_EXPORT_MEM";
    case NVL_RELEASE_MEM:
      return "NVL_RELEASE_MEM";
    case IB_EXPORT_MEM:
      return "IB_EXPORT_MEM";
    case SYNC:
      return "SYNC";
    case UNSPECIFIED:
      return "UNSPECIFIED";
    default:
      return "Unknown";
  }
}

template <>
struct fmt::formatter<ControlMsgType> : fmt::formatter<const char*> {
  template <typename FormatContext>
  auto format(ControlMsgType type, FormatContext& ctx) const {
    return fmt::formatter<const char*>::format(
        ControlMsgTypeToString(type), ctx);
  }
};

struct CtranIbConfig {
  int numQps{NCCL_CTRAN_IB_MAX_QPS};
  size_t qpScalingTh{NCCL_CTRAN_IB_QP_SCALING_THRESHOLD};
  enum NCCL_CTRAN_IB_VC_MODE vcMode { NCCL_CTRAN_IB_VC_MODE::spray };
  int qpMsgs{static_cast<int>(NCCL_CTRAN_IB_QP_MAX_MSGS)};
  int64_t trafficClass{NCCL_IB_TC};
};

struct CmsgIbExportMem {
  uint64_t remoteAddr{0};
  std::array<uint32_t, CTRAN_MAX_IB_DEVICES_PER_RANK> rkeys{};
  int nKeys{0};

  static const std::string name;

  CmsgIbExportMem() {};
  std::string toString() const {
    std::stringstream ss;
    ss << "[" << name << "] remoteAddr: 0x" << std::hex << remoteAddr;
    for (int i = 0; i < nKeys; i++) {
      ss << ", rkeys[" << i << "]: " << std::dec << rkeys[i];
    }
    return ss.str();
  }
};

struct CmsgNvlReleaseMem {
  void* base{nullptr};

  static const std::string name;

  CmsgNvlReleaseMem() {};
  std::string toString() const {
    std::stringstream ss;
    ss << "[" << name << "] base: " << base;
    return ss.str();
  }
};

/**
 * Packet structure of control message transferred by underlying backend.
 */
struct ControlMsg {
  int type{ControlMsgType::UNSPECIFIED};
  union {
    struct ctran::regcache::IpcDesc nvlDesc;
    struct CmsgNvlReleaseMem nvlRls;
    struct CmsgIbExportMem ibExp;
  };

  AuxData_t<DefaultAuxType> aux; // Used to store the remote aux data

  ControlMsg() {};
  ControlMsg(int type) : type(type) {
    setType(type);
  };
  ~ControlMsg() {};

  inline void setType(int newType) {
    type = newType;
    // Initialization
    switch (type) {
      case ControlMsgType::NVL_EXPORT_MEM:
        nvlDesc = ctran::regcache::IpcDesc{};
        break;
      case ControlMsgType::NVL_RELEASE_MEM:
        nvlRls = CmsgNvlReleaseMem{};
        break;
      case ControlMsgType::IB_EXPORT_MEM:
        ibExp = CmsgIbExportMem{};
        break;
      default:
        break;
    }
  }

  std::string toString() const {
    std::stringstream ss;
    switch (type) {
      case ControlMsgType::NVL_EXPORT_MEM:
        ss << nvlDesc.toString();
        break;
      case ControlMsgType::NVL_RELEASE_MEM:
        ss << nvlRls.toString();
        break;
      case ControlMsgType::IB_EXPORT_MEM:
        ss << ibExp.toString();
        break;
      case ControlMsgType::SYNC:
        ss << "SYNC";
        break;
      default:
        ss << "UNSPECIFIED";
        break;
    }
    return ss.str();
  }
};

using ContrlMsgCbFn =
    std::function<commResult_t(int rank, void* msg, void* ctx)>;

struct ControlMsgCb {
  ContrlMsgCbFn fn;
  // contains comm specific pointer to find the corresponding instance
  void* ctx;
};

class CtranCtrlManager {
 public:
  commResult_t regCb(int type, ContrlMsgCbFn fn, void* ctx);
  commResult_t runCb(int rank, int type, void* msg) const;
  bool hasCb(int type) const;

 private:
  std::unordered_map<int, ControlMsgCb> ctrlMsgCbMap_;
};

#endif
