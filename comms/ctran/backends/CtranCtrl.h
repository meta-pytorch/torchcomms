// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_CTRL_H
#define CTRAN_CTRL_H

#include <comms/utils/cvars/nccl_cvars.h>
#include <fmt/format.h>
#include <cstddef>
#include <sstream>

#include "comms/ctran/backends/CtranAux.h"
#include "comms/ctran/regcache/IpcRegCacheBase.h"
#include "comms/ctran/utils/CtranIpc.h"

/**
 * Define all control message types and packet format used in CTran backends.
 *
 * Control messages are explicitly exchanged between two sides via a control
 * message channel's send/recv API, used for synchronization (e.g., handshake
 * in zero-copy algorithms).
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

/**
 * Packet structure of control message transferred by underlying backend.
 */
struct ControlMsg {
  int type{ControlMsgType::UNSPECIFIED};
  union {
    struct ctran::regcache::IpcDesc ipcDesc;
    struct ctran::regcache::IpcRelease ipcRls;
    struct ctran::regcache::IBDesc ibDesc;
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
        ipcDesc = ctran::regcache::IpcDesc{};
        break;
      case ControlMsgType::NVL_RELEASE_MEM:
        ipcRls = ctran::regcache::IpcRelease{};
        break;
      case ControlMsgType::IB_EXPORT_MEM:
        ibDesc = ctran::regcache::IBDesc{};
        break;
      default:
        break;
    }
  }

  std::string toString() const {
    std::stringstream ss;
    switch (type) {
      case ControlMsgType::NVL_EXPORT_MEM:
        ss << ipcDesc.toString();
        break;
      case ControlMsgType::NVL_RELEASE_MEM:
        ss << ipcRls.toString();
        break;
      case ControlMsgType::IB_EXPORT_MEM:
        ss << ibDesc.toString();
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

#endif
