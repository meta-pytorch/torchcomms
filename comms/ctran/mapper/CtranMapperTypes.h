// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <fmt/format.h>
#include <chrono>
#include <cstddef>
#include <vector>

#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/backends/nvl/CtranNvlBase.h"
#include "comms/ctran/backends/socket/CtranSocketBase.h"

#ifdef CTRAN_DISABLE_TCPDM
#include "comms/ctran/backends/mock/CtranTcpDmBaseMock.h"
#else
#include "comms/ctran/backends/tcpdevmem/CtranTcpDmBase.h"
#endif

#ifdef ENABLE_META_COMPRESSION
#include "comms/ctran/backends/CtranAux.h"
#endif

enum CtranMapperBackend {
  UNSET, // a rank not supported by any backend
  IB,
  NVL,
  SOCKET,
  TCPDM,
  NUM_BACKENDS
};

constexpr const char* CtranMapperBackendToString(CtranMapperBackend backend) {
  switch (backend) {
    case UNSET:
      return "UNSET";
    case IB:
      return "IB";
    case NVL:
      return "NVL";
    case SOCKET:
      return "SOCKET";
    case TCPDM:
      return "TCPDM";
    case NUM_BACKENDS:
      return "NUM_BACKENDS";
    default:
      return "Unknown";
  }
}
template <>
struct fmt::formatter<CtranMapperBackend> : fmt::formatter<const char*> {
  template <typename FormatContext>
  auto format(CtranMapperBackend backend, FormatContext& ctx) const {
    return fmt::formatter<const char*>::format(
        CtranMapperBackendToString(backend), ctx);
  }
};

struct CtranMapperRemoteAccessKey {
  CtranMapperBackend backend{CtranMapperBackend::UNSET};
  struct CtranIbRemoteAccessKey ibKey;
  struct CtranNvlRemoteAccessKey nvlKey;

  std::string toString() const;
};

struct KernelElem;
struct CtranIbConfig;

struct CtranMapperConfig {
  void* memHdl_{nullptr};
  CtranMapperRemoteAccessKey remoteAccessKey_{CtranMapperBackend::UNSET};
  bool notify_{false};
  KernelElem* kernElem_{nullptr};
  CtranIbConfig* ibConfig_{nullptr};
  bool ibFastPath_{false};
};

class CtranMapperNotify {
 public:
  int peer{-1};
  // Number of the notifies to receive.
  int notifyCnt{1};
  KernelElem* kernElem{nullptr};
  CtranMapperBackend backend{CtranMapperBackend::UNSET};
  // TCPDM abuses notifiers to store the receiver-side request. See
  // waitNotifyImpl comment for more details.
  ctran::CtranTcpDmRequest tcpDmReq;

  CtranMapperNotify() = default;

  inline void update(
      int peerRank,
      KernelElem* kernelElem,
      CtranMapperBackend mapperBackend,
      int notifyCnt = 1) {
    this->peer = peerRank;
    this->notifyCnt = notifyCnt;
    this->kernElem = kernelElem;
    this->backend = mapperBackend;
  }

  std::string toString() const;
};

class CtranMapperRequest {
 public:
  enum ReqType {
    SEND_CTRL,
    RECV_CTRL,
    SEND_SYNC_CTRL,
    RECV_SYNC_CTRL,
    SEND_CTRL_MSG,
    RECV_CTRL_MSG,
    IB_PUT,
    IB_GET,
    NVL_PUT,
    TCPDM_PUT,
    COPY,
    ATOMIC_SET
  };
  CtranMapperRequest() {
    CtranMapperRequest(SEND_CTRL, -1, CtranMapperBackend::IB);
  }

  CtranMapperRequest(
      CtranMapperRequest::ReqType type,
      int peer,
      CtranMapperBackend backend = CtranMapperBackend::IB);
  ~CtranMapperRequest();

  // copy constructor
  CtranMapperRequest(const CtranMapperRequest& other) {
    type = other.type;
    peer = other.peer;
    backend = other.backend;
    ibReq = other.ibReq;
    sockReq = other.sockReq;
    tcpDmReq = other.tcpDmReq;
    state_ = other.state_;
    setCtrlMsg(other);
    setConfig(other.getConfig());
  }

  // copy operator
  CtranMapperRequest& operator=(const CtranMapperRequest& other) {
    if (this != &other) {
      type = other.type;
      peer = other.peer;
      backend = other.backend;
      ibReq = other.ibReq;
      sockReq = other.sockReq;
      tcpDmReq = other.tcpDmReq;
      state_ = other.state_;
      setCtrlMsg(other);
      setConfig(other.getConfig());
    }
    return *this;
  }

  inline void setCtrlMsg(const CtranMapperRequest& other) {
    switch (other.type) {
      case RECV_CTRL:
        recvCtrl = other.recvCtrl;
        break;
      case SEND_SYNC_CTRL:
        sendSyncCtrl = other.sendSyncCtrl;
        break;
      case RECV_SYNC_CTRL:
        recvSyncCtrl = other.recvSyncCtrl;
        break;
      case SEND_CTRL:
      default:
        sendCtrl = other.sendCtrl;
        break;
    }
  }

  // Keep the copy the Mapper configuration
  inline void setConfig(const CtranMapperConfig& config) {
    this->config_ = config;
  }

  inline const CtranMapperConfig& getConfig() const {
    return this->config_;
  }

  inline bool isComplete() const {
    return this->state_ == COMPLETE;
  }

  ReqType type{SEND_CTRL};
  // FIXME: use union to save space
  CtranIbRequest ibReq;
  CtranSocketRequest sockReq;
  ctran::CtranTcpDmRequest tcpDmReq;
  int peer{-1};
  CtranMapperBackend backend{CtranMapperBackend::IB};
#ifdef ENABLE_META_COMPRESSION
  AuxData_t<DefaultAuxType> aux{0}; // Used to store a copy of AuxData
#endif

  union {
    struct {
      ControlMsg msg;
    } sendCtrl;
    struct {
      ControlMsg msg;
      void** buf;
      struct CtranMapperRemoteAccessKey* key;
    } recvCtrl;
    struct {
      ControlMsg msg;
    } sendSyncCtrl;
    struct {
      ControlMsg msg;
    } recvSyncCtrl;
  };

 protected:
  // Set state to COMPLETE
  inline void setComplete() {
    this->state_ = COMPLETE;
  }

 private:
  friend class CtranMapper;
  CtranMapperConfig config_;
  enum {
    INCOMPLETE,
    COMPLETE,
  } state_{INCOMPLETE};
  // stream this request is associated with
  std::optional<cudaStream_t> workStream;
};

struct CtranMapperPutMsg {
  const void* sbuf;
  void* dbuf;
  std::size_t len;
  CtranMapperConfig config;
  CtranMapperRequest* req;
};

/*
 * Holds the context of the request, e.g., the algorithm and send size.
 * It can be used to share information from the algo layer to the ctran mapper
 * layer.
 */
class CtranMapperContext {
 public:
  CtranMapperContext() = default;
  CtranMapperContext(
      const std::string& algorithmName,
      size_t sendMessageSize,
      size_t recvMessageSize)
      : algorithmName(algorithmName),
        sendMessageSize(sendMessageSize),
        recvMessageSize(recvMessageSize) {}
  CtranMapperContext(
      const std::string& algorithmName,
      const std::vector<size_t>& sendMessageSizes,
      const std::vector<size_t>& recvMessageSizes)
      : algorithmName(algorithmName),
        sendMessageSizes(sendMessageSizes),
        recvMessageSizes(recvMessageSizes) {}
  ~CtranMapperContext() = default;

  // Explicitly default the move constructor and move assignment operator:
  CtranMapperContext(CtranMapperContext&&) = default;
  CtranMapperContext& operator=(CtranMapperContext&&) = default;

  size_t getSendMsgSize(int peerRank) const {
    if (sendMessageSizes.size() == 0) {
      return sendMessageSize;
    }
    if (sendMessageSizes.size() > peerRank) {
      return sendMessageSizes[peerRank];
    }
    return 0;
  }

  size_t getRecvMsgSize(int peerRank) const {
    if (recvMessageSizes.size() == 0) {
      return recvMessageSize;
    }
    if (recvMessageSizes.size() > peerRank) {
      return recvMessageSizes[peerRank];
    }
    return 0;
  }

  std::string algorithmName{"unknown"};
  std::vector<size_t> sendMessageSizes;
  std::vector<size_t> recvMessageSizes;
  size_t sendMessageSize{0};
  size_t recvMessageSize{0};
};

class CtranMapperTimestampPoint {
 public:
  CtranMapperTimestampPoint(int peer) {
    this->now = std::chrono::high_resolution_clock::now();
    this->peer = peer;
  }
  ~CtranMapperTimestampPoint() = default;

  std::chrono::time_point<std::chrono::high_resolution_clock> now;
  int peer;
};

class CtranMapperTimestamp {
 public:
  CtranMapperTimestamp(const std::string algo) {
    this->algo = algo;
    this->start = std::chrono::high_resolution_clock::now();
  }
  ~CtranMapperTimestamp() = default;

  std::vector<CtranMapperTimestampPoint> recvCtrl;
  std::vector<CtranMapperTimestampPoint> putIssued;
  std::vector<CtranMapperTimestampPoint> putComplete;
  std::vector<CtranMapperTimestampPoint> kernelPost;
  std::vector<CtranMapperTimestampPoint> kernelWait;
  std::vector<CtranMapperTimestampPoint> kernelWaitComplete;
  std::string algo;
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

class CtranMapperTimer {
 public:
  CtranMapperTimer() {
    this->start_ = std::chrono::steady_clock::now();
  }
  ~CtranMapperTimer() = default;
  double durationMs() {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               end - this->start_)
        .count();
  }

  double durationUs() {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
               end - this->start_)
        .count();
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> start_;
};
