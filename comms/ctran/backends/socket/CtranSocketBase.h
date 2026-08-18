// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include <fmt/format.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "comms/ctran/backends/CtranCtrl.h"

class CtranSocketRequest {
 public:
  CtranSocketRequest() {};
  ~CtranSocketRequest() {};
  inline commResult_t complete() {
    state_ = COMPLETE;
    return commSuccess;
  }

  inline bool isComplete() const {
    return this->state_ == CtranSocketRequest::COMPLETE;
  }

 private:
  enum {
    INCOMPLETE,
    COMPLETE,
  } state_{INCOMPLETE};
};

// Length-delimited Socket control frame. The Socket path used to carry only a
// fixed-size ControlMsg, but the mapper also exchanges raw registration
// descriptor batches. One framing format lets a control op be routed to Socket
// per peer regardless of which of the two forms it carries. Only
// header + payloadSize bytes go on the wire, not the full buffer.
struct SocketCtrlPacket {
  uint32_t payloadSize{0};
  std::array<char, CTRAN_CTRL_MAX_PAYLOAD_SIZE> payload{};

  bool copyFrom(const void* src, std::size_t size) {
    if ((size > 0 && src == nullptr) || size > payload.size()) {
      return false;
    }
    payloadSize = static_cast<uint32_t>(size);
    if (size > 0) {
      std::memcpy(payload.data(), src, size);
    }
    return true;
  }

  bool copyTo(void* dst, std::size_t size) const {
    if ((size > 0 && dst == nullptr) || size != payloadSize) {
      return false;
    }
    if (size > 0) {
      std::memcpy(dst, payload.data(), size);
    }
    return true;
  }

  std::size_t wireSize() const {
    return offsetof(SocketCtrlPacket, payload) + payloadSize;
  }
};

struct SockPendingOp {
  enum OpType {
    UNDEFINED,
    ISEND_CTRL,
    IRECV_CTRL,
  };

 public:
  SockPendingOp(
      SockPendingOp::OpType type,
      void* payload,
      std::size_t size,
      int peerRank,
      CtranSocketRequest& req)
      : type(type), payload(payload), size(size), peerRank(peerRank), req(req) {
    // Snapshot sends: a queued op is drained later, by which point the
    // caller's buffer may be gone. Receives write into payload on completion,
    // so the caller owns it until the request completes.
    if (type == ISEND_CTRL) {
      packet.copyFrom(payload, size);
    }
  }
  ~SockPendingOp() {}

  OpType type{UNDEFINED};

  void* payload{nullptr};
  std::size_t size{0};
  SocketCtrlPacket packet;
  int peerRank{-1};
  CtranSocketRequest& req;
};

template <>
struct fmt::formatter<SockPendingOp::OpType> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(SockPendingOp::OpType status, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};
