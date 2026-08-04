// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "comms/uniflow/Result.h"

namespace uniflow {

constexpr uint8_t kTcpWireVersion{1};

enum class TcpOp : uint8_t {
  Write = 1,
  ReadRequest = 2,
  ReadReply = 3,
  Notification = 4,
  Ack = 5,
  Error = 6,
  Send = 7,
};

struct __attribute__((packed)) TcpMsgHeader {
  uint8_t version{kTcpWireVersion};
  uint8_t op{0};
  uint8_t flags{0};
  uint8_t rsvd{0};
  uint64_t reqId{0};
  uint64_t segId{0};
  uint64_t offset{0};
  uint64_t len{0};
};

static_assert(sizeof(TcpMsgHeader) == 36);

inline std::vector<uint8_t> serializeTcpHeader(const TcpMsgHeader& header) {
  std::vector<uint8_t> data(sizeof(TcpMsgHeader));
  std::memcpy(data.data(), &header, sizeof(header));
  return data;
}

inline Result<TcpMsgHeader> deserializeTcpHeader(
    std::span<const uint8_t> data) {
  if (data.size() < sizeof(TcpMsgHeader)) {
    return Err(
        ErrCode::InvalidArgument,
        "tcp header truncated: need " + std::to_string(sizeof(TcpMsgHeader)) +
            " bytes, got " + std::to_string(data.size()));
  }

  TcpMsgHeader header;
  std::memcpy(&header, data.data(), sizeof(header));
  if (header.version != kTcpWireVersion) {
    return Err(
        ErrCode::InvalidArgument,
        "unsupported tcp wire version " + std::to_string(header.version));
  }
  return header;
}

inline bool tcpOpHasPayload(TcpOp op) {
  return op == TcpOp::Write || op == TcpOp::ReadReply || op == TcpOp::Send;
}

} // namespace uniflow
