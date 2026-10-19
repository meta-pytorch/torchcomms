// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "comms/uniflow/Result.h"

namespace uniflow {

constexpr uint8_t kTcpWireVersion{1};

// Hard wire-frame cap from the controller's TcpConn framing (4-byte length
// prefix, 64 MiB max message). Any frame this transport emits -- including a
// ReadReply it builds on a peer's behalf -- must fit within it, because the
// controller refuses an oversized send and the sender thread treats that
// refusal as fatal.
constexpr size_t kMaxFrameSize = 64u << 20; // 64 MiB

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

/// Capability blob exchanged through TransportFactory::getTopology() /
/// canConnect(), before any connection is attempted. Deliberately separate from
/// TcpTransportInfo, which is *addressing* and is exchanged later via
/// bind()/connect(): reusing the addressing struct as the capability blob left
/// canConnect() with nothing meaningful to validate, so a wire-format mismatch
/// could only be discovered mid-transfer as a dropped frame.
///
/// Versioned so that mismatch is rejected at handshake time instead. Mirrors
/// RdmaTopologyInfo (transport/rdma/RdmaTransport.cpp). Shares kTcpWireVersion
/// with the frame header so a single bump covers both; the size check is exact,
/// so adding a capability field here requires bumping that version.
struct __attribute__((packed)) TcpTopologyInfo {
  uint8_t version{kTcpWireVersion};

  std::vector<uint8_t> serialize() const {
    std::vector<uint8_t> data(sizeof(TcpTopologyInfo));
    std::memcpy(data.data(), this, sizeof(TcpTopologyInfo));
    return data;
  }

  static Result<TcpTopologyInfo> deserialize(std::span<const uint8_t> data) {
    if (data.size() != sizeof(TcpTopologyInfo)) {
      return Err(
          ErrCode::InvalidArgument,
          "tcp topology payload size mismatch: expected " +
              std::to_string(sizeof(TcpTopologyInfo)) + " bytes, got " +
              std::to_string(data.size()));
    }
    TcpTopologyInfo info;
    std::memcpy(&info, data.data(), sizeof(TcpTopologyInfo));
    return info;
  }
};

constexpr uint32_t kTcpLaneMagic{0x554C414E}; // "ULAN"

/// Identifies one data socket within a multi-lane connection. The dialer sends
/// this on each socket immediately after it is established; the listener reads
/// it before using the socket, and places it at `laneIndex`. Accept order is
/// deliberately not used as identity: the sockets are dialed concurrently with
/// no ordering guarantee, so pairing lane N to lane N on both sides needs the
/// index carried explicitly.
///
/// Only exchanged when more than one lane is configured. A single-lane
/// transport therefore stays byte-identical on the wire to one built before
/// lanes existed, which is what lets kTcpWireVersion stay at its current value:
/// bumping it would make every new peer incompatible with every old one, and a
/// rollout of that kind should not be spent on a field only multi-lane peers
/// read.
///
/// sessionId is echoed by every lane of one connection, so a stray dialer
/// reaching this listener is rejected rather than silently occupying a lane.
struct __attribute__((packed)) TcpLaneHello {
  uint32_t magic{kTcpLaneMagic};
  uint8_t version{kTcpWireVersion};
  // 16-bit, so these bound the lane count rather than the reverse: a uint8_t
  // capped a connection at 255 lanes, and lanes are now counted per device, so
  // the ceiling is reached by device count as much as by configuration. The
  // spare byte this replaces was not enough on its own.
  uint16_t laneIndex{0};
  uint16_t laneCount{1};
  uint64_t sessionId{0};

  std::vector<uint8_t> serialize() const {
    std::vector<uint8_t> data(sizeof(TcpLaneHello));
    std::memcpy(data.data(), this, sizeof(TcpLaneHello));
    return data;
  }

  static Result<TcpLaneHello> deserialize(std::span<const uint8_t> data) {
    if (data.size() != sizeof(TcpLaneHello)) {
      return Err(
          ErrCode::InvalidArgument,
          "tcp lane hello size mismatch: expected " +
              std::to_string(sizeof(TcpLaneHello)) + " bytes, got " +
              std::to_string(data.size()));
    }
    TcpLaneHello hello;
    std::memcpy(&hello, data.data(), sizeof(TcpLaneHello));
    if (hello.magic != kTcpLaneMagic) {
      return Err(
          ErrCode::InvalidArgument,
          "tcp lane hello bad magic " + std::to_string(hello.magic));
    }
    if (hello.version != kTcpWireVersion) {
      return Err(
          ErrCode::InvalidArgument,
          "unsupported tcp wire version " + std::to_string(hello.version) +
              " in lane hello");
    }
    return hello;
  }
};

static_assert(sizeof(TcpLaneHello) == 17);

} // namespace uniflow
