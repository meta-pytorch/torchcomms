// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/Rendezvous.h"

#include <algorithm>
#include <cerrno>
#include <cstring>

#include "comms/uniflow/controller/TcpController.h"
#include "comms/uniflow/logging/Logger.h"

namespace uniflow::benchmark {

namespace {

std::string serverAddr(const BootstrapConfig& config) {
  return config.masterAddr + ":" + std::to_string(config.masterPort);
}

std::vector<uint8_t> serializeInt(int32_t val) {
  std::vector<uint8_t> buf(sizeof(val));
  std::memcpy(buf.data(), &val, sizeof(val));
  return buf;
}

int32_t deserializeInt(const std::vector<uint8_t>& buf) {
  int32_t val = 0;
  if (buf.size() < sizeof(val)) {
    UNIFLOW_LOG_ERROR(
        "deserializeInt: buffer too small ({} < {})", buf.size(), sizeof(val));
    return -1;
  }
  std::memcpy(&val, buf.data(), sizeof(val));
  return val;
}

} // namespace

Result<std::vector<PeerConnection>> Rendezvous::establish(
    const BootstrapConfig& config) {
  std::vector<PeerConnection> peers;

  if (config.worldSize < 2) {
    UNIFLOW_LOG_WARN("Rendezvous: worldSize < 2, no peers to connect to");
    return peers;
  }

  if (config.isRank0()) {
    controller::TcpServer server(
        serverAddr(config), controller::TcpSocketConfig{});
    auto status = server.init();
    if (!status) {
      return Err(
          ErrCode::ConnectionFailed,
          "Rendezvous server init failed: " + status.error().toString());
    }

    UNIFLOW_LOG_INFO("Rendezvous: rank 0 listening on {}", serverAddr(config));

    for (int i = 1; i < config.worldSize; ++i) {
      auto conn = server.accept().get();
      if (!conn) {
        return Err(
            ErrCode::ConnectionFailed,
            "Rendezvous: failed to accept connection from peer");
      }

      std::vector<uint8_t> peerRankBuf;
      auto recvResult = conn->recv(peerRankBuf).get();
      if (!recvResult) {
        return Err(
            ErrCode::ConnectionFailed,
            "Rendezvous: failed to recv peer rank: " +
                recvResult.error().toString());
      }
      int peerRank = deserializeInt(peerRankBuf);
      if (peerRank < 0) {
        return Err(
            ErrCode::ConnectionFailed,
            "Rendezvous: received invalid peer rank");
      }

      UNIFLOW_LOG_INFO("Rendezvous: rank 0 accepted rank {}", peerRank);

      PeerConnection pc;
      pc.peerRank = peerRank;
      pc.ctrl = std::move(conn);
      peers.push_back(std::move(pc));
    }

    std::sort(peers.begin(), peers.end(), [](const auto& a, const auto& b) {
      return a.peerRank < b.peerRank;
    });
  } else {
    controller::TcpClient client(controller::TcpSocketConfig{});
    auto conn = client.connect(serverAddr(config)).get();
    if (!conn) {
      return Err(
          ErrCode::ConnectionFailed,
          "Rendezvous: rank " + std::to_string(config.rank) +
              " failed to connect to rank 0 at " + serverAddr(config));
    }

    auto sendResult = conn->send(serializeInt(config.rank)).get();
    if (!sendResult) {
      return Err(
          ErrCode::ConnectionFailed,
          "Rendezvous: failed to send rank: " + sendResult.error().toString());
    }

    UNIFLOW_LOG_INFO("Rendezvous: rank {} connected to rank 0", config.rank);

    PeerConnection pc;
    pc.peerRank = 0;
    pc.ctrl = std::move(conn);
    peers.push_back(std::move(pc));
  }

  return peers;
}

/// Retry recv on EAGAIN — the peer may be slow to reach the barrier during
/// long benchmark runs, causing SO_RCVTIMEO to fire.
/// With a 1-second SO_RCVTIMEO, 300 retries ≈ 5 minutes before giving up.
static Result<size_t> recvRetryEagain(
    controller::Conn& conn,
    std::vector<uint8_t>& buf) {
  constexpr int kMaxRetries = 300;
  int retries = 0;
  for (;;) {
    auto result = conn.recv(buf).get();
    if (result) {
      return result;
    }
    if ((errno == EAGAIN || errno == EWOULDBLOCK) && retries < kMaxRetries) {
      ++retries;
      UNIFLOW_LOG_INFO(
          "barrier: recv got EAGAIN, retrying ({}/{})...",
          retries,
          kMaxRetries);
      continue;
    }
    if (retries >= kMaxRetries) {
      return Err(
          ErrCode::ConnectionFailed,
          "barrier: recv timed out after " + std::to_string(kMaxRetries) +
              " EAGAIN retries (~5 minutes)");
    }
    return result;
  }
}

Status barrier(
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& config) {
  std::vector<uint8_t> token = {1};
  std::vector<uint8_t> buf;

  if (config.isRank0()) {
    for (auto& peer : peers) {
      auto result = recvRetryEagain(*peer.ctrl, buf);
      if (!result) {
        UNIFLOW_LOG_ERROR(
            "barrier: recv from rank {} failed: {}",
            peer.peerRank,
            result.error().toString());
        return Err(
            ErrCode::ConnectionFailed,
            "barrier: recv from rank " + std::to_string(peer.peerRank) +
                " failed: " + result.error().toString());
      }
    }
    for (auto& peer : peers) {
      auto result = peer.ctrl->send(token).get();
      if (!result) {
        UNIFLOW_LOG_ERROR(
            "barrier: send to rank {} failed: {}",
            peer.peerRank,
            result.error().toString());
        return Err(
            ErrCode::ConnectionFailed,
            "barrier: send to rank " + std::to_string(peer.peerRank) +
                " failed: " + result.error().toString());
      }
    }
  } else {
    if (peers.empty()) {
      UNIFLOW_LOG_ERROR("barrier: no peers for non-rank0");
      return Err(ErrCode::InvalidArgument, "barrier: no peers for non-rank0");
    }
    auto sendResult = peers[0].ctrl->send(token).get();
    if (!sendResult) {
      UNIFLOW_LOG_ERROR(
          "barrier: send to rank 0 failed: {}", sendResult.error().toString());
      return Err(
          ErrCode::ConnectionFailed,
          "barrier: send to rank 0 failed: " + sendResult.error().toString());
    }
    auto recvResult = recvRetryEagain(*peers[0].ctrl, buf);
    if (!recvResult) {
      UNIFLOW_LOG_ERROR(
          "barrier: recv from rank 0 failed: {}",
          recvResult.error().toString());
      return Err(
          ErrCode::ConnectionFailed,
          "barrier: recv from rank 0 failed: " + recvResult.error().toString());
    }
  }
  return Ok();
}

Result<std::vector<uint8_t>> exchangeMetadata(
    controller::Conn& ctrl,
    const std::vector<uint8_t>& localData,
    bool isRank0) {
  std::vector<uint8_t> remoteData;
  if (isRank0) {
    auto s = ctrl.send(localData).get();
    if (!s) {
      return Err(
          ErrCode::ConnectionFailed,
          "exchangeMetadata: send failed: " + s.error().toString());
    }
    auto r = ctrl.recv(remoteData).get();
    if (!r) {
      return Err(
          ErrCode::ConnectionFailed,
          "exchangeMetadata: recv failed: " + r.error().toString());
    }
  } else {
    auto r = ctrl.recv(remoteData).get();
    if (!r) {
      return Err(
          ErrCode::ConnectionFailed,
          "exchangeMetadata: recv failed: " + r.error().toString());
    }
    auto s = ctrl.send(localData).get();
    if (!s) {
      return Err(
          ErrCode::ConnectionFailed,
          "exchangeMetadata: send failed: " + s.error().toString());
    }
  }
  return remoteData;
}

} // namespace uniflow::benchmark
