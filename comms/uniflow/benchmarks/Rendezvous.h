// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <vector>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/benchmarks/Bootstrap.h"
#include "comms/uniflow/controller/Controller.h"

namespace uniflow::benchmark {

struct PeerConnection {
  int peerRank{-1};
  std::unique_ptr<controller::Conn> ctrl;
};

/// Star-topology rendezvous via TcpController.
/// Rank 0 acts as server; all other ranks connect as clients.
/// Returns control-plane connections only — transport connections are
/// established separately by individual benchmarks.
class Rendezvous {
 public:
  /// Returns one PeerConnection per peer (N-1 total).
  static Result<std::vector<PeerConnection>> establish(
      const BootstrapConfig& config);
};

/// Rank 0 collects a token from each peer, then broadcasts back.
Status barrier(
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& config);

/// Exchange opaque byte payloads between two peers over the control channel.
/// Rank 0 sends first then receives; other ranks receive first then send.
Result<std::vector<uint8_t>> exchangeMetadata(
    controller::Conn& ctrl,
    const std::vector<uint8_t>& localData,
    bool isRank0);

} // namespace uniflow::benchmark
