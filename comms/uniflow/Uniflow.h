// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <string>

#include "comms/uniflow/Connection.h"
#include "comms/uniflow/MultiTransport.h"
#include "comms/uniflow/Result.h"
#include "comms/uniflow/Segment.h"
#include "comms/uniflow/controller/Controller.h"

namespace uniflow {

struct UniflowAgentConfig {
  int deviceId{-1};
  std::string name{};
  /// Listen address for incoming connections (e.g., "*:0" for auto-port).
  /// If non-empty, a TcpServer is created and initialized automatically.
  /// If empty, accept() is not available.
  std::string listenAddress{};
  int connectRetries{10};
  int connectTimeoutMs{1000};
};

class UniflowAgent {
 public:
  explicit UniflowAgent(
      const UniflowAgentConfig& config,
      std::unique_ptr<controller::Client> client = nullptr,
      std::unique_ptr<controller::Server> server = nullptr);
  ~UniflowAgent() = default;

  Result<std::string> getUniqueId() const;

  /// Register a local memory segment for data transfer.
  /// The returned RegisteredSegment can be used across multiple connections.
  Result<RegisteredSegment> registerSegment(Segment& segment);

  Result<RemoteRegisteredSegment> importSegment(
      std::span<const uint8_t> exportId);

  /// Server-side: accept an incoming connection from a peer
  Result<std::unique_ptr<Connection>> accept();

  /// Client-side: connect to a remote peer by its UniqueId
  Result<std::unique_ptr<Connection>> connect(std::string peerId);

 private:
  friend class UniflowAgentTest;

  /// Test-only constructor: inject a pre-built factory.
  UniflowAgent(
      const UniflowAgentConfig& config,
      std::shared_ptr<MultiTransportFactory> factory,
      std::unique_ptr<controller::Client> client = nullptr,
      std::unique_ptr<controller::Server> server = nullptr)
      : config_(config),
        factory_(std::move(factory)),
        client_(std::move(client)),
        server_(std::move(server)) {}

  /// Exchange topology and transport info, create and connect transport.
  /// If sendFirst is true, sends local data before receiving (used by accept).
  Result<std::unique_ptr<Connection>> establishConnection(
      std::unique_ptr<controller::Conn> ctrl,
      bool sendFirst);

  UniflowAgentConfig config_;
  std::shared_ptr<MultiTransportFactory> factory_;
  std::unique_ptr<controller::Client> client_;
  std::unique_ptr<controller::Server> server_;
};

} // namespace uniflow
