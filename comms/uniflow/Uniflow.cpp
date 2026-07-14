// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/Uniflow.h"

#include "comms/uniflow/controller/TcpController.h"
#include "comms/uniflow/logging/Logger.h"

namespace uniflow {

UniflowAgent::UniflowAgent(
    const UniflowAgentConfig& config,
    std::unique_ptr<controller::Client> client,
    std::unique_ptr<controller::Server> server)
    : config_(config),
      factory_(std::make_shared<MultiTransportFactory>(config.deviceId)),
      client_(std::move(client)),
      server_(std::move(server)) {
  if (!client_) {
    controller::TcpSocketConfig tcpCfg;
    tcpCfg.connectRetries = config.connectRetries;
    tcpCfg.retryTimeout = std::chrono::milliseconds(config.connectTimeoutMs);
    client_ = std::make_unique<controller::TcpClient>(std::move(tcpCfg));
  }

  if (!server_ && !config.listenAddress.empty()) {
    auto tcpServer =
        std::make_unique<controller::TcpServer>(config.listenAddress);
    auto status = tcpServer->init();
    if (!status) {
      throw std::runtime_error(
          "TcpServer init failed: " + status.error().toString());
    }
    server_ = std::move(tcpServer);
  }
}

Result<std::string> UniflowAgent::getUniqueId() const {
  if (!server_) {
    return Err(ErrCode::InvalidArgument, "no server configured");
  }
  return server_->getId();
}

Result<RegisteredSegment> UniflowAgent::registerSegment(Segment& segment) {
  return factory_->registerSegment(segment);
}

Result<RemoteRegisteredSegment> UniflowAgent::importSegment(
    std::span<const uint8_t> exportId) {
  return factory_->importSegment(exportId);
}

Result<std::unique_ptr<Connection>> UniflowAgent::accept() {
  if (!server_) {
    return Err(ErrCode::InvalidArgument, "no server configured");
  }

  auto ctrl = server_->accept().get();
  if (!ctrl) {
    return Err(ErrCode::ConnectionFailed, "accept failed");
  }

  // accept sends first, connect receives first (mirrored to avoid deadlock)
  return establishConnection(std::move(ctrl), true);
}

Result<std::unique_ptr<Connection>> UniflowAgent::connect(std::string peerId) {
  if (!client_) {
    return Err(ErrCode::InvalidArgument, "no client configured");
  }

  auto ctrl = client_->connect(std::move(peerId)).get();
  if (!ctrl) {
    return Err(ErrCode::ConnectionFailed, "connect failed");
  }

  // connect receives first, accept sends first
  return establishConnection(std::move(ctrl), false);
}

Result<std::unique_ptr<Connection>> UniflowAgent::establishConnection(
    std::unique_ptr<controller::Conn> ctrl,
    bool sendFirst) {
  auto myTopo = factory_->getTopology();

  /*
   * The recv() calls below are bounded by the connected-socket timeout: if the
   * peer's transport setup silently failed, a blocked recv now fails with
   * ConnectionFailed instead of hanging forever. These DEBUG phase logs bracket
   * each blocking exchange so, when enabled, a stall/timeout is attributable to
   * a specific handshake step. Kept at DEBUG to avoid per-connection noise.
   */
  UNIFLOW_LOG_DEBUG("establishConnection: start (sendFirst={})", sendFirst);

  // Exchange topology
  std::vector<uint8_t> peerTopo;
  if (sendFirst) {
    CHECK_EXPR(ctrl->send({myTopo.begin(), myTopo.end()}).get());
    UNIFLOW_LOG_DEBUG("establishConnection: topology sent, awaiting peer");
    CHECK_EXPR(ctrl->recv(peerTopo).get());
  } else {
    UNIFLOW_LOG_DEBUG("establishConnection: awaiting peer topology");
    CHECK_EXPR(ctrl->recv(peerTopo).get());
    CHECK_EXPR(ctrl->send({myTopo.begin(), myTopo.end()}).get());
  }

  // Create transport from peer topology
  auto transportResult = factory_->createTransport(peerTopo);
  CHECK_RETURN(transportResult);
  auto& transport = transportResult.value();

  // Bind and exchange transport info
  UNIFLOW_LOG_DEBUG(
      "establishConnection: topology exchanged, binding transport");
  auto bindResult = transport->bind();
  CHECK_RETURN(bindResult);
  auto& localInfo = bindResult.value();

  std::vector<uint8_t> remoteInfo;
  if (sendFirst) {
    CHECK_EXPR(ctrl->send(localInfo).get());
    UNIFLOW_LOG_DEBUG(
        "establishConnection: transport info sent, awaiting peer");
    CHECK_EXPR(ctrl->recv(remoteInfo).get());
  } else {
    UNIFLOW_LOG_DEBUG("establishConnection: awaiting peer transport info");
    CHECK_EXPR(ctrl->recv(remoteInfo).get());
    CHECK_EXPR(ctrl->send(localInfo).get());
  }

  // Connect transport with remote endpoint info
  UNIFLOW_LOG_DEBUG(
      "establishConnection: transport info exchanged, connecting");
  CHECK_EXPR(transport->connect(remoteInfo));
  UNIFLOW_LOG_DEBUG("establishConnection: connection established");
  return std::make_unique<Connection>(std::move(ctrl), std::move(transport));
}

} // namespace uniflow
