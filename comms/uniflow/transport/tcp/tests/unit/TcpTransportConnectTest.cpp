// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/tcp/TcpTransport.h"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/tcp/TcpWireProtocol.h"

namespace uniflow {

// connect() decides who listens and who dials by comparing the two endpoints'
// order keys. Identical endpoints have no valid assignment: both peers would
// dial and nobody would accept, so connect() must reject rather than hang.
class TcpTransportConnectTest : public ::testing::Test {
 protected:
  void SetUp() override {
    evbThread_ = std::make_unique<ScopedEventBaseThread>("tcp-connect-test");
    registry_ = std::make_shared<TcpSegmentRegistry>();
    transport_ = std::make_unique<TcpTransport>(
        /*deviceId=*/-1,
        evbThread_->getEventBase(),
        registry_,
        controller::TcpSocketConfig{},
        /*host=*/"127.0.0.1");
  }

  /// A transport whose handshake wait is short, so timeout paths finish fast.
  std::unique_ptr<TcpTransport> makeShortTimeoutTransport() {
    controller::TcpSocketConfig config;
    config.connTimeout = std::chrono::seconds{2};
    return std::make_unique<TcpTransport>(
        /*deviceId=*/-1,
        evbThread_->getEventBase(),
        registry_,
        config,
        /*host=*/"127.0.0.1");
  }

  void TearDown() override {
    if (transport_) {
      transport_->shutdown();
      transport_.reset();
    }
    evbThread_.reset();
  }

  std::unique_ptr<ScopedEventBaseThread> evbThread_;
  std::shared_ptr<TcpSegmentRegistry> registry_;
  std::unique_ptr<TcpTransport> transport_;
};

TEST_F(TcpTransportConnectTest, RejectsIdenticalEndpoint) {
  // bind() publishes this transport's own host:port. Feeding that straight back
  // into connect() is exactly the degenerate case.
  const TransportInfo self = transport_->bind();
  ASSERT_FALSE(self.empty()) << "bind() must publish a routable endpoint";
  ASSERT_EQ(transport_->state(), TransportState::Initialized);

  const Status status = transport_->connect(self);

  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::ConnectionFailed);
  // The code alone does not pin this guard: with the guard removed, dialing our
  // own listener also fails with ConnectionFailed ("tcp connect: data
  // connection failed"). Assert on the message so this test goes red if the
  // identical-endpoint check is ever dropped.
  EXPECT_NE(status.error().message().find("identical"), std::string::npos)
      << "expected the identical-endpoint guard to fire, got: "
      << status.error().message();
  // Fenced, so a caller cannot then drive transfers over a transport that never
  // established a connection.
  EXPECT_EQ(transport_->state(), TransportState::Error);
}

// The listener branch of connect() waits on AsyncAccept, whose promise is
// resolved only by an inbound connection or by teardown. If the dialing peer
// never arrives, an unbounded wait would wedge this thread forever -- and
// nothing could rescue it, because server_ belongs to the transport whose
// connect() is parked. Assert the wait is bounded and reports failure.
TEST_F(TcpTransportConnectTest, ListenerTimesOutWhenNoPeerDials) {
  auto listener = makeShortTimeoutTransport();
  const TransportInfo self = listener->bind();
  ASSERT_FALSE(self.empty());

  // Choose a peer endpoint that orders ABOVE ours so connect() takes the
  // listener branch, then never dial it.
  auto peer = TcpTransportInfo::deserialize(self);
  ASSERT_TRUE(peer.hasValue()) << peer.error().message();
  TcpTransportInfo absent = peer.value();
  absent.port = static_cast<uint16_t>(absent.port + 1);

  const auto start = std::chrono::steady_clock::now();
  const Status status = listener->connect(absent.serialize());
  const auto elapsed = std::chrono::steady_clock::now() - start;

  ASSERT_TRUE(status.hasError())
      << "connect() must fail rather than block when no peer dials in";
  EXPECT_EQ(status.error().code(), ErrCode::ConnectionFailed);
  // Bounded: comfortably under the 30s default, proving the configured
  // timeout is what released it rather than some unrelated failure.
  EXPECT_LT(elapsed, std::chrono::seconds{20})
      << "connect() took "
      << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
      << "s; the handshake wait is not bounded by connTimeout";

  listener->shutdown();
}

TEST_F(TcpTransportConnectTest, RejectsMalformedPeerInfo) {
  ASSERT_FALSE(transport_->bind().empty());

  // A truncated payload cannot name an endpoint; connect() must reject it
  // rather than proceed with a partially parsed peer.
  const std::vector<uint8_t> truncated(2, 0);
  const Status status = transport_->connect(truncated);

  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

// shutdown() used to load state_ and then store Disconnected as two separate
// steps, so a bind() failure landing in the gap was clobbered and a transport
// that never came up reported as cleanly closed. Error must survive teardown.
TEST_F(TcpTransportConnectTest, ShutdownPreservesTheErrorState) {
  // TEST-NET-1 (RFC 5737): never assigned to a local interface, so the listen
  // socket genuinely fails to bind rather than the failure being simulated.
  auto unbindable = std::make_unique<TcpTransport>(
      /*deviceId=*/-1,
      evbThread_->getEventBase(),
      registry_,
      controller::TcpSocketConfig{},
      /*host=*/"192.0.2.1");

  EXPECT_TRUE(unbindable->bind().empty()) << "bind() was expected to fail";
  ASSERT_EQ(unbindable->state(), TransportState::Error);

  unbindable->shutdown();

  EXPECT_EQ(unbindable->state(), TransportState::Error)
      << "shutdown() must not report a failed transport as cleanly closed";
}

// shutdown() is called twice in the normal flow (MultiTransport::shutdown()
// then ~TcpTransport), and may be called on a transport that never connected.
TEST_F(TcpTransportConnectTest, ShutdownIsIdempotent) {
  ASSERT_FALSE(transport_->bind().empty());

  transport_->shutdown();
  transport_->shutdown();

  EXPECT_EQ(transport_->state(), TransportState::Disconnected);
}

TEST_F(TcpTransportConnectTest, ShutdownWithoutBindOrConnectIsSafe) {
  transport_->shutdown();

  EXPECT_EQ(transport_->state(), TransportState::Disconnected);
}

// Teardown is one-shot. connect() parks for the whole handshake, so without a
// terminal check it could install a data connection and both worker threads
// after shutdown() had already returned, leaving state() reporting Connected on
// a torn-down transport.
TEST_F(TcpTransportConnectTest, ConnectAfterShutdownIsRefused) {
  const TransportInfo self = transport_->bind();
  ASSERT_FALSE(self.empty());
  auto peer = TcpTransportInfo::deserialize(self);
  ASSERT_TRUE(peer.hasValue()) << peer.error().message();
  TcpTransportInfo other = peer.value();
  other.port = static_cast<uint16_t>(other.port + 1);

  transport_->shutdown();
  const Status status = transport_->connect(other.serialize());

  ASSERT_TRUE(status.hasError())
      << "connect() must not revive a shut-down transport";
  EXPECT_EQ(status.error().code(), ErrCode::NotConnected);
  EXPECT_NE(transport_->state(), TransportState::Connected);
}

TEST_F(TcpTransportConnectTest, BindAfterShutdownIsRefused) {
  transport_->shutdown();

  EXPECT_TRUE(transport_->bind().empty())
      << "bind() must not re-open a shut-down transport";
}

// The factory's topology blob used to be a default-constructed *addressing*
// struct, so it advertised 127.0.0.1:0 rather than a real endpoint and carried
// no version. canConnect() therefore validated nothing, and a wire-format
// mismatch could only surface later as a dropped frame mid-transfer.
class TcpTransportTopologyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    evbThread_ = std::make_unique<ScopedEventBaseThread>("tcp-topology-test");
    factory_ = std::make_unique<TcpTransportFactory>(
        /*deviceId=*/-1,
        evbThread_->getEventBase(),
        controller::TcpSocketConfig{},
        /*host=*/"127.0.0.1");
  }

  void TearDown() override {
    factory_.reset();
    evbThread_.reset();
  }

  std::unique_ptr<ScopedEventBaseThread> evbThread_;
  std::unique_ptr<TcpTransportFactory> factory_;
};

TEST_F(TcpTransportTopologyTest, TopologyIsAVersionedCapabilityBlob) {
  const std::vector<uint8_t> topology = factory_->getTopology();

  // Capability, not addressing: it is exactly the topology struct, so there is
  // no host or port in it to be mistaken for a real endpoint.
  ASSERT_EQ(topology.size(), sizeof(TcpTopologyInfo));
  auto info = TcpTopologyInfo::deserialize(topology);
  ASSERT_TRUE(info.hasValue()) << info.error().message();
  EXPECT_EQ(info->version, kTcpWireVersion);
  EXPECT_FALSE(factory_->canConnect(topology).hasError())
      << "a peer running this build must be accepted";
}

TEST_F(TcpTransportTopologyTest, RejectsMismatchedTopologyVersion) {
  TcpTopologyInfo future;
  future.version = static_cast<uint8_t>(kTcpWireVersion + 1);

  const Status status = factory_->canConnect(future.serialize());

  ASSERT_TRUE(status.hasError())
      << "a peer on a different wire version must be rejected at handshake";
  EXPECT_EQ(status.error().code(), ErrCode::TopologyDisconnect);
}

TEST_F(TcpTransportTopologyTest, RejectsWrongSizedTopology) {
  const std::vector<uint8_t> oversized(sizeof(TcpTopologyInfo) + 4, 0);

  const Status status = factory_->canConnect(oversized);

  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_F(TcpTransportTopologyTest, CreateTransportValidatesPeerTopology) {
  TcpTopologyInfo future;
  future.version = static_cast<uint8_t>(kTcpWireVersion + 1);

  auto result = factory_->createTransport(future.serialize());

  ASSERT_TRUE(result.hasError())
      << "createTransport must not build against an incompatible peer";
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);
}

} // namespace uniflow
