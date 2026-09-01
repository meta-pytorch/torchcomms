// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/tcp/TcpTransport.h"

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
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

  /// True if something accepts a TCP connection on 127.0.0.1:port right now.
  /// The kernel completes the handshake into the accept queue whether or not
  /// userspace has called accept(), so this reports whether the listening
  /// socket is still open -- which is exactly what shutdown() is meant to
  /// change.
  static bool canConnectTo(uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      return false;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const bool connected =
        ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    ::close(fd);
    return connected;
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

// shutdown() must close the listeners bind() opened, rather than leaving them
// to the destructor.
//
// servers_ is a member, so leaving it untouched means the listener fds stay
// open until the transport object is destroyed -- which in a real process is
// during teardown, concurrent with the EventBase thread stopping.
// AsyncAccept::shutdown() reaches the loop through an unbounded
// dispatchAndWait(), so that overlap is precisely the window where teardown
// wedges (see TcpAsyncAcceptMiscTest.ShutdownDoesNotBlockOnABusyEventBase).
// Closing the listeners while shutdown() still owns a loop it knows is running
// removes the window instead of narrowing it.
//
// Observed from outside the transport: once shutdown() returns, nothing is
// listening on the port bind() published.
TEST_F(TcpTransportConnectTest, ShutdownClosesTheListener) {
  const TransportInfo self = transport_->bind();
  ASSERT_FALSE(self.empty()) << "bind() must publish a routable endpoint";
  auto info = TcpTransportInfo::deserialize(self);
  ASSERT_TRUE(info.hasValue()) << info.error().message();
  const uint16_t port = info.value().port;

  // Sanity first, so a refusal below is shutdown()'s doing rather than a port
  // that was never listening.
  ASSERT_TRUE(canConnectTo(port)) << "bind() left no listening socket";

  transport_->shutdown();

  EXPECT_FALSE(canConnectTo(port))
      << "the listener outlived shutdown(): it closes only when the transport is "
         "destroyed, which in a real process races the EventBase thread stopping "
         "and is where teardown wedges";
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

// bind() re-arms the Initialized state that connect() gates on, so without an
// already-bound guard a second bind() lets a second connect() through on a live
// transport. establishLanes() clears lanes_ as its first act, which destroys
// the current lanes' joinable reader/sender threads -- and ~std::thread on a
// joinable thread calls std::terminate, taking down every other transport in
// the process with it.
TEST_F(TcpTransportConnectTest, BindAfterConnectIsRefused) {
  auto peerEvb =
      std::make_unique<ScopedEventBaseThread>("tcp-connect-test-peer");
  auto peer = std::make_unique<TcpTransport>(
      /*deviceId=*/-1,
      peerEvb->getEventBase(),
      registry_,
      controller::TcpSocketConfig{},
      /*host=*/"127.0.0.1");

  const TransportInfo selfInfo = transport_->bind();
  const TransportInfo peerInfo = peer->bind();
  ASSERT_FALSE(selfInfo.empty());
  ASSERT_FALSE(peerInfo.empty());
  // Each side dials or accepts by endpoint order, so both connects have to be
  // in flight at once for the handshake to complete.
  std::thread dialer([&]() { (void)peer->connect(selfInfo); });
  const Status status = transport_->connect(peerInfo);
  dialer.join();
  ASSERT_FALSE(status.hasError()) << status.error().message();
  ASSERT_EQ(transport_->state(), TransportState::Connected);

  EXPECT_TRUE(transport_->bind().empty())
      << "bind() must not re-arm a transport that is already connected";
  EXPECT_EQ(transport_->state(), TransportState::Connected)
      << "a refused bind() must leave the live connection intact";

  peer->shutdown();
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

TEST(TcpTransportConfigTest, AsyncGetH2dDefaultsOn) {
  EXPECT_TRUE(TcpTransportConfig{}.asyncGetH2d);
}

// Data sockets must leave SO_SNDBUF/SO_RCVBUF to the kernel no matter how the
// config was built. Asserted through the converting constructor specifically,
// because that is the path that defeated the previous attempt: the default on
// socketConfig was bulkDataDefaults(), but any caller handing over a bare
// TcpSocketConfig -- the benchmark, the cross-host tests, anything arriving via
// MultiTransport -- replaced it wholesale and silently restored that type's own
// 1 MiB. A test on the default alone passed throughout, which is why this one
// sets a value and demands it be dropped.
TEST(TcpTransportConfigTest, DataSocketsAlwaysLeaveSocketBuffersToTheKernel) {
  controller::TcpSocketConfig explicitCfg;
  explicitCfg.socketBufSize = 4 << 20;
  const TcpTransportConfig config{explicitCfg};

  ASSERT_EQ(config.socketConfig.socketBufSize, 4 << 20)
      << "precondition: the caller's value reaches the config";
  EXPECT_FALSE(config.dataSocketConfig(std::nullopt).socketBufSize.has_value())
      << "an explicit SO_RCVBUF pins the window below a 200G link's BDP";
}

// Only the buffer size is overridden. This is not osDefaults(): a data socket
// still wants the keepalive and timeout settings the caller asked for, so a
// future field cleared here by accident should show up as a failure.
TEST(TcpTransportConfigTest, DataSocketConfigKeepsEveryOtherField) {
  const TcpTransportConfig config{};
  const auto data = config.dataSocketConfig(std::nullopt);
  const auto& given = config.socketConfig;

  EXPECT_EQ(data.connTimeout, given.connTimeout);
  EXPECT_EQ(data.tcpNoDelay, given.tcpNoDelay);
  EXPECT_EQ(data.enableKeepalive, given.enableKeepalive);
  EXPECT_EQ(data.keepaliveIdle, given.keepaliveIdle);
  EXPECT_EQ(data.keepaliveInterval, given.keepaliveInterval);
  EXPECT_EQ(data.keepaliveCount, given.keepaliveCount);
  EXPECT_EQ(data.userTimeout, given.userTimeout);
}

// The egress device is per-connection rather than per-config, so a nullopt
// device has to clear an inherited binding rather than leave it. Asserted with
// a binding already on the config, because asserting it on the default proves
// nothing -- the default has none, so such a test passes whether the clearing
// happens or not. A leaked binding would pin every lane to one device while the
// transport believed it was striping across several.
TEST(TcpTransportConfigTest, DataSocketConfigOverridesTheInheritedDevice) {
  controller::TcpSocketConfig staleCfg;
  staleCfg.bindToDevice = "eth9";
  const TcpTransportConfig config{staleCfg};

  ASSERT_EQ(config.socketConfig.bindToDevice, "eth9")
      << "precondition: the caller's binding reaches the config";
  EXPECT_EQ(config.dataSocketConfig("eth2").bindToDevice, "eth2")
      << "the transport's own device model must win";
  EXPECT_FALSE(config.dataSocketConfig(std::nullopt).bindToDevice.has_value())
      << "no device given must leave egress to the routing table";
}

// The NIC cap is a single field on the config, reachable by both MultiTransport
// and the benchmark. It used to be two constants -- one here and one on
// MultiTransportOptions -- which could drift apart while each looked right on
// its own, so the default is asserted rather than assumed.
TEST(TcpTransportConfigTest, FrontendDeviceCapDefaultsToTwo) {
  EXPECT_EQ(kDefaultMaxFrontendDevices, 2UL);
  EXPECT_EQ(TcpTransportConfig{}.maxFrontendDevices, 2UL);
}

// Zero means "no opinion", not "no devices": a caller threading an unset flag
// through would otherwise ask discovery for nothing and get a transport bound
// to no NIC at all.
TEST(TcpTransportConfigTest, FrontendDeviceCapOfZeroMeansDefault) {
  const std::string prefix{kDefaultFrontendDevicePrefix};
  const size_t capacity = frontendDeviceCapacity(prefix);
  if (capacity == 0) {
    GTEST_SKIP() << "no usable '" << prefix << "' device on this host";
  }
  TcpTransportConfig config;
  config.maxFrontendDevices = 0;
  EXPECT_EQ(
      config.resolveMaxFrontendDevices(prefix),
      std::min(kDefaultMaxFrontendDevices, capacity));
}

// The ceiling is what the host has, not a number in the source. A request above
// it is clamped rather than honoured, so a caller cannot end up with lanes
// bound to ports that do not exist.
TEST(TcpTransportConfigTest, FrontendDeviceCapIsClampedToTheHostsPortCount) {
  const std::string prefix{kDefaultFrontendDevicePrefix};
  const size_t capacity = frontendDeviceCapacity(prefix);
  if (capacity == 0) {
    GTEST_SKIP() << "no usable '" << prefix << "' device on this host";
  }
  TcpTransportConfig config;
  config.maxFrontendDevices = capacity + 100;
  EXPECT_EQ(config.resolveMaxFrontendDevices(prefix), capacity);
}

// Anything at or below capacity is the caller's to choose -- that is the point
// of making it configurable.
TEST(TcpTransportConfigTest, FrontendDeviceCapIsRaisableUpToCapacity) {
  const std::string prefix{kDefaultFrontendDevicePrefix};
  const size_t capacity = frontendDeviceCapacity(prefix);
  if (capacity == 0) {
    GTEST_SKIP() << "no usable '" << prefix << "' device on this host";
  }
  TcpTransportConfig config;
  config.maxFrontendDevices = capacity;
  EXPECT_EQ(config.resolveMaxFrontendDevices(prefix), capacity);
}

// Capacity is a property of the host, so it must agree with what discovery
// returns when asked for everything. Comparing the two rather than asserting a
// number keeps this a test of the accessor and not of the machine.
TEST(TcpTransportConfigTest, CapacityMatchesUnboundedDiscovery) {
  const std::string prefix{kDefaultFrontendDevicePrefix};
  EXPECT_EQ(
      frontendDeviceCapacity(prefix),
      enumerateFrontendDevices(prefix, std::numeric_limits<size_t>::max())
          .size());
}

// An unknown prefix has no ports, and capacity 0 must not be mistaken for "cap
// of none": resolve leaves the request alone so the caller's own no-device
// handling reports the real problem.
TEST(TcpTransportConfigTest, UnknownPrefixHasNoCapacityAndDoesNotClampToZero) {
  const std::string absent = "nosuchdev";
  ASSERT_EQ(frontendDeviceCapacity(absent), 0UL);
  TcpTransportConfig config;
  config.maxFrontendDevices = 4;
  EXPECT_EQ(config.resolveMaxFrontendDevices(absent), 4UL);
}

// Discovery has to honour a raised cap for the config field to mean anything.
// Bounded by what the host actually has: asserting a count would make this test
// a statement about the test machine rather than about the cap.
TEST(TcpTransportConfigTest, DiscoveryHonoursARaisedCap) {
  const std::string prefix{kDefaultFrontendDevicePrefix};
  const auto atTwo = enumerateFrontendDevices(prefix, 2);
  const auto atCapacity =
      enumerateFrontendDevices(prefix, frontendDeviceCapacity(prefix));
  EXPECT_LE(atTwo.size(), 2UL);
  EXPECT_GE(atCapacity.size(), atTwo.size())
      << "a larger cap must not return fewer devices";
  if (atCapacity.size() > atTwo.size()) {
    // The lower cap must be a prefix of the higher one, or lane i would map to
    // a different physical port depending only on the cap.
    EXPECT_TRUE(std::equal(atTwo.begin(), atTwo.end(), atCapacity.begin()))
        << "device order must not depend on the cap";
  }
}

TEST_F(TcpTransportTopologyTest, FactoryPropagatesDisabledAsyncGetH2d) {
  TcpTransportConfig config;
  config.asyncGetH2d = false;
  TcpTransportFactory factory(
      /*deviceId=*/-1,
      evbThread_->getEventBase(),
      config,
      /*host=*/"127.0.0.1");

  auto result = factory.createTransport(factory.getTopology());

  ASSERT_TRUE(result.hasValue()) << result.error().message();
  auto* transport = dynamic_cast<TcpTransport*>(result.value().get());
  ASSERT_NE(transport, nullptr);
  EXPECT_FALSE(transport->asyncGetH2dEnabled());
}

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

// ---------------------------------------------------------------------------
// TcpTransportInfo: multi-endpoint wire format for device striping
// ---------------------------------------------------------------------------

// The compatibility claim for striping rests entirely on this: one endpoint has
// to serialize to exactly the pre-striping bytes, or every existing peer
// breaks.
TEST(TcpTransportInfoTest, SingleEndpointWireIsUnchanged) {
  TcpTransportInfo info;
  info.host = "2401:db00::1";
  info.port = 4242;

  const auto bytes = info.serialize();

  std::vector<uint8_t> expected(sizeof(TcpTransportInfo::Header));
  TcpTransportInfo::Header header{
      .port = 4242,
      .hostLen = static_cast<uint16_t>(info.host.size()),
  };
  std::memcpy(expected.data(), &header, sizeof(header));
  expected.insert(expected.end(), info.host.begin(), info.host.end());

  EXPECT_EQ(std::vector<uint8_t>(bytes.begin(), bytes.end()), expected)
      << "a single-device transport must stay byte-identical to a peer that "
         "predates striping";
}

TEST(TcpTransportInfoTest, ExtraEndpointsRoundTrip) {
  TcpTransportInfo info;
  info.host = "2401:db00::1";
  info.port = 100;
  info.extraEndpoints.push_back({"2401:db00::2", 200});
  info.extraEndpoints.push_back({"2401:db00::3", 300});

  auto parsed = TcpTransportInfo::deserialize(info.serialize());

  ASSERT_TRUE(parsed.hasValue()) << parsed.error().message();
  EXPECT_EQ(parsed.value().host, "2401:db00::1");
  EXPECT_EQ(parsed.value().port, 100);
  EXPECT_EQ(parsed.value().endpointCount(), 3u);
  ASSERT_EQ(parsed.value().extraEndpoints.size(), 2u);
  EXPECT_EQ(parsed.value().extraEndpoints[0].host, "2401:db00::2");
  EXPECT_EQ(parsed.value().extraEndpoints[0].port, 200);
  EXPECT_EQ(parsed.value().extraEndpoints[1].host, "2401:db00::3");
  EXPECT_EQ(parsed.value().extraEndpoints[1].port, 300);
}

// endpointAt is what maps a lane to its device, so an off-by-one here would put
// a lane's payload on the wrong NIC.
TEST(TcpTransportInfoTest, EndpointAtIndexesFromZero) {
  TcpTransportInfo info;
  info.host = "2401:db00::1";
  info.port = 100;
  info.extraEndpoints.push_back({"2401:db00::2", 200});

  EXPECT_EQ(info.endpointAt(0).host, "2401:db00::1");
  EXPECT_EQ(info.endpointAt(0).port, 100);
  EXPECT_EQ(info.endpointAt(1).host, "2401:db00::2");
  EXPECT_EQ(info.endpointAt(1).port, 200);
  // Out of range clamps to endpoint 0 rather than reading past the end.
  EXPECT_EQ(info.endpointAt(2).host, "2401:db00::1");
}

// The old format enforced an exact size. That check is what rejects a corrupt
// or truncated info, so parsing extras must not silently accept a short tail.
TEST(TcpTransportInfoTest, RejectsTruncatedExtraEndpoint) {
  TcpTransportInfo info;
  info.host = "2401:db00::1";
  info.port = 100;
  info.extraEndpoints.push_back({"2401:db00::2", 200});

  auto bytes = info.serialize();
  std::vector<uint8_t> truncated(bytes.begin(), bytes.end() - 4);

  auto parsed = TcpTransportInfo::deserialize(truncated);

  ASSERT_TRUE(parsed.hasError())
      << "a truncated extra endpoint must be rejected, not silently dropped";
  EXPECT_EQ(parsed.error().code(), ErrCode::InvalidArgument);
}

TEST(TcpTransportInfoTest, RejectsTrailingGarbage) {
  TcpTransportInfo info;
  info.host = "2401:db00::1";
  info.port = 100;

  auto bytes = info.serialize();
  std::vector<uint8_t> extended(bytes.begin(), bytes.end());
  extended.push_back(0x01); // shorter than a Header, so not a valid endpoint

  auto parsed = TcpTransportInfo::deserialize(extended);

  ASSERT_TRUE(parsed.hasError());
  EXPECT_EQ(parsed.error().code(), ErrCode::InvalidArgument);
}

// Placement is derived from the lane index on both sides rather than
// negotiated, so a device-count mismatch has to be caught up front. Left
// undetected it would put lanes on the wrong NIC, which is the exact class of
// silent mislabeling that motivated device binding.
TEST_F(TcpTransportConnectTest, RejectsPeerWithDifferentDeviceCount) {
  controller::TcpSocketConfig socketConfig;
  socketConfig.connTimeout = std::chrono::seconds{2};
  TcpTransportConfig config{socketConfig};
  config.numSocketsPerDevice = 2;
  // The device list is left empty, so this transport binds exactly one
  // endpoint. That is the case being contrasted against a peer that advertises
  // two.
  auto local = std::make_unique<TcpTransport>(
      /*deviceId=*/-1,
      evbThread_->getEventBase(),
      registry_,
      config,
      /*host=*/"127.0.0.1");

  const TransportInfo self = local->bind();
  ASSERT_FALSE(self.empty());
  auto peer = TcpTransportInfo::deserialize(self);
  ASSERT_TRUE(peer.hasValue()) << peer.error().message();

  TcpTransportInfo striped = peer.value();
  striped.port = static_cast<uint16_t>(striped.port + 1);
  striped.extraEndpoints.push_back({striped.host, striped.port});

  const Status status = local->connect(striped.serialize());

  ASSERT_TRUE(status.hasError())
      << "a peer striping across more devices must be rejected, not silently "
         "collapsed onto one NIC";
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
  local->shutdown();
}

// There is no longer a guard for "fewer lanes than devices": lanes are counted
// per device, so the product is always at least the device count and that state
// is unreachable by construction.

} // namespace uniflow
