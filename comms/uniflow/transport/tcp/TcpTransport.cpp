// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/tcp/TcpTransport.h"

#include <arpa/inet.h>
#include <netinet/in.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/tcp/TcpRegistrationHandle.h"
#include "comms/uniflow/transport/tcp/TcpWireProtocol.h"

namespace uniflow {

// Frontend data NICs to stripe TCP lanes across, one port per card where
// possible.
//
// Candidates are found by name prefix because nothing else separates them:
// every eth port here is a 200G link that is up, so speed carries no signal,
// and the backend fabric (beth*) is addressed exactly like the frontend.
//
// Selection then spreads across PCI cards rather than taking the first two
// names, because a name sort always lands on a single card: the two ports of
// one card are adjacent in the numbering, being functions .0 and .1 of one PCI
// device.
//
// Both ports of a card share its upstream bandwidth, and that shows up as a
// ceiling rather than as a fixed cost: no single-card pair beat the best
// two-card pairs, while half the two-card pairs beat every single-card one, and
// the medians differ little and overlap. Spreading removes a cap rather than
// reliably buying a fixed gain.
//
// When only one card is present its second port is used: a second port on the
// same card is worth far more than no second port at all.
//
// Order is deterministic -- cards by PCI address, ports by name -- because lane
// i maps to device i on both peers, so an unstable order would pair a different
// physical port from one run to the next.
//
// Usability is delegated to deviceGlobalIpv6 rather than re-derived: it already
// applies the address-flag rules, and it is what bind() will call for the
// address, so discovery cannot disagree with what actually gets bound.
size_t TcpTransport::adaptiveGetChunk(size_t len, size_t laneCount) {
  if (len == 0 || len > kMaxChunkSize || laneCount <= 1) {
    return kMaxChunkSize;
  }
  const size_t perLane = (len + laneCount - 1) / laneCount;
  return std::max(perLane, kMinAdaptiveChunkSize);
}

std::vector<std::string> enumerateFrontendDevices(
    const std::string& prefix,
    size_t maxDevices) {
  std::map<std::string, std::set<std::string>> byCard;
  std::error_code ec;
  std::filesystem::directory_iterator it("/sys/class/net", ec);
  if (ec) {
    return {};
  }
  for (const auto& entry : it) {
    const std::string dev = entry.path().filename().string();
    if (dev.rfind(prefix, 0) != 0) {
      continue;
    }
    std::ifstream st("/sys/class/net/" + dev + "/operstate");
    std::string state;
    if (!st.is_open() || !(st >> state) || state != "up") {
      continue;
    }
    if (!controller::deviceGlobalIpv6(dev)) {
      continue;
    }
    // The PCI function identifies the port, so drop it: the remaining
    // domain:bus:device is the card whose bandwidth the ports share.
    std::string card;
    const auto link =
        std::filesystem::read_symlink("/sys/class/net/" + dev + "/device", ec);
    if (!ec) {
      card = link.filename().string();
      const auto dot = card.rfind('.');
      if (dot != std::string::npos) {
        card.resize(dot);
      }
    }
    if (card.empty()) {
      // Topology unreadable; treating it as its own card spreads rather than
      // stacks, which is the safer guess.
      card = dev;
    }
    byCard[card].insert(dev);
  }

  std::vector<std::string> devices;
  for (size_t round = 0; devices.size() < maxDevices; ++round) {
    bool tookAny = false;
    for (const auto& [card, ports] : byCard) {
      if (ports.size() <= round) {
        continue;
      }
      auto port = ports.begin();
      std::advance(port, round);
      devices.push_back(*port);
      tookAny = true;
      if (devices.size() == maxDevices) {
        break;
      }
    }
    if (!tookAny) {
      break;
    }
  }
  return devices;
}

namespace {
// Fallback when TcpSocketConfig::connTimeout is unset.
constexpr int kDefaultHandshakeTimeoutSeconds = 30;
} // namespace

namespace {

std::vector<uint8_t> makeHeaderFrame(TcpOp op, uint64_t reqId) {
  TcpMsgHeader header;
  header.op = static_cast<uint8_t>(op);
  header.reqId = reqId;
  return serializeTcpHeader(header);
}

// Representation-independent ordering key for a bind host. Parses IPv6 then
// IPv4 to raw address bytes so different textual forms of the same address
// (e.g. 2001:db8::1 vs its fully-expanded form) order and compare equal on both
// peers; falls back to the raw string for non-numeric hosts. Used only for
// deterministic listener/dialer role assignment.
std::string hostOrderKey(const std::string& host) {
  in6_addr a6{};
  if (::inet_pton(AF_INET6, host.c_str(), &a6) == 1) {
    return "6:" + std::string(reinterpret_cast<const char*>(&a6), sizeof(a6));
  }
  in_addr a4{};
  if (::inet_pton(AF_INET, host.c_str(), &a4) == 1) {
    return "4:" + std::string(reinterpret_cast<const char*>(&a4), sizeof(a4));
  }
  return "s:" + host;
}

} // namespace

// ---------------------------------------------------------------------------
// TcpTransportInfo
// ---------------------------------------------------------------------------

TcpTransportInfo::Endpoint TcpTransportInfo::endpointAt(size_t index) const {
  if (index == 0 || index > extraEndpoints.size()) {
    return Endpoint{host, port};
  }
  return extraEndpoints[index - 1];
}

TransportInfo TcpTransportInfo::serialize() const {
  Header header{
      .port = port,
      .hostLen = static_cast<uint16_t>(host.size()),
  };
  size_t extraBytes = 0;
  for (const auto& ep : extraEndpoints) {
    extraBytes += sizeof(Header) + ep.host.size();
  }
  TransportInfo data(sizeof(Header) + host.size() + extraBytes);
  std::memcpy(data.data(), &header, sizeof(header));
  if (!host.empty()) {
    std::memcpy(data.data() + sizeof(header), host.data(), host.size());
  }
  // Same {port, hostLen, host} shape repeated, so with no extra endpoints the
  // bytes are identical to a build that predates striping.
  size_t offset = sizeof(header) + host.size();
  for (const auto& ep : extraEndpoints) {
    Header extra{
        .port = ep.port,
        .hostLen = static_cast<uint16_t>(ep.host.size()),
    };
    std::memcpy(data.data() + offset, &extra, sizeof(extra));
    offset += sizeof(extra);
    if (!ep.host.empty()) {
      std::memcpy(data.data() + offset, ep.host.data(), ep.host.size());
      offset += ep.host.size();
    }
  }
  return data;
}

Result<TcpTransportInfo> TcpTransportInfo::deserialize(
    std::span<const uint8_t> data) {
  if (data.size() < sizeof(Header)) {
    return Err(ErrCode::InvalidArgument, "tcp transport info is truncated");
  }

  Header header;
  std::memcpy(&header, data.data(), sizeof(header));
  if (data.size() < sizeof(header) + header.hostLen) {
    return Err(ErrCode::InvalidArgument, "tcp transport info size mismatch");
  }

  TcpTransportInfo info;
  info.port = header.port;
  info.host.assign(
      reinterpret_cast<const char*>(data.data() + sizeof(header)),
      header.hostLen);

  // Anything left over is extra endpoints. Consuming exactly the remainder
  // keeps the old exact-size guarantee: trailing junk is still rejected.
  size_t offset = sizeof(header) + header.hostLen;
  while (offset < data.size()) {
    if (data.size() - offset < sizeof(Header)) {
      return Err(
          ErrCode::InvalidArgument,
          "tcp transport info has a truncated extra endpoint");
    }
    Header extra;
    std::memcpy(&extra, data.data() + offset, sizeof(extra));
    offset += sizeof(extra);
    if (data.size() - offset < extra.hostLen) {
      return Err(
          ErrCode::InvalidArgument,
          "tcp transport info extra endpoint host is truncated");
    }
    Endpoint ep;
    ep.port = extra.port;
    ep.host.assign(
        reinterpret_cast<const char*>(data.data() + offset), extra.hostLen);
    offset += extra.hostLen;
    info.extraEndpoints.push_back(std::move(ep));
  }
  return info;
}

// ---------------------------------------------------------------------------
// TcpTransport
// ---------------------------------------------------------------------------

TcpTransport::TcpTransport(
    int deviceId,
    EventBase* evb,
    std::shared_ptr<TcpSegmentRegistry> registry,
    TcpTransportConfig config,
    std::string host,
    std::shared_ptr<CudaApi> cudaApi)
    : deviceId_(deviceId),
      evb_(evb),
      registry_(std::move(registry)),
      cudaApi_(std::move(cudaApi)),
      config_(std::move(config)) {
  if (!registry_) {
    registry_ = std::make_shared<TcpSegmentRegistry>();
  }
  if (!cudaApi_) {
    cudaApi_ = std::make_shared<CudaApi>();
  }
  if (!host.empty()) {
    host_ = std::move(host);
  }
  h2dState_ = std::make_shared<H2dPollState>();
  h2dState_->evb = evb_;
  h2dState_->cudaApi = cudaApi_;
}

Status TcpTransport::hostFromDevice(
    void* hostDst,
    const void* devSrc,
    size_t len,
    int deviceId,
    void* stream) {
  if (len == 0) {
    return Ok();
  }
  auto s = static_cast<cudaStream_t>(stream);
  CudaDeviceGuard guard(*cudaApi_, deviceId);
  auto st =
      cudaApi_->memcpyAsync(hostDst, devSrc, len, cudaMemcpyDeviceToHost, s);
  if (!st) {
    return st;
  }
  return cudaApi_->streamSynchronize(s);
}

Status TcpTransport::deviceFromHost(
    void* devDst,
    const void* hostSrc,
    size_t len,
    int deviceId,
    void* stream) {
  if (len == 0) {
    return Ok();
  }
  auto s = static_cast<cudaStream_t>(stream);
  CudaDeviceGuard guard(*cudaApi_, deviceId);
  auto st =
      cudaApi_->memcpyAsync(devDst, hostSrc, len, cudaMemcpyHostToDevice, s);
  if (!st) {
    return st;
  }
  return cudaApi_->streamSynchronize(s);
}

TcpTransport::~TcpTransport() {
  shutdown();
}

TransportInfo TcpTransport::bind() {
  std::lock_guard<std::mutex> lk(lifecycleMu_);
  if (shutdown_.load(std::memory_order_acquire)) {
    UNIFLOW_LOG_ERROR("TcpTransport::bind: transport is already shut down");
    return TransportInfo{};
  }
  // Refuse a re-bind rather than re-arming Initialized, which is the only state
  // connect() admits. Re-arming lets a second connect() reach
  // establishLanes(), whose first act is to clear lanes_ -- destroying the live
  // lanes' joinable reader/sender threads, and ~std::thread on a joinable
  // thread calls std::terminate. The servers_.clear() below would already have
  // dropped the listener out from under the current connection. Error is
  // terminal here too: nothing retries a failed bind, and a transport that
  // never came up should not be revived.
  if (state_ != TransportState::Disconnected) {
    UNIFLOW_LOG_ERROR("TcpTransport::bind: transport is already bound");
    return TransportInfo{};
  }
  // One listener per device when striping, otherwise a single listener on host_
  // with egress left to the routing table. Each device's listener binds that
  // device's own address *and* sets SO_BINDTODEVICE, because accepted sockets
  // inherit the listener's device binding and that inheritance is what puts
  // `get` payload on the intended NIC.
  std::vector<std::pair<std::string, std::optional<std::string>>> targets;
  if (config_.bindToDevices.empty()) {
    targets.emplace_back(host_, std::nullopt);
  } else {
    for (const auto& device : config_.bindToDevices) {
      auto addr = controller::deviceGlobalIpv6(device);
      if (!addr) {
        UNIFLOW_LOG_ERROR(
            "TcpTransport::bind: cannot resolve device {}: {}",
            device,
            addr.error().message());
        state_ = TransportState::Error;
        return TransportInfo{};
      }
      targets.emplace_back(addr.value(), device);
    }
  }

  servers_.clear();
  localEndpoints_.clear();
  for (const auto& [addr, device] : targets) {
    auto socketConfig = config_.socketConfig;
    if (device) {
      socketConfig.bindToDevice = device;
    }
    auto server = std::make_unique<controller::AsyncTcpServer>(
        addr + ":0", socketConfig, *evb_);
    auto status = server->init();
    if (!status) {
      UNIFLOW_LOG_ERROR(
          "TcpTransport::bind: server init failed for {}{}: {}",
          addr,
          device ? fmt::format(" (dev {})", *device) : "",
          status.error().message());
      state_ = TransportState::Error;
      servers_.clear();
      localEndpoints_.clear();
      return TransportInfo{};
    }
    localEndpoints_.push_back({addr, static_cast<uint16_t>(server->getPort())});
    servers_.push_back(std::move(server));
  }

  // Endpoint 0 stays the transport's identity: it orders the listener/dialer
  // roles and it is what a single-device peer sees.
  host_ = localEndpoints_.front().host;
  port_ = localEndpoints_.front().port;
  state_ = TransportState::Initialized;

  TcpTransportInfo info;
  info.host = host_;
  info.port = port_;
  info.extraEndpoints.assign(
      localEndpoints_.begin() + 1, localEndpoints_.end());
  return info.serialize();
}

Status TcpTransport::connect(std::span<const uint8_t> remoteInfo) {
  // Held across the handshake wait below, so a concurrent shutdown() cannot
  // interleave with the installation of lanes_ at the end of
  // this function. The flag check is what stops a connect() queued behind a
  // completed shutdown() from bringing the transport back to life.
  std::lock_guard<std::mutex> lk(lifecycleMu_);
  if (shutdown_.load(std::memory_order_acquire)) {
    return Err(
        ErrCode::NotConnected, "tcp connect: transport is already shut down");
  }
  if (state_ != TransportState::Initialized) {
    return Err(
        ErrCode::InvalidArgument, "tcp connect: transport must be bound first");
  }

  auto peerResult = TcpTransportInfo::deserialize(remoteInfo);
  if (!peerResult) {
    state_ = TransportState::Error;
    return std::move(peerResult).error();
  }
  const auto peer = std::move(peerResult).value();

  // Deterministic role: the smaller (host, port) listens, the other dials.
  // Compare by the parsed binary address (hostOrderKey), not the raw string, so
  // different textual forms of the same IPv6 address can't make the two peers
  // disagree on ordering. Reject the degenerate identical-endpoint case, which
  // would otherwise leave both peers dialing and nobody accepting (silent
  // hang).
  const auto localKey = std::make_tuple(hostOrderKey(host_), port_);
  const auto peerKey = std::make_tuple(hostOrderKey(peer.host), peer.port);
  if (localKey == peerKey) {
    state_ = TransportState::Error;
    return Err(
        ErrCode::ConnectionFailed,
        "tcp connect: local and peer bind address are identical; cannot assign "
        "distinct listener/dialer roles");
  }
  const bool listener = localKey < peerKey;

  // Both handshake waits are bounded. AsyncAccept::accept() queues a promise
  // that is resolved only by an inbound connection or by teardown, so an
  // unbounded get() here wedges this thread forever if the dialing peer dies
  // between the bind-info exchange and its own connect(). Nothing would rescue
  // it: server_ is owned by this transport, whose connect() is the thing
  // parked. AsyncAccept also ignores acceptRetryCnt and AsyncConnect ignores
  // connectRetries, so no configured bound applies on its own.
  //
  // This matters beyond TCP: MultiTransport::connect() connects every
  // registered transport, so on AMD a wedged TCP handshake would stall
  // connection setup even for jobs whose data path is RDMA.
  const auto handshakeTimeout = config_.socketConfig.connTimeout.value_or(
      std::chrono::seconds{kDefaultHandshakeTimeoutSeconds});

  // Both peers derive lane-to-device placement from the lane index, so they
  // must agree on the device count. The dialer can check this up front because
  // the listener advertises one endpoint per device; catching it here gives a
  // clear error instead of a handshake timeout on the lanes that map to a
  // device the peer never bound.
  const size_t localDevices =
      config_.bindToDevices.empty() ? 1 : config_.bindToDevices.size();
  if (peer.endpointCount() != localDevices) {
    state_ = TransportState::Error;
    return Err(
        ErrCode::InvalidArgument,
        "tcp connect: peer advertised " + std::to_string(peer.endpointCount()) +
            " device endpoints, local is configured for " +
            std::to_string(localDevices) + "; both peers must agree");
  }

  // Lanes are configured per device, so every device gets a full complement and
  // no device can end up without a lane. It is the product that has to fit the
  // uint16_t the hello addresses lanes with.
  constexpr size_t kMaxLanes = 1024;
  const size_t lanesPerDevice =
      std::max<size_t>(config_.numSocketsPerDevice, 1);
  const size_t laneCount = lanesPerDevice * localDevices;
  if (laneCount > kMaxLanes) {
    state_ = TransportState::Error;
    return Err(
        ErrCode::InvalidArgument,
        "tcp connect: numSocketsPerDevice " + std::to_string(lanesPerDevice) +
            " across " + std::to_string(localDevices) + " devices is " +
            std::to_string(laneCount) + " lanes, above the " +
            std::to_string(kMaxLanes) + " the hello can address");
  }

  if (auto status = establishLanes(listener, peer, laneCount, handshakeTimeout);
      !status) {
    state_ = TransportState::Error;
    return status;
  }

  running_.store(true, std::memory_order_release);
  // Started only once every lane is installed, so a reader can never index a
  // lane connect() has not filled in yet.
  for (size_t i = 0; i < lanes_.size(); ++i) {
    lanes_[i]->reader = std::thread([this, i]() { readerLoop(i); });
    lanes_[i]->sender = std::thread([this, i]() { senderLoop(i); });
  }
  state_ = TransportState::Connected;
  UNIFLOW_LOG_INFO(
      "TcpTransport: connected (listener={} lanes={}) {}:{} <-> {}:{}",
      listener,
      lanes_.size(),
      host_,
      port_,
      peer.host,
      peer.port);
  return Ok();
}

// Fills lanes_ so that lane i on one peer is lane i on the other. With one lane
// this is exactly the pre-lane path: no hello is exchanged, so the wire stays
// byte-identical and a peer built before lanes existed still interoperates.
Status TcpTransport::establishLanes(
    bool listener,
    const TcpTransportInfo& peer,
    size_t laneCount,
    std::chrono::seconds handshakeTimeout) {
  lanes_.clear();
  lanes_.reserve(laneCount);
  for (size_t i = 0; i < laneCount; ++i) {
    lanes_.push_back(std::make_unique<TcpLane>());
  }
  const bool exchangeHello = laneCount > 1;

  if (listener) {
    if (servers_.empty()) {
      return Err(ErrCode::ConnectionFailed, "tcp connect: no server bound");
    }
    // Lane i lives on device i % D, so listener d owns exactly the lanes
    // congruent to d.
    //
    // Every listener must be armed before this thread blocks on any of them.
    // AsyncAccept registers its listen fd with the EventBase lazily, on the
    // first accept() call, and an accepted socket's magic exchange runs from
    // that fd's EPOLLIN handler. An unarmed listener therefore leaves a
    // dialed-in connection sitting in the kernel backlog with nobody completing
    // its handshake, so the dialer's exchange times out after 500ms. Accepting
    // device-by-device without pre-arming deadlocks as soon as one device owns
    // two lanes: this thread waits for device 0's second lane, which the dialer
    // only dials after device 1's lane succeeds, which it cannot.
    const size_t deviceCount = servers_.size();
    uint64_t sessionId = 0;
    std::vector<bool> filled(laneCount, false);
    std::vector<size_t> quota(deviceCount);
    for (size_t d = 0; d < deviceCount; ++d) {
      // Lanes d, d+D, d+2D, ... land on this listener.
      quota[d] =
          laneCount / deviceCount + (d < laneCount % deviceCount ? 1 : 0);
    }
    // One outstanding accept per listener, purely to arm the fds. Once armed a
    // listener keeps accepting and handshaking arrivals into readyConns_
    // whether or not this thread is currently waiting on it.
    std::vector<std::future<std::unique_ptr<controller::Conn>>> armed(
        deviceCount);
    for (size_t d = 0; d < deviceCount; ++d) {
      if (quota[d] > 0) {
        armed[d] = servers_[d]->accept();
      }
    }
    size_t accepted = 0;
    for (size_t dev = 0; dev < deviceCount; ++dev) {
      for (size_t n = 0; n < quota[dev]; ++n, ++accepted) {
        auto future = armed[dev].valid() ? std::move(armed[dev])
                                         : servers_[dev]->accept();
        if (future.wait_for(handshakeTimeout) != std::future_status::ready) {
          // shutdown() resolves the queued promise with nullptr (via teardown),
          // so the get() below returns immediately instead of blocking. Safe
          // from this thread: AsyncAccept::shutdown marshals teardown onto the
          // EventBase thread and waits when called from outside the loop.
          UNIFLOW_LOG_ERROR(
              "TcpTransport::connect: only {} of {} lanes dialed in within {}s; "
              "tearing down listener {}:{} (dev {} of {})",
              accepted,
              laneCount,
              handshakeTimeout.count(),
              localEndpoints_[dev].host,
              localEndpoints_[dev].port,
              dev,
              deviceCount);
          servers_[dev]->shutdown();
        }
        auto conn = future.get();
        if (!conn) {
          return Err(
              ErrCode::ConnectionFailed, "tcp connect: data connection failed");
        }

        // Accept order is not lane identity: the peer dials the lanes without
        // any ordering guarantee, so the index has to come off the wire.
        size_t laneIdx = accepted;
        if (exchangeHello) {
          std::vector<uint8_t> msg;
          auto received = conn->recv(msg).get();
          if (!received) {
            return Err(
                ErrCode::ConnectionFailed,
                "tcp connect: lane hello not received: " +
                    received.error().message());
          }
          auto helloResult = TcpLaneHello::deserialize(
              std::span<const uint8_t>{msg.data(), received.value()});
          if (!helloResult) {
            return std::move(helloResult).error();
          }
          const auto hello = helloResult.value();
          if (hello.laneCount != laneCount) {
            return Err(
                ErrCode::InvalidArgument,
                "tcp connect: peer configured " +
                    std::to_string(hello.laneCount) + " lanes, local is " +
                    std::to_string(laneCount) + "; both peers must agree");
          }
          if (hello.laneIndex >= laneCount) {
            return Err(
                ErrCode::InvalidArgument,
                "tcp connect: lane index " + std::to_string(hello.laneIndex) +
                    " out of range for " + std::to_string(laneCount) +
                    " lanes");
          }
          if (accepted == 0) {
            sessionId = hello.sessionId;
          } else if (hello.sessionId != sessionId) {
            // A second dialer reaching this listener would otherwise take a
            // lane and leave the real peer one short, hanging both.
            return Err(
                ErrCode::ConnectionFailed,
                "tcp connect: lane session mismatch; another peer is dialing "
                "this listener");
          }
          laneIdx = hello.laneIndex;
          if (filled[laneIdx]) {
            return Err(
                ErrCode::ConnectionFailed,
                "tcp connect: duplicate lane index " + std::to_string(laneIdx));
          }
          // The mapping is derived, not negotiated, so a lane arriving on the
          // wrong listener means the peers disagree on device count. Left
          // undetected it would silently place this lane's payload on the wrong
          // NIC, which is exactly the mislabeling this feature exists to fix.
          if (laneIdx % deviceCount != dev) {
            return Err(
                ErrCode::InvalidArgument,
                "tcp connect: lane " + std::to_string(laneIdx) +
                    " arrived on device " + std::to_string(dev) +
                    " but maps to " + std::to_string(laneIdx % deviceCount) +
                    "; peers disagree on device count (" +
                    std::to_string(deviceCount) + " local)");
          }
        }
        filled[laneIdx] = true;
        lanes_[laneIdx]->conn = std::move(conn);
      }
    }
  } else {
    // Only has to distinguish this dialer from another one racing for the same
    // listener, so the local port plus a clock reading is enough and avoids
    // pulling in a random-number dependency.
    const uint64_t sessionId =
        (static_cast<uint64_t>(port_) << 48) ^
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    for (size_t i = 0; i < laneCount; ++i) {
      // Same derived mapping as the listener: lane i on device i % D, dialing
      // the endpoint the peer advertised for that device. Binding the dialer
      // matters for `put` and for the request headers; the listener's own
      // binding is what carries `get` payload.
      const size_t deviceCount =
          config_.bindToDevices.empty() ? 1 : config_.bindToDevices.size();
      const size_t dev = i % deviceCount;
      auto socketConfig = config_.socketConfig;
      if (!config_.bindToDevices.empty()) {
        socketConfig.bindToDevice = config_.bindToDevices[dev];
      }
      const auto target = peer.endpointAt(dev);
      controller::AsyncTcpClient client(socketConfig, *evb_);
      auto future =
          client.connect(target.host + ":" + std::to_string(target.port));
      if (future.wait_for(handshakeTimeout) != std::future_status::ready) {
        UNIFLOW_LOG_ERROR(
            "TcpTransport::connect: lane {} dial to {}:{} did not complete "
            "within {}s",
            i,
            target.host,
            target.port,
            handshakeTimeout.count());
        // No teardown hook on the client side; abandon the attempt. The future
        // owns its own state, so letting it go out of scope is safe.
        return Err(ErrCode::ConnectionFailed, "tcp connect: dial timed out");
      }
      auto conn = future.get();
      if (!conn) {
        return Err(
            ErrCode::ConnectionFailed, "tcp connect: data connection failed");
      }
      if (exchangeHello) {
        TcpLaneHello hello;
        hello.laneIndex = static_cast<uint16_t>(i);
        hello.laneCount = static_cast<uint16_t>(laneCount);
        hello.sessionId = sessionId;
        // Named, so the buffer outlives the send rather than depending on
        // temporary lifetime across the future.
        const auto bytes = hello.serialize();
        auto sent = conn->send(bytes).get();
        if (!sent) {
          return Err(
              ErrCode::ConnectionFailed,
              "tcp connect: lane hello send failed: " + sent.error().message());
        }
      }
      lanes_[i]->conn = std::move(conn);
    }
  }
  return Ok();
}

size_t TcpTransport::laneCapBytes() const {
  const size_t lanes = lanes_.empty() ? 1 : lanes_.size();
  return kMaxOutQueueBytes / lanes;
}

size_t TcpTransport::pickLane() {
  if (lanes_.size() <= 1) {
    return 0;
  }
  return static_cast<size_t>(
      nextLane_.fetch_add(1, std::memory_order_relaxed) % lanes_.size());
}

controller::Conn* TcpTransport::primaryConn() const {
  if (lanes_.empty() || lanes_[0] == nullptr) {
    return nullptr;
  }
  return lanes_[0]->conn.get();
}

Result<const TcpRemoteRegistrationHandle*> TcpTransport::findRemoteHandle(
    const RemoteRegisteredSegment::Span& span) const {
  for (const auto& handle : span.handles_) {
    if (auto* tcp =
            dynamic_cast<const TcpRemoteRegistrationHandle*>(handle.get())) {
      return tcp;
    }
  }
  return Err(
      ErrCode::InvalidArgument, "tcp: no TCP remote registration handle found");
}

std::future<Status> TcpTransport::put(
    std::span<const TransferRequest> requests,
    const RequestOptions& options) {
  if (state_ != TransportState::Connected ||
      connBroken_.load(std::memory_order_acquire)) {
    return make_ready_future<Status>(
        Err(ErrCode::NotConnected, "tcp put: not connected"));
  }
  if (requests.empty()) {
    return make_ready_future<Status>(Ok());
  }

  auto state = std::make_shared<TcpOpState>();
  auto future = state->promise.get_future();

  // Pre-flight. Everything that can fail is settled before the first frame is
  // queued, because once a frame reaches enqueueFrame the sender thread may
  // flush it and a Write the peer has applied cannot be recalled. A bail from
  // the middle of the send loop therefore reports failure to the caller while a
  // partial write has landed remotely, with nothing telling the peer about it.
  std::vector<PlannedPutFrame> planned;
  std::vector<PlannedChunk> chunks;

  for (const auto& req : requests) {
    if (req.local.size() != req.remote.size()) {
      state->fail(
          Err(ErrCode::InvalidArgument,
              "tcp put: local and remote buffer sizes must match"));
      return future;
    }
    auto remoteHandle = findRemoteHandle(req.remote);
    if (!remoteHandle) {
      state->fail(std::move(remoteHandle).error());
      return future;
    }
    const size_t len = req.local.size();
    const bool vram = req.local.memType() == MemoryType::VRAM;
    const int deviceId = req.local.deviceId();
    // Probed once per request rather than once per chunk: the cost is a device
    // set/restore, and the point is to reject an unusable device while the peer
    // is still untouched.
    if (vram && len > 0) {
      if (auto usable = validateDeviceForStaging(deviceId); !usable) {
        state->fail(std::move(usable));
        return future;
      }
    }
    const auto segId = remoteHandle.value()->segId();
    const auto baseOffset = static_cast<uint64_t>(req.remote.remoteOffset_);
    const auto* src = static_cast<const uint8_t*>(req.local.data());

    size_t off = 0;
    do {
      const size_t chunk = std::min(kMaxChunkSize, len - off);
      const uint64_t reqId = nextReqId_.fetch_add(1, std::memory_order_relaxed);
      chunks.push_back(
          PlannedChunk{
              reqId, chunk, TcpInflight{state, nullptr, chunk, false}});
      planned.push_back(
          PlannedPutFrame{
              reqId,
              segId,
              baseOffset + off,
              src + off,
              chunk,
              vram,
              deviceId});
      off += chunk;
    } while (off < len);
  }
  // Exact by construction: one entry per frame this put will send.
  state->remaining = chunks.size();

  if (auto admitted = admitInflightBulk(chunks); !admitted) {
    state->fail(std::move(admitted));
    return future;
  }

  // Commit. Only a genuine staging error or transport teardown remains
  // reachable, and both abandon the reservations for frames never queued.
  //
  // VRAM chunks are staged and queued in waves rather than one at a time. A
  // per-chunk commit puts each Write in the queue as soon as its own copy
  // finishes, so a copy that fails partway through a transfer leaves the peer
  // holding the chunks that went before it -- a partial write at offsets the
  // caller is never told about, and one the peer has no way to notice. A wave
  // is queued only once every copy in it has succeeded.
  void* stream = options.stream.has_value()
      ? static_cast<void*>(options.stream.value())
      : nullptr;
  size_t idx = 0;
  while (idx < planned.size()) {
    const auto& first = planned[idx];
    if (!first.vram || first.len == 0) {
      // A host memcpy cannot fail and cannot park this thread, so there is
      // nothing to stage and nothing a wave would protect.
      std::vector<uint8_t> frame(sizeof(TcpMsgHeader) + first.len);
      TcpMsgHeader header;
      header.op = static_cast<uint8_t>(TcpOp::Write);
      header.reqId = first.reqId;
      header.segId = first.segId;
      header.offset = first.offset;
      header.len = static_cast<uint64_t>(first.len);
      std::memcpy(frame.data(), &header, sizeof(header));
      if (first.len > 0) {
        std::memcpy(frame.data() + sizeof(header), first.src, first.len);
      }
      if (!enqueueFrame(std::move(frame), /*mayBlock=*/true)) {
        abandonInflight(chunks, idx);
        state->fail(
            Err(ErrCode::NotConnected,
                "tcp put: transport closed before the write was queued"));
        return future;
      }
      ++idx;
      continue;
    }

    size_t waveEnd = idx;
    while (waveEnd < planned.size() && planned[waveEnd].vram &&
           planned[waveEnd].len > 0 && waveEnd - idx < kMaxPutWaveChunks) {
      ++waveEnd;
    }
    auto staged = stagePutWave(
        std::span<const PlannedPutFrame>(planned).subspan(idx, waveEnd - idx),
        stream);
    if (!staged) {
      abandonInflight(chunks, idx);
      state->fail(std::move(staged).error());
      return future;
    }
    if (!enqueueFrames(std::move(staged).value(), /*mayBlock=*/true)) {
      abandonInflight(chunks, idx);
      state->fail(
          Err(ErrCode::NotConnected,
              "tcp put: transport closed before the write was queued"));
      return future;
    }
    idx = waveEnd;
  }

  return future;
}

Result<std::vector<TcpFrame>> TcpTransport::stagePutWave(
    std::span<const PlannedPutFrame> wave,
    void* stream) {
  auto pool = stagingPool();
  if (!pool) {
    return std::move(pool).error();
  }
  // All-or-nothing, and this thread holds no slab while it waits: a caller that
  // took what was free and waited for the rest would deadlock against another
  // doing the same.
  auto leases = pool.value()->acquire(wave.size());
  if (!leases) {
    return std::move(leases).error();
  }

  auto s = static_cast<cudaStream_t>(stream);
  std::vector<TcpFrame> frames;
  frames.reserve(wave.size());
  // Devices whose copies were launched, so the wait below covers each of them
  // once. A bare synchronize would only cover whichever device happened to be
  // current, leaving copies on any other still running.
  std::vector<int> launchedDevices;
  Status staging = Ok();
  for (size_t i = 0; i < wave.size(); ++i) {
    const auto& chunk = wave[i];
    TcpMsgHeader header;
    header.op = static_cast<uint8_t>(TcpOp::Write);
    header.reqId = chunk.reqId;
    header.segId = chunk.segId;
    header.offset = chunk.offset;
    header.len = static_cast<uint64_t>(chunk.len);
    // The frame owns the slab from here on, so every path out of this function
    // returns it exactly once.
    frames.emplace_back(
        std::move(leases.value()[i]), sizeof(TcpMsgHeader) + chunk.len);
    std::memcpy(frames.back().mutableData(), &header, sizeof(header));
    try {
      CudaDeviceGuard guard(*cudaApi_, chunk.deviceId);
      staging = cudaApi_->memcpyAsync(
          frames.back().mutableData() + sizeof(TcpMsgHeader),
          chunk.src,
          chunk.len,
          cudaMemcpyDeviceToHost,
          s);
    } catch (const std::exception& e) {
      staging =
          Err(ErrCode::InvalidArgument,
              "tcp put: VRAM staging needs a selectable deviceId, got " +
                  std::to_string(chunk.deviceId) + ": " + e.what());
    }
    if (!staging) {
      break;
    }
    if (std::find(
            launchedDevices.begin(), launchedDevices.end(), chunk.deviceId) ==
        launchedDevices.end()) {
      launchedDevices.push_back(chunk.deviceId);
    }
  }
  // One wait per wave rather than one per chunk: the copies are already in
  // flight together, and waiting on each in turn is what serialised staging
  // against itself.
  for (auto deviceId : launchedDevices) {
    try {
      CudaDeviceGuard guard(*cudaApi_, deviceId);
      if (auto st = cudaApi_->streamSynchronize(s); !st && staging) {
        staging = std::move(st);
      }
    } catch (const std::exception&) {
      // The device is already unusable; there is nothing left to wait for.
    }
  }
  if (!staging) {
    return std::move(staging).error();
  }
  return frames;
}

std::future<Status> TcpTransport::get(
    std::span<const TransferRequest> requests,
    const RequestOptions& options) {
  if (state_ != TransportState::Connected ||
      connBroken_.load(std::memory_order_acquire)) {
    return make_ready_future<Status>(
        Err(ErrCode::NotConnected, "tcp get: not connected"));
  }
  if (requests.empty()) {
    return make_ready_future<Status>(Ok());
  }

  auto state = std::make_shared<TcpOpState>();
  auto future = state->promise.get_future();

  // Pre-flight. Same reason as put(): nothing that can fail may run after the
  // first frame is queued, because a queued ReadRequest is already on its way
  // to the peer and cannot be recalled. Resolving segIds here rather than in
  // the emit loop is what makes a rejected get() leave the peer untouched.
  size_t totalChunks = 0;
  std::vector<uint64_t> segIds;
  segIds.reserve(requests.size());
  for (const auto& req : requests) {
    if (req.local.size() != req.remote.size()) {
      state->fail(
          Err(ErrCode::InvalidArgument,
              "tcp get: local and remote buffer sizes must match"));
      return future;
    }
    auto remoteHandle = findRemoteHandle(req.remote);
    if (!remoteHandle) {
      state->fail(std::move(remoteHandle).error());
      return future;
    }
    segIds.push_back(remoteHandle.value()->segId());
    const size_t len = req.local.size();
    const size_t chunkSize = adaptiveGetChunk(len, lanes_.size());
    totalChunks += (len == 0) ? 1 : (len + chunkSize - 1) / chunkSize;
  }
  state->remaining = totalChunks;

  if (config_.asyncGetH2d) {
    const bool needsReceivePool = std::any_of(
        requests.begin(), requests.end(), [](const TransferRequest& req) {
          return req.local.size() > 0 &&
              req.local.memType() == MemoryType::VRAM;
        });
    if (needsReceivePool) {
      (void)ensureReceivePool();
    }
  }

  for (size_t reqIdx = 0; reqIdx < requests.size(); ++reqIdx) {
    const auto& req = requests[reqIdx];
    const uint64_t segId = segIds[reqIdx];
    const uint64_t baseOffset = static_cast<uint64_t>(req.remote.remoteOffset_);
    const size_t len = req.local.size();
    const MemoryType memType = req.local.memType();
    const int deviceId = req.local.deviceId();
    auto* dst = static_cast<uint8_t*>(req.local.mutable_data());

    const size_t chunkSize = adaptiveGetChunk(len, lanes_.size());
    size_t off = 0;
    do {
      const size_t chunk = std::min(chunkSize, len - off);
      const uint64_t reqId = nextReqId_.fetch_add(1, std::memory_order_relaxed);
      if (auto admitted = admitInflight(
              reqId,
              TcpInflight{
                  state,
                  dst + off,
                  chunk,
                  true,
                  memType,
                  deviceId,
                  options.stream.has_value()
                      ? static_cast<void*>(options.stream.value())
                      : nullptr});
          admitted.hasError()) {
        state->fail(std::move(admitted));
        return future;
      }
      TcpMsgHeader header;
      header.op = static_cast<uint8_t>(TcpOp::ReadRequest);
      header.reqId = reqId;
      header.segId = segId;
      header.offset = baseOffset + off;
      header.len = static_cast<uint64_t>(chunk);
      if (!enqueueFrame(serializeTcpHeader(header), /*mayBlock=*/true)) {
        state->fail(
            Err(ErrCode::NotConnected,
                "tcp get: transport closed before the read was queued"));
        return future;
      }
      off += chunk;
    } while (off < len);
  }

  return future;
}

std::shared_ptr<TcpPinnedSlabPool> TcpTransport::ensureReceivePool() {
  if (auto pool = std::atomic_load_explicit(
          &receiveSlabPool_, std::memory_order_acquire)) {
    return pool;
  }
  std::lock_guard<std::mutex> lk(receivePoolCreateMu_);
  if (auto pool = std::atomic_load_explicit(
          &receiveSlabPool_, std::memory_order_relaxed)) {
    return pool;
  }
  if (receivePoolUnavailable_ || cudaApi_ == nullptr) {
    return nullptr;
  }
  auto pool = TcpPinnedSlabPool::create(
      cudaApi_, kMaxFrameSize, kReceiveSlabCount, /*reservedForReader=*/0);
  if (!pool) {
    receivePoolUnavailable_ = true;
    UNIFLOW_LOG_WARN(
        "tcp get: pinned receive pool unavailable; using vector fallback: {}",
        pool.error().message());
    return nullptr;
  }
  std::atomic_store_explicit(
      &receiveSlabPool_, pool.value(), std::memory_order_release);
  return pool.value();
}

std::shared_ptr<TcpPinnedSlabPool> TcpTransport::receivePoolIfCreated() {
  return std::atomic_load_explicit(
      &receiveSlabPool_, std::memory_order_acquire);
}

Result<std::shared_ptr<TcpPinnedSlabPool>> TcpTransport::stagingPool() {
  std::lock_guard<std::mutex> lk(poolMu_);
  if (slabPool_ != nullptr) {
    return slabPool_;
  }
  if (cudaApi_ == nullptr) {
    return Err(ErrCode::InvalidArgument, "tcp read: no CUDA API for VRAM");
  }
  // Header and payload contiguous in one slab, so a staged frame is still a
  // single buffer and the send path needs no scatter-gather.
  auto pool = TcpPinnedSlabPool::create(
      cudaApi_,
      sizeof(TcpMsgHeader) + kMaxChunkSize,
      kStagingSlabCount,
      kStagingSlabsReservedForReader);
  if (!pool) {
    return std::move(pool).error();
  }
  slabPool_ = pool.value();
  return slabPool_;
}

Status TcpTransport::respondToVramRead(
    const TcpMsgHeader& replyHeader,
    TcpSegmentRegistry::Lease lease) {
  // A same-version peer chunks at kMaxChunkSize, so this only rejects a
  // version-skewed peer built with a larger one. Per request, because the
  // sender treats an oversized send as fatal and would take every unrelated
  // transfer on the connection down with it.
  if (replyHeader.len > kMaxChunkSize) {
    return Err(
        ErrCode::InvalidArgument,
        "tcp read: VRAM read of " + std::to_string(replyHeader.len) +
            " bytes exceeds the staging slab payload (" +
            std::to_string(kMaxChunkSize) + ")");
  }
  auto pool = stagingPool();
  if (!pool) {
    return std::move(pool).error();
  }
  // Non-blocking, and allowed the reserved slab: this is the thread the reserve
  // is held for, and it must not wait on anything.
  auto slab = pool.value()->tryAcquire(/*allowReserved=*/true);
  if (!slab) {
    return deferReadReply(replyHeader, std::move(lease));
  }
  return startReadReply(replyHeader, std::move(lease), std::move(slab));
}

Status TcpTransport::startReadReply(
    const TcpMsgHeader& replyHeader,
    TcpSegmentRegistry::Lease lease,
    TcpPinnedSlab slab) {
  const int deviceId = lease->deviceId;
  const void* src =
      static_cast<const uint8_t*>(lease->ptr) + replyHeader.offset;
  TcpFrame frame(std::move(slab), sizeof(TcpMsgHeader) + replyHeader.len);
  std::memcpy(frame.mutableData(), &replyHeader, sizeof(replyHeader));
  cudaEvent_t event{};
  // Once the copy is enqueued, every way out of this function has to wait for
  // it. Returning an error unwinds `frame` and `lease`, and the copy is
  // asynchronous: the GPU would be left writing into a buffer the allocator has
  // taken back, and reading from a segment a waiting erase() is now free to
  // deregister. drainPendingReadReplies() waits for exactly this reason; the
  // error paths here need the same barrier.
  bool copyIssued = false;
  try {
    CudaDeviceGuard guard(*cudaApi_, deviceId);
    // Into pinned memory, so this returns once the copy is enqueued. The same
    // call into a pageable destination -- a plain vector -- is specified to
    // complete synchronously, which parks whichever thread issued it for the
    // length of the transfer. On this path that thread is the reader.
    if (auto st = cudaApi_->memcpyAsync(
            frame.mutableData() + sizeof(TcpMsgHeader),
            src,
            replyHeader.len,
            cudaMemcpyDeviceToHost,
            /*stream=*/nullptr);
        !st) {
      // A launch that reported an error enqueued nothing, so there is no copy
      // to wait for here.
      return st;
    }
    copyIssued = true;
    // The guard already has deviceId current, so these wait on the right device
    // without nesting another guard.
    if (auto st = cudaApi_->eventCreate(&event); !st) {
      (void)cudaApi_->streamSynchronize(/*stream=*/nullptr);
      return st;
    }
    if (auto st = cudaApi_->eventRecord(event, /*stream=*/nullptr); !st) {
      (void)cudaApi_->eventDestroy(event);
      (void)cudaApi_->streamSynchronize(/*stream=*/nullptr);
      return st;
    }
  } catch (const std::exception& e) {
    if (copyIssued) {
      waitForStagedCopy(deviceId);
    }
    return Err(
        ErrCode::InvalidArgument,
        "tcp read: VRAM staging needs a selectable deviceId, got " +
            std::to_string(deviceId) + ": " + e.what());
  }
  {
    std::lock_guard<std::mutex> lk(stagingMu_);
    pendingReplies_.push_back(
        PendingReadReply{
            std::move(frame),
            std::move(lease),
            event,
            replyHeader.reqId,
            deviceId});
  }
  schedulePendingReplyPoll();
  return Ok();
}

Status TcpTransport::deferReadReply(
    const TcpMsgHeader& replyHeader,
    TcpSegmentRegistry::Lease lease) {
  // The fourth admission point, and it has to agree with the other three about
  // connBroken_. failAllPending() clears this queue precisely so a lease is not
  // held "for as long as the transport object lives"; a reader that reaches
  // here after that sweep would put one straight back, and nothing drains it
  // again until drainPendingReadReplies() at teardown -- which is exactly the
  // outcome the sweep exists to prevent. Refusing instead releases the lease as
  // this returns, and the caller's Error frame to the peer is harmlessly
  // refused by the same connBroken_ check on the enqueue path.
  if (connBroken_.load(std::memory_order_acquire)) {
    return Err(
        ErrCode::NotConnected,
        "tcp read: connection broken while deferring a VRAM read");
  }
  {
    std::lock_guard<std::mutex> lk(stagingMu_);
    if (deferredReplies_.size() >= kMaxInflightRequests) {
      return Err(
          ErrCode::ResourceExhausted,
          "tcp read: too many deferred VRAM reads (" +
              std::to_string(deferredReplies_.size()) + ")");
    }
    deferredReplies_.push_back(
        DeferredReadReply{
            std::move(lease),
            replyHeader.reqId,
            replyHeader.segId,
            replyHeader.offset,
            replyHeader.len});
  }
  // Kicked on enqueue, not only where a slab is released.
  //
  // The caller reaches here because tryAcquire() just failed, and that failure
  // and this enqueue are not one atomic step. A release landing in between --
  // the sender retiring a frame, or the error path dropping one -- runs its own
  // scheduleDeferredReadReplies() against a queue that is still empty and
  // dispatches nothing, while its slab is now free. Without this kick the entry
  // would wait for the *next* release, and on a connection that has gone idle
  // there is no next release: the read never starts and its lease keeps erase()
  // blocked. Redundant kicks are harmless -- startDeferredReadReplies()
  // re-tests both the pool and the queue under the lock.
  scheduleDeferredReadReplies();
  return Ok();
}

void TcpTransport::scheduleDeferredReadReplies() {
  {
    std::lock_guard<std::mutex> lk(stagingMu_);
    if (deferredReplies_.empty()) {
      return;
    }
  }
  evb_->dispatch([this]() noexcept { startDeferredReadReplies(); });
}

void TcpTransport::startDeferredReadReplies() {
  auto pool = stagingPool();
  if (!pool) {
    return;
  }
  while (true) {
    // The slab is taken before the entry, so an entry is never off the queue
    // with nowhere to stage it. A slab that turns out not to be needed is
    // released as this scope ends.
    auto slab = pool.value()->tryAcquire(/*allowReserved=*/true);
    if (!slab) {
      return;
    }
    DeferredReadReply deferred;
    {
      std::lock_guard<std::mutex> lk(stagingMu_);
      if (deferredReplies_.empty()) {
        // Returning here releases the slab without using it, and deliberately
        // does not reschedule. An entry queued between this test and that
        // release is still covered, because deferReadReply() is the only
        // producer and it kicks after every push, while this function runs only
        // from evb_->dispatch() -- so that kick is serialized behind this
        // invocation and sees the slab already back in the pool. A new producer
        // that does not kick would reopen the window.
        return;
      }
      deferred = std::move(deferredReplies_.front());
      deferredReplies_.pop_front();
    }
    TcpMsgHeader replyHeader{};
    replyHeader.op = static_cast<uint8_t>(TcpOp::ReadReply);
    replyHeader.reqId = deferred.reqId;
    replyHeader.segId = deferred.segId;
    replyHeader.offset = deferred.offset;
    replyHeader.len = deferred.len;
    if (auto st = startReadReply(
            replyHeader, std::move(deferred.lease), std::move(slab));
        !st) {
      UNIFLOW_LOG_ERROR(
          "tcp read: deferred VRAM staging failed: {}", st.error().message());
      (void)enqueueFrame(
          makeHeaderFrame(TcpOp::Error, deferred.reqId), /*mayBlock=*/false);
    }
  }
}

Status TcpTransport::startAsyncH2d(
    const TcpInflight& entry,
    std::span<const uint8_t> payload,
    TcpPinnedSlab slab,
    uint64_t reqId) {
  if (!entry.state->tryBeginWrite()) {
    return Err(
        ErrCode::ConnectionFailed,
        "tcp get: operation completed before destination copy started");
  }

  auto stream = static_cast<cudaStream_t>(entry.stream);
  cudaEvent_t event{};
  bool copyIssued = false;
  Status status = Ok();
  try {
    CudaDeviceGuard guard(*cudaApi_, entry.deviceId);
    if (status = cudaApi_->eventCreate(&event); status.hasError()) {
      entry.state->endWrite(status);
      return status;
    }
    if (status = cudaApi_->memcpyAsync(
            entry.dst,
            payload.data(),
            payload.size(),
            cudaMemcpyHostToDevice,
            stream);
        status.hasError()) {
      (void)cudaApi_->eventDestroy(event);
      entry.state->endWrite(status);
      return status;
    }
    copyIssued = true;
    if (status = cudaApi_->eventRecord(event, stream); status.hasError()) {
      PendingH2d uncertain{
          entry.state,
          std::move(slab),
          event,
          entry.deviceId,
          entry.stream,
          reqId,
          std::chrono::steady_clock::now()};
      const auto syncStatus =
          waitForH2dCopy(h2dState_, uncertain.deviceId, uncertain.stream);
      if (syncStatus.hasError()) {
        quarantineH2d(h2dState_, std::move(uncertain));
        return syncStatus;
      }
      destroyH2dEvent(h2dState_, uncertain.deviceId, uncertain.event);
      entry.state->endWrite(Ok());
      return Ok();
    }
  } catch (const std::exception& e) {
    if (copyIssued &&
        waitForH2dCopy(h2dState_, entry.deviceId, entry.stream).hasError()) {
      quarantineH2d(
          h2dState_,
          PendingH2d{
              entry.state,
              std::move(slab),
              event,
              entry.deviceId,
              entry.stream,
              reqId,
              std::chrono::steady_clock::now()});
      return Err(
          ErrCode::DriverError,
          "tcp get: destination copy could not be quiesced");
    }
    if (event != nullptr) {
      destroyH2dEvent(h2dState_, entry.deviceId, event);
    }
    status =
        Err(ErrCode::InvalidArgument,
            "tcp get: VRAM destination needs a selectable deviceId, got " +
                std::to_string(entry.deviceId) + ": " + e.what());
    entry.state->endWrite(status);
    return status;
  }

  PendingH2d pending{
      entry.state,
      std::move(slab),
      event,
      entry.deviceId,
      entry.stream,
      reqId,
      std::chrono::steady_clock::now()};
  try {
    std::lock_guard<std::mutex> lk(h2dState_->mu);
    if (h2dState_->stopping) {
      status =
          Err(ErrCode::ConnectionFailed,
              "tcp get: transport stopped while destination copy started");
    } else {
      h2dState_->pending.push_back(std::move(pending));
      return Ok();
    }
  } catch (const std::exception& e) {
    status = Err(
        ErrCode::TransportError,
        "tcp get: could not track destination copy: " + std::string(e.what()));
  }

  if (auto syncStatus = waitForH2dCopy(h2dState_, entry.deviceId, entry.stream);
      syncStatus.hasError()) {
    quarantineH2d(h2dState_, std::move(pending));
    return syncStatus;
  }
  destroyH2dEvent(h2dState_, entry.deviceId, event);
  entry.state->endWrite(status);
  return status;
}

void TcpTransport::schedulePendingH2dPoll() {
  auto state = h2dState_;
  {
    std::lock_guard<std::mutex> lk(state->mu);
    if (state->stopping || state->pollScheduled || state->pending.empty()) {
      return;
    }
    state->pollScheduled = true;
  }
  state->evb->dispatch(
      [state = std::move(state)]() noexcept { pollPendingH2d(state); });
}

void TcpTransport::pollPendingH2d(
    std::shared_ptr<H2dPollState> state) noexcept {
  while (true) {
    PendingH2d retired;
    Status result = Ok();
    bool haveRetired = false;
    bool queryFailed = false;
    bool reschedule = false;
    {
      std::lock_guard<std::mutex> lk(state->mu);
      if (state->stopping) {
        state->pollScheduled = false;
        return;
      }
      for (auto it = state->pending.begin(); it != state->pending.end(); ++it) {
        Result<bool> done =
            Err(ErrCode::DriverError,
                "tcp get: could not select device for event query");
        try {
          CudaDeviceGuard guard(*state->cudaApi, it->deviceId);
          done =
              state->cudaApi->eventQuery(static_cast<cudaEvent_t>(it->event));
        } catch (const std::exception& e) {
          done = Err(ErrCode::DriverError, e.what());
        }
        if (done.hasValue() && !done.value()) {
          continue;
        }
        if (done.hasError()) {
          result = std::move(done).error();
          queryFailed = true;
        }
        retired = std::move(*it);
        state->retiringState = retired.state;
        state->pending.erase(it);
        ++state->activeRetirements;
        haveRetired = true;
        break;
      }
      if (!haveRetired) {
        if (state->pending.empty()) {
          state->pollScheduled = false;
          return;
        }
        reschedule = true;
      }
    }

    if (reschedule) {
      std::this_thread::yield();
      state->evb->dispatch(
          [state = std::move(state)]() noexcept { pollPendingH2d(state); });
      return;
    }

    if (queryFailed) {
      if (waitForH2dCopy(state, retired.deviceId, retired.stream).hasError()) {
        quarantineH2d(state, std::move(retired));
        std::lock_guard<std::mutex> lk(state->mu);
        state->retiringState.reset();
        --state->activeRetirements;
        state->drained.notify_all();
        continue;
      }
      result = Ok();
    }
    destroyH2dEvent(state, retired.deviceId, retired.event);
    state->copyNs.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - retired.launchedAt)
            .count(),
        std::memory_order_relaxed);
    state->copyCount.fetch_add(1, std::memory_order_relaxed);
    retired.state->endWrite(std::move(result));
    retired.slab.reset();

    {
      std::lock_guard<std::mutex> lk(state->mu);
      state->retiringState.reset();
      --state->activeRetirements;
      if (state->activeRetirements == 0) {
        state->drained.notify_all();
      }
    }
  }
}

Status TcpTransport::waitForH2dCopy(
    const std::shared_ptr<H2dPollState>& state,
    int deviceId,
    void* stream) noexcept {
  try {
    CudaDeviceGuard guard(*state->cudaApi, deviceId);
    return state->cudaApi->streamSynchronize(static_cast<cudaStream_t>(stream));
  } catch (const std::exception& e) {
    return Err(ErrCode::DriverError, e.what());
  }
}

void TcpTransport::destroyH2dEvent(
    const std::shared_ptr<H2dPollState>& state,
    int deviceId,
    void* event) noexcept {
  try {
    CudaDeviceGuard guard(*state->cudaApi, deviceId);
    (void)state->cudaApi->eventDestroy(static_cast<cudaEvent_t>(event));
  } catch (const std::exception&) {
  }
}

void TcpTransport::quarantineH2d(
    const std::shared_ptr<H2dPollState>& state,
    PendingH2d copy) noexcept {
  // Neither the event nor stream established quiescence. Deliberately retain
  // the write reservation and slab: resolving or recycling either could let
  // the caller or pool free memory still touched by DMA.
  copy.state->fail(
      Err(ErrCode::DriverError,
          "tcp get: destination copy could not be safely quiesced"));
  std::lock_guard<std::mutex> lk(state->mu);
  for (auto& slot : state->quarantined) {
    if (!slot.has_value()) {
      slot.emplace(std::move(copy));
      state->quarantineKeepalive = state;
      return;
    }
  }
  std::terminate();
}

void TcpTransport::drainPendingH2d() {
  auto state = h2dState_;
  std::deque<PendingH2d> pending;
  {
    std::unique_lock<std::mutex> lk(state->mu);
    pending.swap(state->pending);
    state->drained.wait(lk, [&]() { return state->activeRetirements == 0; });
  }
  for (auto& copy : pending) {
    if (waitForH2dCopy(state, copy.deviceId, copy.stream).hasError()) {
      quarantineH2d(state, std::move(copy));
      continue;
    }
    destroyH2dEvent(state, copy.deviceId, copy.event);
    copy.state->endWrite(
        Err(ErrCode::ConnectionFailed,
            "tcp transport shut down during destination copy"));
  }
}

void TcpTransport::schedulePendingReplyPoll() {
  {
    std::lock_guard<std::mutex> lk(stagingMu_);
    if (replyPollScheduled_ || pendingReplies_.empty()) {
      return;
    }
    replyPollScheduled_ = true;
  }
  evb_->dispatch([this]() noexcept { pollPendingReadReplies(); });
}

void TcpTransport::pollPendingReadReplies() {
  while (true) {
    TcpFrame ready;
    PendingReadReply failed;
    int failedDeviceId = -1;
    uint64_t failedReqId = 0;
    bool haveReady = false;
    bool haveFailure = false;
    bool stillRunning = false;
    {
      std::lock_guard<std::mutex> lk(stagingMu_);
      if (pendingReplies_.empty()) {
        replyPollScheduled_ = false;
        return;
      }
      auto& front = pendingReplies_.front();
      auto done = cudaApi_->eventQuery(static_cast<cudaEvent_t>(front.event));
      if (done.hasValue() && !done.value()) {
        stillRunning = true;
      } else {
        (void)cudaApi_->eventDestroy(static_cast<cudaEvent_t>(front.event));
        if (done.hasError()) {
          failedReqId = front.reqId;
          failedDeviceId = front.deviceId;
          haveFailure = true;
          // A query that failed says nothing about the copy -- it may still be
          // running. The record is carried out of the deque rather than dropped
          // here so the wait for it happens outside this lock: the reader takes
          // stagingMu_ to enqueue, and must not be held behind a device wait.
          failed = std::move(front);
        } else {
          // The event completed, so the copy is done and the frame is safe to
          // hand on with no further wait.
          ready = std::move(front.frame);
          haveReady = true;
        }
        // Releases the lease, which is what lets a waiting erase() proceed.
        pendingReplies_.pop_front();
      }
    }
    if (stillRunning) {
      // Querying is the only way to learn the copy finished -- there is no
      // completion callback, and EventBase has no timer to defer against -- so
      // this cannot back off by sleeping. Yielding first keeps the poll from
      // monopolising a core the reader and sender threads need, and
      // re-dispatching rather than looping inline lets anything else sharing
      // this EventBase run in between. Dispatched outside the lock: the reader
      // takes stagingMu_ to enqueue, and it should never wait on the
      // EventBase's queue to do it.
      std::this_thread::yield();
      evb_->dispatch([this]() noexcept { pollPendingReadReplies(); });
      return;
    }
    // Queued outside the lock: enqueueFrame takes a lane mutex, and holding two
    // of the transport's mutexes at once is how lock cycles start.
    if (haveReady) {
      (void)enqueueFrame(std::move(ready), /*mayBlock=*/false);
    } else if (haveFailure) {
      // Wait before `failed` goes out of scope, for the same reason
      // drainPendingReadReplies() waits: a copy that may still be running would
      // otherwise be left writing into a slab the pool is about to hand to the
      // next staging copy.
      waitForStagedCopy(failedDeviceId);
      // Dropping it here releases the slab, so a deferred read may now be
      // startable -- which is why this happens before the scheduling call
      // below.
      failed = PendingReadReply{};
      (void)enqueueFrame(
          makeHeaderFrame(TcpOp::Error, failedReqId), /*mayBlock=*/false);
      scheduleDeferredReadReplies();
    }
  }
}

void TcpTransport::waitForStagedCopy(int deviceId) noexcept {
  if (cudaApi_ == nullptr) {
    return;
  }
  try {
    // Per-device: a bare streamSynchronize would only cover whichever device
    // happened to be current, leaving a copy on any other device still running.
    CudaDeviceGuard guard(*cudaApi_, deviceId);
    (void)cudaApi_->streamSynchronize(/*stream=*/nullptr);
  } catch (const std::exception&) {
    // The device is already unusable, so there is nothing left to wait for.
  }
}

void TcpTransport::drainPendingReadReplies() {
  std::deque<PendingReadReply> pending;
  std::deque<DeferredReadReply> deferred;
  {
    std::lock_guard<std::mutex> lk(stagingMu_);
    pending.swap(pendingReplies_);
    deferred.swap(deferredReplies_);
  }
  if (pending.empty()) {
    // Deferred entries have no copy running, so dropping `deferred` here is all
    // that is needed: their leases go with it and a waiting erase() proceeds.
    return;
  }
  // The device may still be writing into these frames. Freeing them now would
  // hand the GPU a buffer the allocator has taken back, so wait for the copies
  // first even though the replies themselves are being abandoned.
  if (cudaApi_ != nullptr) {
    for (auto& reply : pending) {
      waitForStagedCopy(reply.deviceId);
      (void)cudaApi_->eventDestroy(static_cast<cudaEvent_t>(reply.event));
    }
  }
}

Status TcpTransport::admitInflightBulk(std::span<PlannedChunk> chunks) {
  std::lock_guard<std::mutex> lk(inflightMu_);
  if (connBroken_.load(std::memory_order_acquire)) {
    return Err(
        ErrCode::NotConnected, "tcp: transport closed while admitting request");
  }
  // Written as a subtraction against the remaining headroom so a large request
  // cannot overflow its way past the cap.
  if (inflight_.size() >= kMaxInflightRequests ||
      chunks.size() > kMaxInflightRequests - inflight_.size()) {
    return Err(
        ErrCode::ResourceExhausted,
        "tcp: too many outstanding requests (" +
            std::to_string(kMaxInflightRequests) + ")");
  }
  for (auto& chunk : chunks) {
    inflight_[chunk.reqId] = std::move(chunk.entry);
  }
  return Ok();
}

void TcpTransport::abandonInflight(
    std::span<const PlannedChunk> chunks,
    size_t fromIdx) {
  if (fromIdx >= chunks.size()) {
    return;
  }
  std::lock_guard<std::mutex> lk(inflightMu_);
  for (const auto& chunk : chunks.subspan(fromIdx)) {
    inflight_.erase(chunk.reqId);
  }
}

Status TcpTransport::validateDeviceForStaging(int deviceId) {
  try {
    CudaDeviceGuard guard(*cudaApi_, deviceId);
    (void)guard;
  } catch (const std::exception& e) {
    return Err(
        ErrCode::InvalidArgument,
        "tcp: VRAM transfer needs a selectable deviceId, got " +
            std::to_string(deviceId) + ": " + e.what());
  }
  return Ok();
}

Status TcpTransport::admitInflight(uint64_t reqId, TcpInflight entry) {
  std::lock_guard<std::mutex> lk(inflightMu_);
  if (connBroken_.load(std::memory_order_acquire)) {
    return Err(
        ErrCode::NotConnected, "tcp: transport closed while admitting request");
  }
  if (inflight_.size() >= kMaxInflightRequests) {
    return Err(
        ErrCode::ResourceExhausted,
        "tcp: too many outstanding requests (" +
            std::to_string(kMaxInflightRequests) + ")");
  }
  inflight_[reqId] = std::move(entry);
  return Ok();
}

bool TcpTransport::enqueueFrame(TcpFrame frame, bool mayBlock) {
  // No lanes means connect() never ran (or teardown already cleared them).
  // Indexing here would be undefined; refusing matches the documented contract
  // that a frame may simply not be queued.
  if (lanes_.empty()) {
    return false;
  }
  const size_t bytes = frame.size();
  // Every frame this queues -- Write, ReadRequest, ReadReply, Ack, Error --
  // carries its own reqId/segId/offset, so the peer places it without needing
  // arrival order and any lane is equivalent. Only Send is order-sensitive, and
  // it does not come through here (see enqueueSendFrame).
  auto& lane = *lanes_[pickLane()];
  const size_t cap = laneCapBytes();
  {
    std::unique_lock<std::mutex> lk(lane.mu);
    if (mayBlock) {
      // Caller threads absorb backpressure by waiting, which is real
      // backpressure: an application issuing put/get faster than the link
      // drains slows down instead of growing the queue. An empty queue always
      // admits, however large the frame, or a payload bigger than the cap could
      // never drain and would wedge here forever.
      lane.cv.wait(lk, [this, &lane, bytes, cap]() {
        return lane.outClosed || connBroken_.load(std::memory_order_acquire) ||
            lane.queue.empty() || lane.bytes + bytes <= cap;
      });
    }
    // The reader thread deliberately gets no cap. It is producing replies the
    // peer is already blocked on, and a get of N bytes legitimately needs up to
    // N bytes of replies queued -- N is bounded only by the segment size, so no
    // fixed cap can tell a large honest get apart from abuse; they are the same
    // traffic. Refusing here would fail every unrelated in-flight transfer on
    // the connection. It cannot wait either: that stops it draining the socket
    // and reintroduces the mutual-READ deadlock the reader/sender split exists
    // to avoid. What bounds this queue is the drain rate, not a byte cap.
    // Refused on connBroken_ as well as outClosed, so this admission point
    // gives the same answer as admitInflight() and recvImpl(). failAllPending()
    // sets connBroken_ and clears this queue but never sets outClosed -- only a
    // dead sender does that -- and handleFrame's exception containment sweeps
    // without closing the connection, leaving the sender alive. Checking
    // outClosed alone therefore lets a frame admitted before the sweep land in
    // the cleared queue and go out on the wire for an op whose caller has
    // already been told it failed.
    if (lane.outClosed || connBroken_.load(std::memory_order_acquire)) {
      return false;
    }
    lane.queue.push_back(TcpOutItem{std::move(frame), nullptr});
    lane.bytes += bytes;
  }
  lane.cv.notify_all();
  return true;
}

bool TcpTransport::enqueueFrames(std::vector<TcpFrame> frames, bool mayBlock) {
  if (frames.empty()) {
    return true;
  }
  // No lanes means connect() never ran (or teardown already cleared them).
  // Indexing here would be undefined; refusing matches the documented contract
  // that a frame may simply not be queued.
  if (lanes_.empty()) {
    return false;
  }
  size_t bytes = 0;
  for (const auto& frame : frames) {
    bytes += frame.size();
  }
  // One lane for the whole group, so a single mutex makes the insert atomic and
  // no sender can transmit a partial group. Judged against the group total, as
  // before.
  auto& lane = *lanes_[pickLane()];
  const size_t cap = laneCapBytes();
  {
    std::unique_lock<std::mutex> lk(lane.mu);
    if (mayBlock) {
      lane.cv.wait(lk, [this, &lane, bytes, cap]() {
        return lane.outClosed || connBroken_.load(std::memory_order_acquire) ||
            lane.queue.empty() || lane.bytes + bytes <= cap;
      });
    }
    if (lane.outClosed || connBroken_.load(std::memory_order_acquire)) {
      return false;
    }
    for (auto& frame : frames) {
      const size_t frameBytes = frame.size();
      lane.queue.push_back(TcpOutItem{std::move(frame), nullptr});
      lane.bytes += frameBytes;
    }
  }
  lane.cv.notify_all();
  return true;
}

void TcpTransport::enqueueSendFrame(
    TcpFrame frame,
    std::shared_ptr<TcpOpState> onSent) {
  if (lanes_.empty()) {
    // Same reasoning as enqueueFrame's guard, but this path owns a promise, so
    // it must be failed rather than dropped or the caller waits forever.
    if (onSent) {
      onSent->fail(
          Err(ErrCode::NotConnected, "tcp send: transport not connected"));
    }
    return;
  }
  bool closed = false;
  std::shared_ptr<TcpOpState> toFail;
  const size_t bytes = frame.size();
  // Pinned to lane 0, never striped. send()/recv() are a two-sided rendezvous
  // matched in FIFO order through pendingRecvs_/unmatchedSends_, so the Nth
  // SEND must be the Nth the peer's reader sees. Spreading these across lanes
  // would let them arrive out of order and pair a SEND with the wrong recv --
  // silent data corruption rather than an error.
  auto& lane = *lanes_[0];
  const size_t cap = laneCapBytes();
  {
    std::unique_lock<std::mutex> lk(lane.mu);
    // send() runs on a caller thread, so it can wait for room.
    lane.cv.wait(lk, [this, &lane, bytes, cap]() {
      return lane.outClosed || connBroken_.load(std::memory_order_acquire) ||
          lane.queue.empty() || lane.bytes + bytes <= cap;
    });
    if (lane.outClosed || connBroken_.load(std::memory_order_acquire)) {
      closed = true;
      // Taken over here so each path has exactly one owner: the queue takes it
      // when the frame is enqueued, this does when it cannot be.
      toFail = std::move(onSent);
    } else {
      lane.queue.push_back(TcpOutItem{std::move(frame), std::move(onSent)});
      lane.bytes += bytes;
    }
  }
  // Settled outside the lane mutex so TcpOpState::mu stays a leaf lock: a woken
  // by this promise may re-enter the transport from its error path, and it must
  // not find a container mutex still held by the thread that woke it.
  if (closed) {
    if (toFail) {
      toFail->fail(Err(ErrCode::NotConnected, "tcp send: transport closing"));
    }
    return;
  }
  lane.cv.notify_all();
}

void TcpTransport::senderLoop(size_t laneIdx) noexcept {
  // The only writer on this lane's socket, which is what keeps TcpConn's
  // single-writer requirement satisfied while N lanes run at once. Captured
  // once, because connect() installs every lane before this thread starts and
  // never replaces them.
  auto& lane = *lanes_[laneIdx];
  auto* conn = lane.conn.get();
  if (conn == nullptr) {
    return;
  }
  for (;;) {
    TcpOutItem item;
    {
      std::unique_lock<std::mutex> lk(lane.mu);
      lane.cv.wait(
          lk, [&lane]() { return lane.outClosed || !lane.queue.empty(); });
      if (lane.outClosed) {
        return;
      }
      item = std::move(lane.queue.front());
      lane.queue.pop_front();
      lane.bytes -= std::min(lane.bytes, item.frame.size());
    }
    lane.cv.notify_all(); // room freed; wake any producer waiting for space

    // `item` -- and so the frame's storage, which for a staged frame is a
    // pinned slab still on loan from the pool -- stays alive for the whole
    // send: Conn::send only borrows the span. The slab goes back when `item` is
    // destroyed at the end of this iteration, never at pop time.
    auto result = conn->send(item.frame.bytes()).get();
    if (!result) {
      UNIFLOW_LOG_ERROR(
          "tcp sender: send failed: {}", result.error().message());
      // Close EVERY lane's queue before unwinding, not just this one. Nothing
      // drains a queue once its sender returns, and enqueueFrame() gates only
      // on outClosed, so the still-running readers would keep appending
      // Ack/Error/ReadReply frames -- the last up to kMaxChunkSize each -- to a
      // queue with no consumer. A failed lane also takes the whole transport
      // down, so leaving the other lanes admitting work would queue frames for
      // a connection that is already gone.
      closeAllLaneQueues();
      if (item.onSent) {
        item.onSent->fail(Err(ErrCode::ConnectionFailed, "tcp: send failed"));
      }
      failAllPending("tcp: send failed");
      return;
    }
    if (item.onSent) {
      item.onSent->completeOne();
    }
    // Releases the frame's storage, and with it any staging slab, before the
    // next wait. A deferred VRAM read may have been waiting on exactly this
    // slab; dispatching the restart rather than running it here keeps device
    // work off the thread whose only job is to keep the socket draining.
    item = TcpOutItem{};
    scheduleDeferredReadReplies();
  }
}

// Logs the get-path phase split and zeroes it, so a caller can bracket one
// measurement. Reports the reader's own view: time blocked waiting for a frame
// to start (first-byte latency -- network plus whatever the peer did before
// replying), time draining a frame once started, and time in the copy to the
// caller's destination. Anything unaccounted for is reader-thread work between
// those points.
void TcpTransport::logAndResetPhaseStats(std::string_view label) {
  if (lanes_.empty()) {
    return;
  }
  // Summed over every lane: each lane has its own reader and its own stats, so
  // reading lane 0 alone would report 1/N of the traffic and hide any imbalance
  // between lanes. Per-lane frame counts are logged too, since an uneven split
  // is itself a finding.
  uint64_t frames = 0, hdrNs = 0, drainNs = 0, bytes = 0;
  std::string perLaneFrames;
  for (size_t i = 0; i < lanes_.size(); ++i) {
    if (lanes_[i] == nullptr || lanes_[i]->conn == nullptr) {
      continue;
    }
    auto& lrs = lanes_[i]->conn->recvPhaseStats();
    const uint64_t lf = lrs.frames.load(std::memory_order_relaxed);
    frames += lf;
    hdrNs += lrs.headerWaitNs.load(std::memory_order_relaxed);
    drainNs += lrs.payloadDrainNs.load(std::memory_order_relaxed);
    bytes += lrs.payloadBytes.load(std::memory_order_relaxed);
    perLaneFrames += (i == 0 ? "" : ",") + std::to_string(lf);
  }
  const uint64_t copyNs = dstCopyNs_.load(std::memory_order_relaxed) +
      h2dState_->copyNs.load(std::memory_order_relaxed);
  const uint64_t copies = dstCopyCount_.load(std::memory_order_relaxed) +
      h2dState_->copyCount.load(std::memory_order_relaxed);
  const uint64_t slabAttempts =
      receiveSlabAttempts_.load(std::memory_order_relaxed);
  const uint64_t slabMisses =
      receiveSlabMisses_.load(std::memory_order_relaxed);
  const uint64_t vectorReceives =
      vectorReceiveCount_.load(std::memory_order_relaxed);

  if (frames == 0) {
    UNIFLOW_LOG_INFO("tcp phases [{}]: no frames", label);
  } else {
    const double totalNs =
        static_cast<double>(hdrNs) + static_cast<double>(drainNs);
    const auto pct = [totalNs](uint64_t v) {
      return totalNs > 0.0 ? 100.0 * static_cast<double>(v) / totalNs : 0.0;
    };
    // Drain throughput is bytes/drainNs: what the socket managed while a frame
    // was actually in flight, with the inter-frame stall excluded. Comparing it
    // to end-to-end bandwidth says whether the gap is on the wire or between
    // frames.
    const double drainGBps = drainNs > 0
        ? static_cast<double>(bytes) / static_cast<double>(drainNs)
        : 0.0;
    UNIFLOW_LOG_INFO(
        "tcp phases [{}]: frames={} bytes={} | first-byte {:.1f}us/frame "
        "({:.1f}%) | drain {:.1f}us/frame ({:.1f}%, {:.2f} GB/s) | dstcopy "
        "{:.1f}us x{} ({:.1f}% of wire) | receive_slabs attempts={} misses={} "
        "vector_recvs={} | lanes={} frames_per_lane=[{}]",
        label,
        frames,
        bytes,
        static_cast<double>(hdrNs) / frames / 1000.0,
        pct(hdrNs),
        static_cast<double>(drainNs) / frames / 1000.0,
        pct(drainNs),
        drainGBps,
        copies > 0 ? static_cast<double>(copyNs) / copies / 1000.0 : 0.0,
        copies,
        pct(copyNs),
        slabAttempts,
        slabMisses,
        vectorReceives,
        lanes_.size(),
        perLaneFrames);
  }
  for (auto& lane : lanes_) {
    if (lane != nullptr && lane->conn != nullptr) {
      lane->conn->recvPhaseStats().reset();
    }
  }
  dstCopyNs_.store(0, std::memory_order_relaxed);
  dstCopyCount_.store(0, std::memory_order_relaxed);
  h2dState_->copyNs.store(0, std::memory_order_relaxed);
  h2dState_->copyCount.store(0, std::memory_order_relaxed);
  receiveSlabAttempts_.store(0, std::memory_order_relaxed);
  receiveSlabMisses_.store(0, std::memory_order_relaxed);
  vectorReceiveCount_.store(0, std::memory_order_relaxed);
}

void TcpTransport::readerLoop(size_t laneIdx) noexcept {
  // One socket per reader: TcpConn is not safe for concurrent operations, and
  // the single-reader-per-socket invariant is what keeps it so.
  auto* conn = lanes_[laneIdx]->conn.get();
  if (conn == nullptr) {
    return;
  }
  // Reused for every frame that cannot use a receive slab: async H2D disabled,
  // no VRAM get has created the pool yet, allocation failed, or both slabs are
  // busy. The reader never waits for a slab, because draining the socket is
  // what prevents mutual-READ deadlock.
  std::vector<uint8_t> msg;
  while (running_.load(std::memory_order_acquire)) {
    TcpPinnedSlab receiveSlab;
    if (config_.asyncGetH2d) {
      if (auto pool = receivePoolIfCreated()) {
        receiveSlabAttempts_.fetch_add(1, std::memory_order_relaxed);
        receiveSlab = pool->tryAcquire(/*allowReserved=*/true);
        if (!receiveSlab) {
          receiveSlabMisses_.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
    if (receiveSlab) {
      auto result = conn->recv(
                            std::span<uint8_t>{
                                receiveSlab.data(), receiveSlab.capacity()})
                        .get();
      if (!result) {
        break;
      }
      const auto* receiveData = receiveSlab.data();
      const auto receivedSize = result.value();
      handleFrame(
          std::span<const uint8_t>{receiveData, receivedSize},
          std::move(receiveSlab));
      continue;
    }

    vectorReceiveCount_.fetch_add(1, std::memory_order_relaxed);
    auto result = conn->recv(msg).get();
    if (!result) {
      // Connection closed, errored, or idle-timed-out; stop reading.
      break;
    }
    handleFrame(msg);
  }
  // The reader is the only place that resolves in-flight put/get/recv replies.
  // If it exits on a recv error (peer disconnect) while requests are still
  // outstanding and the sender is idle, nothing else would fulfill their
  // promises and callers would block forever on future.get(). Fail them here.
  // Idempotent with shutdown()'s own failAllPending().
  failAllPending("tcp: reader stopped (connection closed or read error)");
}

void TcpTransport::handleFrame(
    std::span<const uint8_t> frame,
    TcpPinnedSlab receiveSlab) noexcept {
  // An exception leaving a noexcept function is std::terminate, and the throw
  // sites in here are reachable from the wire: a ReadRequest's length sizes an
  // allocation, and the VRAM staging path throws if the segment was registered
  // with an invalid deviceId. Neither may let a peer abort the process.
  //
  // The connection is failed rather than the frame dropped, because an
  // exception can land midway through a staging copy or a completion, leaving
  // protocol state that cannot be reasoned about. Stopping the reader also
  // resolves outstanding ops, so no caller is left blocked in future.get().
  try {
    handleFrameImpl(frame, std::move(receiveSlab));
  } catch (const std::exception& e) {
    UNIFLOW_LOG_ERROR(
        "tcp: frame handling raised '{}'; failing the connection", e.what());
    running_.store(false, std::memory_order_release);
    failAllPending("tcp: frame handling raised an exception");
  } catch (...) {
    UNIFLOW_LOG_ERROR(
        "tcp: frame handling raised a non-standard exception; failing the "
        "connection");
    running_.store(false, std::memory_order_release);
    failAllPending("tcp: frame handling raised an exception");
  }
}

void TcpTransport::handleFrameImpl(
    std::span<const uint8_t> frame,
    TcpPinnedSlab receiveSlab) {
  auto headerResult = deserializeTcpHeader(frame);
  if (!headerResult) {
    UNIFLOW_LOG_ERROR(
        "tcp: dropping malformed frame: {}", headerResult.error().message());
    return;
  }
  const TcpMsgHeader header = headerResult.value();
  const auto op = static_cast<TcpOp>(header.op);
  const std::span<const uint8_t> payload = frame.subspan(sizeof(TcpMsgHeader));

  switch (op) {
    case TcpOp::Write: {
      bool ok = false;
      // Lease, not a plain lookup: it must outlive the copy below, because it
      // is what stops the owner deregistering and freeing the buffer underneath
      // us.
      auto entry = registry_->find(header.segId);
      if (entry && header.len <= entry->len &&
          header.offset <= entry->len - header.len &&
          payload.size() == header.len) {
        Status st = Ok();
        if (header.len > 0) {
          void* dst = static_cast<uint8_t*>(entry->ptr) + header.offset;
          if (entry->memType == MemoryType::VRAM) {
            st = deviceFromHost(
                dst, payload.data(), header.len, entry->deviceId);
          } else {
            std::memcpy(dst, payload.data(), header.len);
          }
        }
        ok = !st.hasError();
      }
      (void)enqueueFrame(
          makeHeaderFrame(ok ? TcpOp::Ack : TcpOp::Error, header.reqId),
          /*mayBlock=*/false);
      break;
    }

    case TcpOp::ReadRequest: {
      // Held across the read below for the same reason as the Write path.
      auto entry = registry_->find(header.segId);
      Status readStatus = Ok();
      // Bound the reply by the wire-frame cap, not just by the segment length.
      // A peer that registered a segment larger than the cap can request a read
      // whose reply exceeds kMaxMessageSize; the controller then refuses the
      // send, and senderLoop treats that as fatal -- killing the sender and
      // failing every unrelated in-flight transfer. Our own get() chunks at
      // kMaxChunkSize so a same-version peer never asks for this, which is
      // exactly the version-skew case the header version byte guards: a peer
      // built with a larger chunk size would otherwise turn a recoverable
      // per-request error into a permanently dead connection.
      if (header.len > kMaxFrameSize - sizeof(TcpMsgHeader)) {
        readStatus =
            Err(ErrCode::InvalidArgument,
                "tcp read: requested length would exceed the wire-frame cap");
      } else if (
          entry && header.len <= entry->len &&
          header.offset <= entry->len - header.len) {
        TcpMsgHeader replyHeader;
        replyHeader.op = static_cast<uint8_t>(TcpOp::ReadReply);
        replyHeader.reqId = header.reqId;
        replyHeader.segId = header.segId;
        replyHeader.offset = header.offset;
        replyHeader.len = header.len;
        if (header.len > 0 && entry->memType == MemoryType::VRAM) {
          // Staged into pinned memory rather than copied here: the reply is
          // queued once the copy signals. Copying on this thread would stop the
          // reader draining the socket for the length of a device operation,
          // and the lease it holds would stall any concurrent deregistration
          // for just as long.
          readStatus = respondToVramRead(replyHeader, std::move(entry));
        } else {
          // DRAM, or a zero-length read: a host memcpy cannot fail and cannot
          // park the reader, so there is nothing to stage.
          std::vector<uint8_t> reply(sizeof(TcpMsgHeader) + header.len);
          std::memcpy(reply.data(), &replyHeader, sizeof(replyHeader));
          if (header.len > 0) {
            std::memcpy(
                reply.data() + sizeof(replyHeader),
                static_cast<const uint8_t*>(entry->ptr) + header.offset,
                header.len);
          }
          (void)enqueueFrame(std::move(reply), /*mayBlock=*/false);
        }
      } else {
        readStatus = Err(ErrCode::InvalidArgument, "tcp read: bad segment");
      }
      if (readStatus.hasError()) {
        (void)enqueueFrame(
            makeHeaderFrame(TcpOp::Error, header.reqId), /*mayBlock=*/false);
      }
      break;
    }

    case TcpOp::Send: {
      if (payload.size() != header.len) {
        UNIFLOW_LOG_ERROR("tcp send: inbound payload size mismatch");
        break;
      }
      std::shared_ptr<TcpOpState> state;
      void* dst = nullptr;
      size_t cap = 0;
      MemoryType memType = MemoryType::DRAM;
      int deviceId = -1;
      void* stream = nullptr;
      bool matched = false;
      bool overflow = false;
      {
        std::lock_guard<std::mutex> lk(recvMu_);
        if (!pendingRecvs_.empty()) {
          auto pr = std::move(pendingRecvs_.front());
          pendingRecvs_.pop_front();
          state = std::move(pr.state);
          dst = pr.dst;
          cap = pr.cap;
          memType = pr.memType;
          deviceId = pr.deviceId;
          stream = pr.stream;
          matched = true;
        } else if (unmatchedBytes_ + payload.size() > kMaxUnmatchedSendBytes) {
          overflow = true;
        } else {
          unmatchedSends_.emplace_back(payload.begin(), payload.end());
          unmatchedBytes_ += payload.size();
        }
      }
      if (overflow) {
        // No backpressure is available on this path, so absorbing more would be
        // unbounded host-memory growth driven by the peer. Refuse the
        // connection instead: closing it makes the reader's next recv() fail,
        // and the reader's own failAllPending() then resolves outstanding ops.
        UNIFLOW_LOG_ERROR(
            "tcp recv: unmatched inbound sends exceed {} bytes; closing the "
            "connection",
            kMaxUnmatchedSendBytes);
        closeLanesOnce();
        break;
      }
      if (matched && state) {
        if (payload.size() > cap) {
          state->fail(
              Err(ErrCode::InvalidArgument,
                  "tcp recv: buffer too small for incoming send"));
        } else {
          Status st = Ok();
          if (!payload.empty() && dst != nullptr) {
            if (memType == MemoryType::VRAM) {
              st = deviceFromHost(
                  dst, payload.data(), payload.size(), deviceId, stream);
            } else {
              std::memcpy(dst, payload.data(), payload.size());
            }
          }
          if (st.hasError()) {
            state->fail(std::move(st));
          } else {
            state->completeOne();
          }
        }
      }
      break;
    }

    case TcpOp::Ack:
    case TcpOp::ReadReply:
    case TcpOp::Error: {
      TcpInflight entry;
      bool found = false;
      {
        std::lock_guard<std::mutex> lk(inflightMu_);
        auto it = inflight_.find(header.reqId);
        if (it != inflight_.end()) {
          entry = it->second;
          found = true;
        }
      }
      if (!found || !entry.state) {
        break;
      }

      const auto eraseInflight = [&]() {
        std::lock_guard<std::mutex> lk(inflightMu_);
        inflight_.erase(header.reqId);
      };
      // An Ack answers a Write and a ReadReply answers a ReadRequest, so a
      // reply whose kind disagrees with the request it names is version skew or
      // a hostile peer. Rejected here rather than per-branch because the Ack
      // direction is the dangerous one and it used to fall straight through to
      // completeOne(): the get chunk resolved Ok with entry.dst never written,
      // handing the caller back whatever its buffer already held. Every other
      // peer-supplied dimension on this path fails loudly; this one did not
      // fail at all. Error is exempt because it is kind-agnostic by design.
      if (op != TcpOp::Error && (op == TcpOp::ReadReply) != entry.isRead) {
        eraseInflight();
        entry.state->fail(
            Err(ErrCode::TransportError,
                "tcp: reply op does not match the request kind"));
        break;
      }
      if (op == TcpOp::Error) {
        eraseInflight();
        entry.state->fail(
            Err(ErrCode::TransportError, "tcp: peer reported an error"));
      } else if (op == TcpOp::ReadReply) {
        // Kind is settled above, so this is purely a size check and its message
        // says so.
        if (payload.size() != header.len || header.len != entry.len) {
          eraseInflight();
          entry.state->fail(Err(
              ErrCode::TransportError, "tcp get: read reply size mismatch"));
          break;
        }

        const bool asyncH2d = config_.asyncGetH2d && receiveSlab &&
            header.len > 0 && entry.dst != nullptr &&
            entry.memType == MemoryType::VRAM;
        if (asyncH2d) {
          const auto status = startAsyncH2d(
              entry, payload, std::move(receiveSlab), header.reqId);
          // The pending record is visible before this erase. A concurrent
          // failure therefore sees the state in inflight_, the H2D queue, or
          // both, matching RDMA's handoff into transport-owned progress state.
          eraseInflight();
          if (status.hasValue()) {
            schedulePendingH2dPoll();
          }
          break;
        }

        eraseInflight();
        // Vector-backed VRAM replies cannot outlive this call because the
        // reader reuses their storage. Keep that fallback synchronous, with the
        // same reservation that prevents a concurrent failure releasing
        // entry.dst.
        entry.state->writeAndComplete([&]() -> Status {
          if (header.len == 0 || entry.dst == nullptr) {
            return Ok();
          }
          const auto tCopyStart = std::chrono::steady_clock::now();
          Status st = Ok();
          if (entry.memType == MemoryType::VRAM) {
            st = deviceFromHost(
                entry.dst,
                payload.data(),
                header.len,
                entry.deviceId,
                entry.stream);
          } else {
            std::memcpy(entry.dst, payload.data(), header.len);
          }
          dstCopyNs_.fetch_add(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - tCopyStart)
                  .count(),
              std::memory_order_relaxed);
          dstCopyCount_.fetch_add(1, std::memory_order_relaxed);
          return st;
        });
      } else { // Ack
        eraseInflight();
        entry.state->completeOne();
      }
      break;
    }

    case TcpOp::Notification:
      UNIFLOW_LOG_WARN("tcp: unexpected inbound Notification frame");
      break;

    default:
      UNIFLOW_LOG_WARN(
          "tcp: unexpected opcode {}", static_cast<int>(header.op));
      break;
  }
}

void TcpTransport::closeAllLaneQueues() {
  // Marks every lane's queue closed and wakes everyone waiting on it: senders
  // so they exit, and producers blocked for room so they observe the close
  // instead of waiting for space that will never be freed.
  for (auto& lane : lanes_) {
    if (lane == nullptr) {
      continue;
    }
    {
      std::lock_guard<std::mutex> lk(lane->mu);
      lane->outClosed = true;
    }
    lane->cv.notify_all();
  }
}

void TcpTransport::closeLanesOnce() {
  // Conn::close() tests the fd, closes it, then clears it, with no
  // synchronisation. A reader refuses the connection on unmatched-send overflow
  // while an application thread can be in shutdown() doing the same, and
  // neither holds lifecycleMu_ (a reader must not, since shutdown() holds it
  // across the reader joins). Both callers can therefore observe the fd open
  // and both ::close() it, and the second reaps a descriptor that another
  // thread in the process may already have been handed. Winning the exchange
  // for a lane is what earns the right to close it.
  for (auto& lane : lanes_) {
    if (lane != nullptr && lane->conn != nullptr &&
        !lane->closed.exchange(true, std::memory_order_acq_rel)) {
      lane->conn->close();
    }
  }
}

void TcpTransport::failAllPending(const char* message) {
  connBroken_.store(true, std::memory_order_release);
  // Collect first, settle after every mutex is released. Fulfilling a promise
  // under a container mutex hands control to the waiting caller while this
  // thread still holds it; a caller that tears the transport down from its
  // error path would then destroy the mutex and the containers underneath this
  // function. Settling afterwards also keeps TcpOpState::mu a leaf lock.
  std::vector<std::shared_ptr<TcpOpState>> toFail;
  {
    std::lock_guard<std::mutex> lk(inflightMu_);
    for (auto& [reqId, entry] : inflight_) {
      if (entry.state) {
        toFail.push_back(std::move(entry.state));
      }
    }
    inflight_.clear();
  }
  {
    std::lock_guard<std::mutex> lk(h2dState_->mu);
    for (const auto& copy : h2dState_->pending) {
      if (copy.state) {
        toFail.push_back(copy.state);
      }
    }
    if (h2dState_->retiringState) {
      toFail.push_back(h2dState_->retiringState);
    }
  }
  {
    std::lock_guard<std::mutex> lk(recvMu_);
    for (auto& pr : pendingRecvs_) {
      if (pr.state) {
        toFail.push_back(std::move(pr.state));
      }
    }
    pendingRecvs_.clear();
    unmatchedSends_.clear();
    unmatchedBytes_ = 0;
  }
  for (auto& lane : lanes_) {
    if (lane == nullptr) {
      continue;
    }
    std::lock_guard<std::mutex> lk(lane->mu);
    for (auto& item : lane->queue) {
      if (item.onSent) {
        toFail.push_back(std::move(item.onSent));
      }
    }
    lane->queue.clear();
    lane->bytes = 0;
  }
  // Deferred reads will never be answered on a broken connection, and each one
  // holds a lease. Dropped here rather than left for teardown so a
  // deregistration is not blocked for as long as the transport object lives.
  // pendingReplies_ is deliberately untouched: those have copies running, and
  // freeing their frames now would hand the GPU memory the allocator has taken
  // back. drainPendingReadReplies() waits for them.
  std::deque<DeferredReadReply> deferred;
  {
    std::lock_guard<std::mutex> lk(stagingMu_);
    deferred.swap(deferredReplies_);
  }
  deferred.clear();
  // Wake producers blocked for space on every lane so they see the break.
  for (auto& lane : lanes_) {
    if (lane != nullptr) {
      lane->cv.notify_all();
    }
  }
  for (auto& state : toFail) {
    state->fail(Err(ErrCode::ConnectionFailed, message));
  }
}

std::future<Status> TcpTransport::sendImpl(
    const void* data,
    size_t len,
    MemoryType memType,
    int deviceId,
    const RequestOptions& options) {
  if (state_ != TransportState::Connected ||
      connBroken_.load(std::memory_order_acquire)) {
    return make_ready_future<Status>(
        Err(ErrCode::NotConnected, "tcp send: not connected"));
  }
  // Staging VRAM on the null stream is the one outcome that can silently
  // transmit stale device data, so require the caller to say which stream the
  // D2H must be ordered against. Matches RdmaTransport::rdmaSendRecvTransfer.
  if (memType == MemoryType::VRAM && !options.stream.has_value()) {
    return make_ready_future<Status>(
        Err(ErrCode::InvalidArgument,
            "VRAM transfer requires an explicit CUDA stream"));
  }
  // send() is single-frame (no chunking); reject payloads that would exceed the
  // wire-frame cap so callers get a diagnostic instead of a silent drop/hang.
  if (len > kMaxFrameSize - sizeof(TcpMsgHeader)) {
    return make_ready_future<Status>(
        Err(ErrCode::InvalidArgument,
            "tcp send: payload exceeds the 64 MiB wire-frame cap; use put/get "
            "(which chunk) for large transfers"));
  }
  auto state = std::make_shared<TcpOpState>();
  state->remaining = 1;
  auto future = state->promise.get_future();

  std::vector<uint8_t> frame(sizeof(TcpMsgHeader) + len);
  TcpMsgHeader header;
  header.op = static_cast<uint8_t>(TcpOp::Send);
  header.len = static_cast<uint64_t>(len);
  std::memcpy(frame.data(), &header, sizeof(header));
  if (len > 0 && data != nullptr) {
    Status st = Ok();
    if (memType == MemoryType::VRAM) {
      st = hostFromDevice(
          frame.data() + sizeof(header),
          data,
          len,
          deviceId,
          static_cast<void*>(options.stream.value()));
    } else {
      std::memcpy(frame.data() + sizeof(header), data, len);
    }
    if (!st) {
      state->fail(std::move(st));
      return future;
    }
  }
  enqueueSendFrame(std::move(frame), std::move(state));
  return future;
}

std::future<Status> TcpTransport::recvImpl(
    void* dst,
    size_t cap,
    MemoryType memType,
    int deviceId,
    const RequestOptions& options) {
  if (state_ != TransportState::Connected ||
      connBroken_.load(std::memory_order_acquire)) {
    return make_ready_future<Status>(
        Err(ErrCode::NotConnected, "tcp recv: not connected"));
  }
  // Without an explicit stream the H2D would land on the null stream, letting
  // the caller launch kernels against a buffer the payload has not reached yet.
  // Matches RdmaTransport::rdmaSendRecvTransfer.
  if (memType == MemoryType::VRAM && !options.stream.has_value()) {
    return make_ready_future<Status>(
        Err(ErrCode::InvalidArgument,
            "VRAM transfer requires an explicit CUDA stream"));
  }
  void* const stream = options.stream.has_value()
      ? static_cast<void*>(options.stream.value())
      : nullptr;
  auto state = std::make_shared<TcpOpState>();
  state->remaining = 1;
  auto future = state->promise.get_future();

  std::vector<uint8_t> payload;
  bool matched = false;
  bool closed = false;
  {
    std::lock_guard<std::mutex> lk(recvMu_);
    // Same admission race as the inflight_ path: a failAllPending() landing
    // between the entry check above and this push_back would leave a posted
    // recv no sweep will ever see, blocking the caller forever. Checked under
    // recvMu_ so the two orderings are exclusive.
    if (connBroken_.load(std::memory_order_acquire)) {
      closed = true;
    } else if (!unmatchedSends_.empty()) {
      payload = std::move(unmatchedSends_.front());
      unmatchedSends_.pop_front();
      unmatchedBytes_ -= std::min(unmatchedBytes_, payload.size());
      matched = true;
    } else {
      pendingRecvs_.push_back(
          TcpPendingRecv{dst, cap, state, memType, deviceId, stream});
    }
  }
  if (closed) {
    state->fail(
        Err(ErrCode::NotConnected,
            "tcp recv: transport closed while posting receive"));
    return future;
  }
  if (matched) {
    if (payload.size() > cap) {
      state->fail(
          Err(ErrCode::InvalidArgument,
              "tcp recv: buffer too small for buffered send"));
    } else {
      Status st = Ok();
      if (!payload.empty() && dst != nullptr) {
        if (memType == MemoryType::VRAM) {
          st = deviceFromHost(
              dst, payload.data(), payload.size(), deviceId, stream);
        } else {
          std::memcpy(dst, payload.data(), payload.size());
        }
      }
      if (st.hasError()) {
        state->fail(std::move(st));
      } else {
        state->completeOne();
      }
    }
  }
  return future;
}

std::future<Status> TcpTransport::send(
    RegisteredSegment::Span src,
    const RequestOptions& options) {
  return sendImpl(
      src.data(), src.size(), src.memType(), src.deviceId(), options);
}

std::future<Status> TcpTransport::recv(
    RegisteredSegment::Span dst,
    const RequestOptions& options) {
  return recvImpl(
      dst.mutable_data(), dst.size(), dst.memType(), dst.deviceId(), options);
}

std::future<Status> TcpTransport::send(
    Segment::Span src,
    const RequestOptions& options) {
  return sendImpl(
      src.data(), src.size(), src.memType(), src.deviceId(), options);
}

std::future<Status> TcpTransport::recv(
    Segment::Span dst,
    const RequestOptions& options) {
  return recvImpl(
      dst.mutable_data(), dst.size(), dst.memType(), dst.deviceId(), options);
}

void TcpTransport::shutdown() {
  std::lock_guard<std::mutex> lifecycleLock(lifecycleMu_);
  // One-shot, but checked under the mutex rather than before taking it:
  // shutdown() is called twice in the normal flow (MultiTransport::shutdown()
  // then ~TcpTransport), and the second caller must not return while the first
  // is still tearing down -- the destructor would otherwise race it.
  if (shutdown_.exchange(true)) {
    return;
  }
  running_.store(false, std::memory_order_release);

  closeAllLaneQueues();

  // Closing the data connections unblocks the readers' blocking recv and any
  // in-progress send on the sender thread. A no-op for a lane a reader already
  // refused, in which case it is on its way out anyway.
  closeLanesOnce();

  // Closed before the joins: this is the one pool with a *blocking* acquire,
  // and the thread parked in it is not one we own. acquire() waits on `freed_`
  // with no deadline and `closed_` as its only escape, and the put path calls
  // it on the application's own thread, which shutdown() never joins. Without
  // this a put() in flight across shutdown() parks forever, because the senders
  // that would have freed a staging slab are about to be joined away.
  //
  // close() only sets the flag and notifies, so outstanding leases stay valid
  // and doing this early costs nothing. The receive pool needs no equivalent --
  // the reader only ever tryAcquire()s it. Read under poolMu_, which is what
  // stagingPool() publishes slabPool_ under.
  std::shared_ptr<TcpPinnedSlabPool> staging;
  {
    std::lock_guard<std::mutex> lk(poolMu_);
    staging = slabPool_;
  }
  if (staging != nullptr) {
    staging->close();
  }

  for (auto& lane : lanes_) {
    if (lane == nullptr) {
      continue;
    }
    if (lane->reader.joinable()) {
      lane->reader.join();
    }
    if (lane->sender.joinable()) {
      lane->sender.join();
    }
  }

  if (auto pool = std::atomic_load_explicit(
          &receiveSlabPool_, std::memory_order_acquire)) {
    pool->close();
  }

  {
    std::lock_guard<std::mutex> lk(h2dState_->mu);
    h2dState_->stopping = true;
  }
  failAllPending("tcp transport shut down");
  drainPendingH2d();

  // After the reader is joined, so nothing can add to the queue, and before the
  // transport goes away: a staged reply's frame is memory the device may still
  // be writing into.
  drainPendingReadReplies();

  // Compare-exchange, not load-then-store: a concurrent bind() failure setting
  // Error in the gap between a load and a store would be clobbered back to
  // Disconnected, so a transport that failed to bind would report as cleanly
  // closed.
  auto expected = state_.load(std::memory_order_acquire);
  while (
      expected != TransportState::Error &&
      !state_.compare_exchange_weak(expected, TransportState::Disconnected)) {
  }
}

// ---------------------------------------------------------------------------
// TcpTransportFactory
// ---------------------------------------------------------------------------

Status TcpTransportFactory::supported() {
  return Ok();
}

TcpTransportFactory::TcpTransportFactory(
    int deviceId,
    EventBase* evb,
    TcpTransportConfig config,
    std::string host,
    std::shared_ptr<CudaApi> cudaApi)
    : TransportFactory(TransportType::TCP),
      deviceId_(deviceId),
      evb_(evb),
      config_(std::move(config)),
      host_(host.empty() ? std::string("127.0.0.1") : std::move(host)),
      cudaApi_(cudaApi ? std::move(cudaApi) : std::make_shared<CudaApi>()) {}

Result<std::unique_ptr<RegistrationHandle>>
TcpTransportFactory::registerSegment(Segment& segment) {
  if (segment.memType() != MemoryType::DRAM &&
      segment.memType() != MemoryType::VRAM) {
    return Err(
        ErrCode::MemoryRegistrationError,
        "tcp transport supports only DRAM and VRAM segments");
  }

  const auto segId = nextSegId_.fetch_add(1, std::memory_order_relaxed);
  registry_->add(
      segId,
      segment.mutable_data(),
      segment.len(),
      segment.memType(),
      segment.deviceId());
  return std::make_unique<TcpRegistrationHandle>(
      segId,
      static_cast<uint64_t>(segment.len()),
      [weakRegistry = std::weak_ptr<TcpSegmentRegistry>(registry_), segId]() {
        if (auto registry = weakRegistry.lock()) {
          // Blocks until no reader is mid-copy on this segment, so that when
          // the handle finishes destructing the owner can free the buffer.
          registry->erase(segId);
        }
      });
}

Result<std::unique_ptr<RemoteRegistrationHandle>>
TcpTransportFactory::importSegment(
    size_t segmentLength,
    std::span<const uint8_t> payload) {
  auto handle =
      TcpRemoteRegistrationHandle::deserialize(segmentLength, payload);
  if (!handle) {
    return std::move(handle).error();
  }
  return std::move(handle).value();
}

Result<std::unique_ptr<Transport>> TcpTransportFactory::createTransport(
    std::span<const uint8_t> peerTopology) {
  // Validate the peer's capability blob before building anything, the way
  // RdmaTransportFactory::createTransport does. The peer's *address* is not
  // here; it arrives later through bind()/connect().
  auto status = canConnect(peerTopology);
  if (!status) {
    return std::move(status).error();
  }
  return std::make_unique<TcpTransport>(
      deviceId_, evb_, registry_, config_, host_, cudaApi_);
}

std::vector<uint8_t> TcpTransportFactory::getTopology() {
  return TcpTopologyInfo{}.serialize();
}

Status TcpTransportFactory::canConnect(std::span<const uint8_t> peerTopology) {
  auto info = TcpTopologyInfo::deserialize(peerTopology);
  if (!info) {
    return std::move(info).error();
  }
  if (info->version != kTcpWireVersion) {
    return Err(
        ErrCode::TopologyDisconnect,
        "tcp: unsupported peer wire version " + std::to_string(info->version) +
            ", local is " + std::to_string(kTcpWireVersion));
  }
  return Ok();
}

} // namespace uniflow
