// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/MultiTransport.h"
#include "comms/uniflow/drivers/DeviceAdapter.h"
#include "comms/uniflow/drivers/TopologyDiscovery.h"
#include "comms/uniflow/logging/Logger.h"

// RDMA is the GPU transport on AMD as well as NVIDIA. NVLink is NVIDIA-only
// (NVML-backed topology, fabric/FD IPC) and is compiled out on AMD/HIP.
#include "comms/uniflow/transport/rdma/RdmaTransport.h"
#ifdef UNIFLOW_ENABLE_TCP_TRANSPORT
#include "comms/uniflow/transport/tcp/TcpTransport.h"
#endif
#ifndef __HIP_PLATFORM_AMD__
#include "comms/uniflow/transport/nvlink/NVLinkTransport.h"
#else
#include "comms/uniflow/transport/p2p/P2pTransport.h"
#endif

#ifdef UNIFLOW_ENABLE_TCP_TRANSPORT
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>

namespace uniflow {

namespace {

bool isCpu(int deviceId) {
  return deviceId == -1;
}

#ifdef UNIFLOW_ENABLE_TCP_TRANSPORT
// Resolve a routable bind host for the TCP transport. Prefers an interface
// whose name starts with netdevPrefix, else the first global (non-loopback,
// non-link-local) address found, else falls back to loopback. Binding a
// routable address (instead of 127.0.0.1) is what lets the TCP transport's
// connect() succeed across hosts.
std::string resolveTcpBindHost(const std::string& netdevPrefix) {
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) != 0 || ifaddr == nullptr) {
    return "127.0.0.1";
  }
  std::string prefixMatch;
  std::string anyGlobal;
  for (auto* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) {
      continue;
    }
    char buf[INET6_ADDRSTRLEN] = {};
    std::string addr;
    const int family = ifa->ifa_addr->sa_family;
    if (family == AF_INET6) {
      auto* sa = reinterpret_cast<sockaddr_in6*>(ifa->ifa_addr);
      const auto* b = sa->sin6_addr.s6_addr;
      const bool linkLocal = b[0] == 0xfe && (b[1] & 0xc0) == 0x80;
      bool loopback = true;
      for (int i = 0; i < 15; ++i) {
        loopback = loopback && b[i] == 0;
      }
      loopback = loopback && b[15] == 1;
      // Unspecified (::) and IPv4-mapped-IPv6 loopback (::ffff:127.0.0.0/8) are
      // not routable; skip them so anyGlobal never holds a loopback-equivalent
      // over a genuinely global address enumerated later on the same host.
      bool unspecified = true;
      for (int i = 0; i < 16; ++i) {
        unspecified = unspecified && b[i] == 0;
      }
      bool v4Mapped = true;
      for (int i = 0; i < 10; ++i) {
        v4Mapped = v4Mapped && b[i] == 0;
      }
      const bool v4MappedLoopback =
          v4Mapped && b[10] == 0xff && b[11] == 0xff && b[12] == 127;
      if (linkLocal || loopback || unspecified || v4MappedLoopback ||
          inet_ntop(AF_INET6, &sa->sin6_addr, buf, sizeof(buf)) == nullptr) {
        continue;
      }
      addr = buf;
    } else if (family == AF_INET) {
      auto* sa = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr);
      if ((ntohl(sa->sin_addr.s_addr) >> 24) == 127 ||
          inet_ntop(AF_INET, &sa->sin_addr, buf, sizeof(buf)) == nullptr) {
        continue;
      }
      addr = buf;
    } else {
      continue;
    }
    const std::string name = ifa->ifa_name != nullptr ? ifa->ifa_name : "";
    if (prefixMatch.empty() && !netdevPrefix.empty() &&
        name.rfind(netdevPrefix, 0) == 0) {
      prefixMatch = addr;
    }
    if (anyGlobal.empty()) {
      anyGlobal = addr;
    }
  }
  freeifaddrs(ifaddr);
  std::string host = !prefixMatch.empty()
      ? prefixMatch
      : (!anyGlobal.empty() ? anyGlobal : std::string("127.0.0.1"));
  UNIFLOW_LOG_INFO(
      "TCP transport bind host resolved to {} (netdevPrefix={})",
      host,
      netdevPrefix);
  return host;
}

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
// ceiling rather than as a fixed cost. Across 10 runs each, no single-card pair
// exceeded 35.4 GB/s while half the two-card pairs did, reaching 40.3; medians
// differ by only ~6% (34.1 against 36.3) and overlap, so spreading removes a
// cap rather than reliably buying a fixed gain.
//
// When only one card is present its second port is used: a second port on the
// same card is worth far more than no second port at all (~34 against ~21).
//
// Order is deterministic -- cards by PCI address, ports by name -- because lane
// i maps to device i on both peers, so an unstable order would pair a different
// physical port from one run to the next.
//
// Usability is delegated to deviceGlobalIpv6 rather than re-derived: it already
// applies the address-flag rules, and it is what bind() will call for the
// address, so discovery cannot disagree with what actually gets bound.
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

// True if @p host is empty or loopback. Advertising a loopback bind address in
// the connect handshake would break cross-host connections for all transports,
// so TCP is not auto-registered in that case.
bool isLoopbackHost(const std::string& host) {
  return host.empty() || host == "::1" || host.rfind("127.", 0) == 0 ||
      // IPv4-mapped-IPv6 loopback forms (dotted and pure-hex 127.0.0.0/8).
      host.rfind("::ffff:127.", 0) == 0 || host.rfind("::ffff:7f", 0) == 0;
}
#endif

} // namespace

// ============================================================================
// MultiTransport Implementation
// ============================================================================

std::vector<std::string> MultiTransportFactory::selectCpuNics() {
  auto& topo = sharedTopology();
  auto nics = topo.selectCpuNics(options_.nicFilter, options_.netdevPrefix);
  const auto candidateNicCount = nics.size();
  if (options_.cpuNicSelectionPolicy == CpuNicSelectionPolicy::kAll) {
    UNIFLOW_LOG_INFO(
        "CPU RDMA NIC selection policy=all selected_nics={}",
        candidateNicCount);
    return nics;
  }

  auto localNics =
      topo.selectCpuNicsForNumaNodes(options_.nicFilter, options_.maxCpuNics);
  if (localNics.empty()) {
    localNics = std::move(nics);
    if (options_.maxCpuNics > 0 && localNics.size() > options_.maxCpuNics) {
      localNics.resize(options_.maxCpuNics);
    }
  }

  UNIFLOW_LOG_INFO(
      "CPU RDMA NIC selection policy=numa_local_bounded selected_nics={} "
      "candidate_nics={} numa_nodes={} max_cpu_nics_per_numa={}",
      localNics.size(),
      candidateNicCount,
      topo.numaNodeCount(),
      options_.maxCpuNics);
  return localNics;
}

std::vector<std::string> MultiTransportFactory::selectNics() {
  auto& topo = sharedTopology();
  if (isCpu(deviceId_)) {
    return selectCpuNics();
  }
  return topo.selectGpuNics(
      deviceId_, options_.nicFilter, options_.netdevPrefix);
}

Status MultiTransportFactory::supported(TransportType type) {
  switch (type) {
    case TransportType::RDMA:
      return RdmaTransportFactory::supported();
#ifndef __HIP_PLATFORM_AMD__
    case TransportType::NVLink:
      return NVLinkTransportFactory::supported();
#else
    case TransportType::NVLink:
      // On AMD the NVLink tier is served by the P2P (XGMI) transport, whose
      // supported() encodes the all-XGMI arch gate.
      return P2pTransportFactory::supported();
#endif
    case TransportType::TCP:
#ifdef UNIFLOW_ENABLE_TCP_TRANSPORT
      return TcpTransportFactory::supported();
#else
      return Err(
          ErrCode::NotImplemented,
          "tcp transport is not enabled in this platform build");
#endif
    case TransportType::Mock:
      return Ok();
    case TransportType::NumTransportType:
      break;
  }
  return Err(ErrCode::InvalidArgument, "unknown transport type");
}

Status MultiTransport::validateRequests(
    const std::vector<TransferRequest>& requests) {
  if (requests.empty()) {
    return Err(ErrCode::InvalidArgument, "empty request list");
  }

  auto localMemType = requests.front().local.memType();
  auto remoteMemType = requests.front().remote.memType();
  auto localMemDeviceId = requests.front().local.deviceId();

  for (size_t i = 1; i < requests.size(); ++i) {
    const auto& local = requests[i].local;
    const auto& remote = requests[i].remote;
    if (local.memType() != localMemType || remote.memType() != remoteMemType ||
        local.deviceId() != localMemDeviceId) {
      return Err(
          ErrCode::InvalidArgument,
          "all requests must have the same memory type and device id");
    }
  }

  // Handle counts and transport types are intentionally not validated here.
  // A segment's handles describe the transports that were usable when that
  // specific segment was registered or imported; they are capabilities, not a
  // batch schema. For example, a GPU cache segment may expose both NVLink and
  // RDMA, while a peer process that cannot import the NVLink handle still has a
  // valid RDMA handle. selectTransport() enforces the real transfer invariant:
  // one common transport must be available on every local and remote segment in
  // the batch.
  return Ok();
}

Transport* MultiTransport::findTransport(TransportType type) const {
  for (auto& t : transports_) {
    if (t->transportType() == type) {
      return t.get();
    }
  }
  return nullptr;
}

MultiTransportFactory::MultiTransportFactory(
    int deviceId,
    MultiTransportFactoryOptions options)
    : deviceId_(deviceId),
      options_(std::move(options)),
      eventBaseThread_(std::make_shared<ScopedEventBaseThread>()) {
  auto& topo = sharedTopology();
  CHECK_THROW_EXCEPTION(
      deviceId_ >= -1 && deviceId_ < static_cast<int>(topo.gpuCount()),
      std::runtime_error);

#ifdef UNIFLOW_ENABLE_TCP_TRANSPORT
  // An explicit TCP request (via preferred/intra/interNodeTransport) must carry
  // a bind host AND enableTcp; otherwise TCP would not register (silently
  // falling back to RDMA/NVLink) or would auto-resolve a non-front-end NIC.
  // Fail fast at construction rather than mis-routing at transfer time.
  const auto requestsTcp = [](const std::optional<TransportType>& t) {
    return t.has_value() && *t == TransportType::TCP;
  };
  CHECK_THROW_EXCEPTION(
      !((requestsTcp(options_.preferredTransport) ||
         requestsTcp(options_.intraNodeTransport) ||
         requestsTcp(options_.interNodeTransport)) &&
        (options_.tcpBindHost.empty() || !options_.enableTcp)),
      std::invalid_argument);
#endif

  // Register the intra-node interconnect tier whenever the hardware is present
  // (NVLink on NVIDIA, P2P/XGMI on AMD). This tier and RDMA are both registered
  // when available; selectTransport chooses per transfer (intraNodeTransport
  // can flip the intra-node default -- see selectTransport).
#ifndef __HIP_PLATFORM_AMD__
  if (deviceId_ >= 0 && isNvlinkAvailable()) {
    auto nvlink = std::make_shared<NVLinkTransportFactory>(
        deviceId, eventBaseThread_->getEventBase());
    factories_.emplace_back(std::move(nvlink));
  }
#else
  // AMD: the NVLink tier is served by the P2P (XGMI) transport. Its supported()
  // owns the all-XGMI arch gate (selectTransport is presence-driven; see §5.6).
  if (deviceId_ >= 0) {
    auto p2pSupported = P2pTransportFactory::supported();
    if (!p2pSupported.hasError()) {
      auto p2p = std::make_shared<P2pTransportFactory>(
          deviceId, eventBaseThread_->getEventBase());
      factories_.emplace_back(std::move(p2p));
    } else {
      UNIFLOW_LOG_INFO(
          "P2P transport disabled for device {}: {}",
          deviceId_,
          p2pSupported.error().message());
    }
  }
#endif

  auto nics = selectNics();
  if (!nics.empty()) {
    RdmaTransportConfig config;
    config.gidIndex = options_.gidIndex;
    config.trafficClass = options_.trafficClass;
    config.numQps = static_cast<uint32_t>(nics.size());
    auto rdma = std::make_shared<RdmaTransportFactory>(
        std::move(nics), eventBaseThread_->getEventBase(), config);
    factories_.emplace_back(std::move(rdma));
  }

#ifdef UNIFLOW_ENABLE_TCP_TRANSPORT
  // TCP joins the transport pool automatically whenever a routable bind address
  // is available (selectTransport still prefers NVLink/P2P -> RDMA and uses TCP
  // only as the common fallback or when explicitly selected). Registration is
  // skipped when only a loopback address resolves, since advertising loopback
  // in the connect handshake would break cross-host connections for all
  // transports. enableTcp force-registers even on loopback (e.g. same-host
  // testing).
  std::string tcpHost = options_.tcpBindHost;
  if (tcpHost.empty()) {
    tcpHost = resolveTcpBindHost(options_.netdevPrefix);
  }
  bool tcpDevicesOk = true;
  TcpTransportConfig tcpConfig = options_.tcpTransportConfig
      ? *options_.tcpTransportConfig
      : TcpTransportConfig{};
  {
    // Discovery fills bindToDevices when the caller left it empty. A caller
    // that named devices keeps them, but only if discovery would have accepted
    // them: a name that is missing, down, or carries no usable global address
    // would otherwise bind a lane to a NIC that cannot serve it. Rejecting
    // rather than substituting is deliberate -- quietly swapping in a different
    // NIC would hide that the requested one went away.
    const auto usable = enumerateFrontendDevices(
        options_.tcpDevicePrefix, std::numeric_limits<size_t>::max());
    if (tcpConfig.bindToDevices.empty()) {
      tcpConfig.bindToDevices = usable;
      if (tcpConfig.bindToDevices.size() > options_.tcpMaxDevices) {
        tcpConfig.bindToDevices.resize(options_.tcpMaxDevices);
      }
    } else {
      for (const auto& dev : tcpConfig.bindToDevices) {
        if (std::find(usable.begin(), usable.end(), dev) == usable.end()) {
          UNIFLOW_LOG_ERROR(
              "MultiTransport: not registering tcp: configured bind device '{}' "
              "is not a usable '{}' device (needs operstate up and a "
              "non-deprecated global address)",
              dev,
              options_.tcpDevicePrefix);
          tcpDevicesOk = false;
          break;
        }
      }
    }
  }
  if (tcpDevicesOk && (!isLoopbackHost(tcpHost) || options_.enableTcp)) {
    std::string devList;
    for (const auto& d : tcpConfig.bindToDevices) {
      devList += (devList.empty() ? "" : ",") + d;
    }
    UNIFLOW_LOG_INFO(
        "MultiTransport: tcp striping across {} device(s) [{}]",
        tcpConfig.bindToDevices.size(),
        devList);
    auto tcp = std::make_shared<TcpTransportFactory>(
        deviceId,
        eventBaseThread_->getEventBase(),
        std::move(tcpConfig),
        tcpHost);
    factories_.emplace_back(std::move(tcp));
  } else {
    UNIFLOW_LOG_INFO(
        "TCP transport not registered: only a loopback bind address is "
        "available (set tcpBindHost or enableTcp to force)");
  }
#endif
}

void MultiTransport::addTransport(std::unique_ptr<Transport> transport) {
  transports_.emplace_back(std::move(transport));
}

Result<TransportInfo> MultiTransport::bind() {
  const auto numTransport = static_cast<uint8_t>(transports_.size());
  std::vector<std::vector<uint8_t>> infoData;
  infoData.reserve(numTransport);

  size_t totalSize = sizeof(uint8_t);
  totalSize += sizeof(uint32_t) * numTransport;
  for (auto& t : transports_) {
    auto data = t->bind();
    /*
     * A successful bind always yields a non-empty serialized TransportInfo
     * (header + QP/NIC info). An empty result means the underlying transport
     * failed to acquire its resources (CQ/QP/MR) and set itself to Error.
     * Surface that as a real error instead of packing an empty sub-info that
     * the peer would fail to deserialize during connect().
     */
    if (data.empty()) {
      return Err(
          ErrCode::ConnectionFailed,
          "MultiTransport::bind: a transport failed to bind (empty info)");
    }
    infoData.emplace_back(std::move(data));
    totalSize += infoData.back().size();
  }

  std::vector<uint8_t> info(totalSize);
  size_t pos = 0;

  info[pos++] = numTransport;
  for (const auto& data : infoData) {
    auto size = static_cast<uint32_t>(data.size());
    std::memcpy(info.data() + pos, &size, sizeof(size));
    pos += sizeof(size);

    std::memcpy(info.data() + pos, data.data(), size);
    pos += size;
  }

  return info;
}

Status MultiTransport::connect(std::span<const uint8_t> info) {
  if (info.empty()) {
    return Err(ErrCode::ConnectionFailed, "empty transport info");
  }

  size_t pos = 0;
  uint8_t num = info[pos++];
  if (num != transports_.size()) {
    return Err(
        ErrCode::ConnectionFailed,
        "transport count mismatch: local=" +
            std::to_string(transports_.size()) +
            ", peer=" + std::to_string(num));
  }

  size_t headerSize = sizeof(uint32_t);
  for (auto& t : transports_) {
    if (pos + headerSize > info.size()) {
      return Err(
          ErrCode::ConnectionFailed,
          "peer topology info truncated at transport " + t->name() + ": need " +
              std::to_string(headerSize) + " bytes at pos " +
              std::to_string(pos) + ", but only " +
              std::to_string(info.size() - pos) + " remaining");
    }

    uint32_t size = 0;
    std::memcpy(&size, info.data() + pos, sizeof(size));
    pos += sizeof(size);

    if (pos + size > info.size()) {
      return Err(
          ErrCode::ConnectionFailed,
          "peer topology truncated at transport " + t->name() + ": need " +
              std::to_string(size) + " bytes at pos " + std::to_string(pos) +
              ", but only " + std::to_string(info.size() - pos) + " remaining");
    }

    std::span<const uint8_t> infoData(
        info.data() + pos, static_cast<size_t>(size));
    pos += size;

    CHECK_EXPR(t->connect(infoData));
  }
  return Ok();
}

Result<Transport*> MultiTransport::selectTransport(
    const std::vector<TransferRequest>& requests) {
  CHECK_EXPR(validateRequests(requests));

  auto localMemType = requests.front().local.memType();
  auto remoteMemType = requests.front().remote.memType();
  auto localDeviceId = requests.front().local.deviceId();

  auto hasHandleType = [](const auto& handles, TransportType type) {
    for (const auto& h : handles) {
      if (h->transportType() == type) {
        return true;
      }
    }
    return false;
  };

  // A transport is eligible only if every request in the batch has a matching
  // handle on both local and remote sides. This preserves the invariant that a
  // single transport implementation executes the entire batch. Example use
  // case: distributed KV-cache transfer can mix peers where NVLink is available
  // with peers where RDMA is the only common capability; the batch should use
  // NVLink only when all segments support it, otherwise fall back to RDMA.
  auto allHaveTransport = [&](TransportType type) {
    for (const auto& req : requests) {
      if (!hasHandleType(req.local.handles_, type) ||
          !hasHandleType(req.remote.handles_, type)) {
        return false;
      }
    }
    return true;
  };

#ifndef UNIFLOW_ENABLE_TCP_TRANSPORT
  const auto tcpUnavailable = [&]() -> Result<Transport*> {
    return Err(
        ErrCode::NotImplemented,
        "tcp transport is not enabled in this platform build");
  };
  if (preferredTransport_ == TransportType::TCP) {
    return tcpUnavailable();
  }
#endif

  // preferredTransport: global force -- if set and available on all requests,
  // it wins regardless of topology (both intra- and inter-node).
  if (preferredTransport_.has_value() &&
      allHaveTransport(*preferredTransport_)) {
    if (auto* t = findTransport(*preferredTransport_)) {
      return t;
    }
  }

  // Intra-node = VRAM<->VRAM on this device (reachable over the on-host
  // interconnect tier); otherwise inter-node. One transport per batch.
  const bool intraNode = localMemType == MemoryType::VRAM &&
      remoteMemType == MemoryType::VRAM && localDeviceId == deviceId_;

  // Per-case ordered preference, evaluated allocation-free with no transport
  // checked twice: the case override first (if set), then the case defaults
  // skipping whichever tier equals the override.
  //   intra-node: [intraNodeTransport?] NVLink -> RDMA -> TCP
  //   inter-node: [interNodeTransport?] RDMA -> TCP
  const std::optional<TransportType>& caseOverride =
      intraNode ? intraNodeTransport_ : interNodeTransport_;
  auto tryTier = [&](TransportType type) -> Transport* {
    return allHaveTransport(type) ? findTransport(type) : nullptr;
  };

#ifndef UNIFLOW_ENABLE_TCP_TRANSPORT
  if (caseOverride == TransportType::TCP) {
    return tcpUnavailable();
  }
#endif

  if (caseOverride.has_value()) {
    if (auto* t = tryTier(*caseOverride)) {
      return t;
    }
  }
  if (intraNode && caseOverride != TransportType::NVLink) {
    if (auto* t = tryTier(TransportType::NVLink)) {
      return t;
    }
  }
  if (caseOverride != TransportType::RDMA) {
    if (auto* t = tryTier(TransportType::RDMA)) {
      return t;
    }
  }
#ifdef UNIFLOW_ENABLE_TCP_TRANSPORT
  if (caseOverride != TransportType::TCP) {
    if (auto* t = tryTier(TransportType::TCP)) {
      return t;
    }
  }
#else
  if (allHaveTransport(TransportType::TCP)) {
    return tcpUnavailable();
  }
#endif

  return Err(
      ErrCode::NotConnected,
      "no common transport available across all requests");
}

std::future<Status> MultiTransport::doTransfer(
    const std::vector<TransferRequest>& requests,
    const RequestOptions& options,
    TransferOp op) {
  auto transport = selectTransport(requests);
  if (transport.hasError()) {
    return make_ready_future<Status>(std::move(transport).error());
  }

  auto& t = transport.value();
  ++transferCounts_[t->transportType()];
  return (t->*op)(requests, options);
}

std::future<Status> MultiTransport::put(
    const std::vector<TransferRequest>& requests,
    const RequestOptions& options) {
  return doTransfer(requests, options, &Transport::put);
}

std::future<Status> MultiTransport::get(
    const std::vector<TransferRequest>& requests,
    const RequestOptions& options) {
  return doTransfer(requests, options, &Transport::get);
}

// send/recv left as NotImplemented — Phase 6 only uses put/get
std::future<Status> MultiTransport::send(
    RegisteredSegment::Span src,
    const RequestOptions& options) {
  return make_ready_future<Status>(ErrCode::NotImplemented);
}

std::future<Status> MultiTransport::send(
    Segment::Span src,
    const RequestOptions& options) {
  return make_ready_future<Status>(ErrCode::NotImplemented);
}

std::future<Status> MultiTransport::recv(
    RegisteredSegment::Span dst,
    const RequestOptions& options) {
  return make_ready_future<Status>(ErrCode::NotImplemented);
}

std::future<Status> MultiTransport::recv(
    Segment::Span dst,
    const RequestOptions& options) {
  return make_ready_future<Status>(ErrCode::NotImplemented);
}

Result<RegisteredSegment> MultiTransportFactory::registerSegment(
    Segment& segment) {
  RegisteredSegment regSeg(segment);
  for (auto& f : factories_) {
    auto handle = f->registerSegment(segment);
    if (handle) {
      regSeg.handles_.emplace_back(std::move(handle).value());
    } else {
      UNIFLOW_LOG_WARN(
          "Segment {} cannot be registered on transport {}: {}",
          segment.data(),
          toStringView(f->transportType()),
          handle.error().message());
    }
  }
  if (regSeg.handles_.empty()) {
    return Err(
        ErrCode::MemoryRegistrationError,
        "no transport backend could register this segment");
  }
  return regSeg;
}

Result<RemoteRegisteredSegment> MultiTransportFactory::importSegment(
    std::span<const uint8_t> exportId) {
  return RemoteRegisteredSegment::from(
      exportId,
      [this](
          TransportType transportType,
          size_t segmentLength,
          std::span<const uint8_t> payload)
          -> RemoteRegisteredSegment::remoteHandleT {
        for (auto& f : factories_) {
          if (f->transportType() == transportType) {
            return f->importSegment(segmentLength, payload);
          }
        }
        return Err(
            ErrCode::InvalidArgument,
            "importSegment: Invalid transport type " +
                std::to_string(transportType));
      });
}

Result<std::vector<MultiTransportFactory::TopologyEntry>>
MultiTransportFactory::parse(std::span<const uint8_t> peerTopology) {
  if (peerTopology.empty()) {
    return Err(ErrCode::TopologyDisconnect, "empty peer topology");
  }

  size_t pos = 0;
  uint8_t num = peerTopology[pos++];
  std::vector<TopologyEntry> entries(num);

  size_t topoHeaderSize = sizeof(uint8_t) + sizeof(uint32_t);
  for (size_t i = 0; i < entries.size(); ++i) {
    if (pos + topoHeaderSize > peerTopology.size()) {
      return Err(
          ErrCode::TopologyDisconnect,
          "peer topology truncated at transport " + std::to_string(i) +
              ": need " + std::to_string(topoHeaderSize) + " bytes at pos " +
              std::to_string(pos) + ", but only " +
              std::to_string(peerTopology.size() - pos) + " remaining");
    }
    auto transportType = static_cast<TransportType>(peerTopology[pos++]);

    uint32_t topoSize = 0;
    std::memcpy(&topoSize, peerTopology.data() + pos, sizeof(topoSize));
    pos += sizeof(topoSize);

    if (pos + topoSize > peerTopology.size()) {
      return Err(
          ErrCode::TopologyDisconnect,
          "peer topology truncated at transport " + std::to_string(i) +
              ": need " + std::to_string(topoSize) + " bytes at pos " +
              std::to_string(pos) + ", but only " +
              std::to_string(peerTopology.size() - pos) + " remaining");
    }

    entries[i] = {transportType, peerTopology.subspan(pos, topoSize)};
    pos += topoSize;
  }
  return entries;
}

Result<std::unique_ptr<MultiTransport>> MultiTransportFactory::createTransport(
    std::span<const uint8_t> peerTopology) {
  auto parsed = MultiTransportFactory::parse(peerTopology);
  CHECK_RETURN(parsed);
  auto& entries = parsed.value();
  if (entries.size() != factories_.size()) {
    UNIFLOW_LOG_WARN(
        "transport count mismatch: local={}, peer={}",
        factories_.size(),
        entries.size());
  }

  auto mt = std::make_unique<MultiTransport>(
      deviceId_,
      eventBaseThread_,
      options_.intraNodeTransport,
      options_.preferredTransport,
      options_.interNodeTransport);
  for (size_t i = 0, j = 0; i < entries.size() && j < factories_.size();) {
    if (entries[i].type < factories_[j]->transportType()) {
      ++i;
      continue;
    }
    if (entries[i].type > factories_[j]->transportType()) {
      ++j;
      continue;
    }

    auto transport = factories_[j]->createTransport(entries[i].data);
    if (transport) {
      mt->addTransport(std::move(transport).value());
    } else {
      UNIFLOW_LOG_WARN(
          "Transport {} cannot be created: {}",
          factories_[j]->transportType(),
          transport.error().message());
    }
    ++i;
    ++j;
  }
  if (mt->transports_.empty()) {
    return Err(ErrCode::TopologyDisconnect, "no transport can be connected");
  }
  return mt;
}

std::vector<uint8_t> MultiTransportFactory::getTopology() {
  uint8_t numTransport = factories_.size();
  std::vector<std::vector<uint8_t>> topoData;
  topoData.reserve(numTransport);

  // total number of transports
  size_t totalSize = sizeof(uint8_t);
  // header for each transport
  totalSize += (sizeof(uint8_t) + sizeof(uint32_t)) * numTransport;
  for (auto& f : factories_) {
    auto topo = f->getTopology();
    CHECK_THROW_EXCEPTION(!topo.empty(), std::runtime_error);
    auto size = topo.size();
    topoData.emplace_back(std::move(topo));
    CHECK_THROW_EXCEPTION(size > 0, std::runtime_error);
    totalSize += size;
  }

  size_t pos = 0;
  std::vector<uint8_t> topology(totalSize);
  topology[pos++] = numTransport;
  for (size_t i = 0; i < topoData.size(); ++i) {
    // topo transport type
    topology[pos++] = factories_[i]->transportType();

    // topo data size
    auto size = static_cast<uint32_t>(topoData[i].size());
    std::memcpy(topology.data() + pos, &size, sizeof(size));
    pos += sizeof(size);

    // topo data
    std::memcpy(topology.data() + pos, topoData[i].data(), size);
    pos += size;
  }

  return topology;
}

void MultiTransport::shutdown() {
  for (auto& tt : transports_) {
    tt->shutdown();
  }
}

} // namespace uniflow
