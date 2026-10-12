// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/TransportComposition.h"

#include "comms/uniflow/drivers/DeviceAdapter.h"
#include "comms/uniflow/drivers/TopologyDiscovery.h"
#include "comms/uniflow/logging/Logger.h"

#ifndef UNIFLOW_CMAKE_BUILD
#define UNIFLOW_ENABLE_RDMA_TRANSPORT 1
#ifdef __HIP_PLATFORM_AMD__
#define UNIFLOW_ENABLE_P2P_TRANSPORT 1
#else
#define UNIFLOW_ENABLE_NVLINK_TRANSPORT 1
#endif
#endif

#ifdef UNIFLOW_ENABLE_RDMA_TRANSPORT
#include "comms/uniflow/transport/rdma/RdmaTransportProvider.h"
#endif
#ifdef UNIFLOW_ENABLE_TCP_TRANSPORT
#include "comms/uniflow/transport/tcp/TcpTransportProvider.h"
#endif
#if defined(UNIFLOW_ENABLE_NVLINK_TRANSPORT) && !defined(__HIP_PLATFORM_AMD__)
#include "comms/uniflow/transport/nvlink/NVLinkTransportProvider.h"
#elif defined(UNIFLOW_ENABLE_P2P_TRANSPORT)
#include "comms/uniflow/transport/p2p/P2pTransportProvider.h"
#endif

#ifdef UNIFLOW_ENABLE_TCP_TRANSPORT
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace uniflow {

namespace {

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

// True if @p host is empty or loopback. Advertising a loopback bind address in
// the connect handshake would break cross-host connections for all transports,
// so TCP is not auto-registered in that case.
bool isLoopbackHost(const std::string& host) {
  return host.empty() || host == "::1" || host.rfind("127.", 0) == 0 ||
      // IPv4-mapped-IPv6 loopback forms (dotted and pure-hex 127.0.0.0/8).
      host.rfind("::ffff:127.", 0) == 0 || host.rfind("::ffff:7f", 0) == 0;
}
#endif

#ifdef UNIFLOW_ENABLE_RDMA_TRANSPORT
std::vector<std::string> selectCpuNics(
    const Topology& topo,
    const MultiTransportFactoryOptions& options) {
  auto nics = topo.selectCpuNics(options.nicFilter, options.netdevPrefix);
  const auto candidateNicCount = nics.size();
  if (options.cpuNicSelectionPolicy == CpuNicSelectionPolicy::kAll) {
    UNIFLOW_LOG_INFO(
        "CPU RDMA NIC selection policy=all selected_nics={}",
        candidateNicCount);
    return nics;
  }

  auto localNics =
      topo.selectCpuNicsForNumaNodes(options.nicFilter, options.maxCpuNics);
  if (localNics.empty()) {
    localNics = std::move(nics);
    if (options.maxCpuNics > 0 && localNics.size() > options.maxCpuNics) {
      localNics.resize(options.maxCpuNics);
    }
  }

  UNIFLOW_LOG_INFO(
      "CPU RDMA NIC selection policy=numa_local_bounded selected_nics={} "
      "candidate_nics={} numa_nodes={} max_cpu_nics_per_numa={}",
      localNics.size(),
      candidateNicCount,
      topo.numaNodeCount(),
      options.maxCpuNics);
  return localNics;
}

std::vector<std::string> selectRdmaNics(
    const Topology& topo,
    int deviceId,
    const MultiTransportFactoryOptions& options) {
  if (deviceId == -1) {
    return selectCpuNics(topo, options);
  }
  return topo.selectGpuNics(deviceId, options.nicFilter, options.netdevPrefix);
}

#endif

} // namespace

Status MultiTransportFactory::supported(TransportType type) {
  switch (type) {
    case TransportType::RDMA:
#ifdef UNIFLOW_ENABLE_RDMA_TRANSPORT
      return RdmaTransportProvider({}).supported();
#else
      return Err(ErrCode::NotImplemented, "rdma transport is not enabled");
#endif
#if defined(UNIFLOW_ENABLE_NVLINK_TRANSPORT) && !defined(__HIP_PLATFORM_AMD__)
    case TransportType::NVLink:
      return NVLinkTransportProvider().supported();
#elif defined(UNIFLOW_ENABLE_P2P_TRANSPORT)
    case TransportType::NVLink:
      // On AMD the NVLink tier is served by the P2P (XGMI) transport, whose
      // supported() encodes the all-XGMI arch gate.
      return P2pTransportProvider().supported();
#else
    case TransportType::NVLink:
      return Err(ErrCode::NotImplemented, "nvlink transport is not enabled");
#endif
    case TransportType::TCP:
#ifdef UNIFLOW_ENABLE_TCP_TRANSPORT
      return TcpTransportProvider().supported();
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

MultiTransportFactory::MultiTransportFactory(
    int deviceId,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    MultiTransportFactoryOptions options)
    : MultiTransportFactory(
          deviceId,
          options,
          makeDefaultTransportProviders(deviceId, options)) {}

TransportProviderList makeDefaultTransportProviders(
    int deviceId,
    const MultiTransportFactoryOptions& options) {
  TransportProviderList providers;
  auto& topo = sharedTopology();
  CHECK_THROW_EXCEPTION(
      deviceId >= -1 && deviceId < static_cast<int>(topo.gpuCount()),
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
      !((requestsTcp(options.preferredTransport) ||
         requestsTcp(options.intraNodeTransport) ||
         requestsTcp(options.interNodeTransport)) &&
        (options.tcpBindHost.empty() || !options.enableTcp)),
      std::invalid_argument);
#endif

  // Register the intra-node interconnect tier whenever the hardware is present
  // (NVLink on NVIDIA, P2P/XGMI on AMD). This tier and RDMA are both registered
  // when available; selectTransport chooses per transfer (intraNodeTransport
  // can flip the intra-node default -- see selectTransport).
#if defined(UNIFLOW_ENABLE_NVLINK_TRANSPORT) && !defined(__HIP_PLATFORM_AMD__)
  if (deviceId >= 0 && isNvlinkAvailable()) {
    providers.emplace_back(std::make_shared<NVLinkTransportProvider>());
  }
#elif defined(UNIFLOW_ENABLE_P2P_TRANSPORT)
  // AMD: the NVLink tier is served by the P2P (XGMI) transport. Its supported()
  // owns the all-XGMI arch gate (selectTransport is presence-driven; see §5.6).
  if (deviceId >= 0) {
    auto p2pSupported = P2pTransportProvider().supported();
    if (!p2pSupported.hasError()) {
      providers.emplace_back(std::make_shared<P2pTransportProvider>());
    } else {
      UNIFLOW_LOG_INFO(
          "P2P transport disabled for device {}: {}",
          deviceId,
          p2pSupported.error().message());
    }
  }
#endif

#ifdef UNIFLOW_ENABLE_RDMA_TRANSPORT
  auto nics = selectRdmaNics(topo, deviceId, options);
  if (!nics.empty()) {
    RdmaTransportConfig config;
    config.gidIndex = options.gidIndex;
    config.trafficClass = options.trafficClass;
    config.numQps = static_cast<uint32_t>(nics.size());
    providers.emplace_back(
        std::make_shared<RdmaTransportProvider>(std::move(nics), config));
  }
#endif

#ifdef UNIFLOW_ENABLE_TCP_TRANSPORT
  // TCP joins the transport pool automatically whenever a routable bind address
  // is available (selectTransport still prefers NVLink/P2P -> RDMA and uses TCP
  // only as the common fallback or when explicitly selected). Registration is
  // skipped when only a loopback address resolves, since advertising loopback
  // in the connect handshake would break cross-host connections for all
  // transports. enableTcp force-registers even on loopback (e.g. same-host
  // testing).
  std::string tcpHost = options.tcpBindHost;
  if (tcpHost.empty()) {
    tcpHost = resolveTcpBindHost(options.netdevPrefix);
  }
  bool tcpDevicesOk = true;
  TcpTransportConfig tcpConfig = options.tcpTransportConfig
      ? *options.tcpTransportConfig
      : TcpTransportConfig{};
  {
    // Discovery fills bindToDevices when the caller left it empty. A caller
    // that named devices keeps them, but only if discovery would have accepted
    // them: a name that is missing, down, or carries no usable global address
    // would otherwise bind a lane to a NIC that cannot serve it. Rejecting
    // rather than substituting is deliberate -- quietly swapping in a different
    // NIC would hide that the requested one went away.
    const auto usable = enumerateFrontendDevices(
        options.tcpDevicePrefix, std::numeric_limits<size_t>::max());
    if (tcpConfig.bindToDevices.empty()) {
      tcpConfig.bindToDevices = usable;
      // The cap comes off the transport config, which is the single place it is
      // declared: MultiTransportOptions used to carry its own copy, so this and
      // the benchmark could disagree about the default while each looked right.
      // resolveMaxFrontendDevices also clamps to what this host actually has.
      const size_t maxDevices =
          tcpConfig.resolveMaxFrontendDevices(options.tcpDevicePrefix);
      if (tcpConfig.bindToDevices.size() > maxDevices) {
        tcpConfig.bindToDevices.resize(maxDevices);
      }
    } else {
      for (const auto& dev : tcpConfig.bindToDevices) {
        if (std::find(usable.begin(), usable.end(), dev) == usable.end()) {
          UNIFLOW_LOG_ERROR(
              "MultiTransport: not registering tcp: configured bind device '{}' "
              "is not a usable '{}' device (needs operstate up and a "
              "non-deprecated global address)",
              dev,
              options.tcpDevicePrefix);
          tcpDevicesOk = false;
          break;
        }
      }
    }
  }
  if (tcpDevicesOk && (!isLoopbackHost(tcpHost) || options.enableTcp)) {
    std::string devList;
    for (const auto& d : tcpConfig.bindToDevices) {
      devList += (devList.empty() ? "" : ",") + d;
    }
    UNIFLOW_LOG_INFO(
        "MultiTransport: tcp striping across {} device(s) [{}]",
        tcpConfig.bindToDevices.size(),
        devList);
    providers.emplace_back(
        std::make_shared<TcpTransportProvider>(
            std::move(tcpConfig), std::move(tcpHost)));
  } else {
    UNIFLOW_LOG_INFO(
        "TCP transport not registered: only a loopback bind address is "
        "available (set tcpBindHost or enableTcp to force)");
  }
#endif
  return providers;
}

} // namespace uniflow
