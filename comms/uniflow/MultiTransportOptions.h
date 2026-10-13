// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "comms/uniflow/transport/Topology.h"
#include "comms/uniflow/transport/TransportType.h"

namespace uniflow {

// Defined by the optional TCP data transport. Keep this header independent
// of which transports are included in the build.
struct TcpTransportConfig;

enum class CpuNicSelectionPolicy {
  kAll,
  kNumaLocalBounded,
};

struct MultiTransportFactoryOptions {
  NicFilter nicFilter;
  std::string netdevPrefix{"beth"};
  int16_t gidIndex{-1};
  uint8_t trafficClass{0};
  CpuNicSelectionPolicy cpuNicSelectionPolicy{
      CpuNicSelectionPolicy::kNumaLocalBounded};
  size_t maxCpuNics{2};
  // Runtime intra-node transport selection. Intra-node VRAM<->VRAM defaults to
  // the on-host interconnect tier (NVLink on NVIDIA, P2P/XGMI on AMD). Set this
  // to flip that choice -- e.g. TransportType::RDMA to route intra-node traffic
  // over RDMA instead. Both tiers stay registered, so this is a selection
  // override, not a kill-switch. Caller-owned (no transport-internal
  // config-system dependency).
  std::optional<TransportType> intraNodeTransport;
  // preferredTransport: global force, checked first in selectTransport --
  // applies to both intra- and inter-node. interNodeTransport: inter-node
  // override (default RDMA). Each falls through to the remaining tiers if the
  // chosen transport is unavailable on all requests.
  std::optional<TransportType> preferredTransport;
  std::optional<TransportType> interNodeTransport;
  // The TCP data transport auto-registers whenever a routable bind address is
  // available (tcpBindHost if set, else resolveTcpBindHost: netdevPrefix match,
  // else first global address). It is skipped when only loopback resolves,
  // since advertising loopback in the connect handshake breaks cross-host
  // connections for all transports. Set enableTcp to force-register even on
  // loopback (e.g. same-host testing).
  bool enableTcp{false};
  std::string tcpBindHost{};
  // Overrides for the TCP data transport: socket options, lane count
  // (numSocketsPerDevice, applied per device), and the devices lanes bind to.
  // Null keeps the
  // TcpTransportConfig defaults, so this is an override seam rather than
  // required config, and it is the only way to reach lane striping from outside
  // the transport.
  //
  // Held by pointer and forward-declared so this header does not depend on the
  // optional TCP data transport. The layout stays the same in every build.
  std::shared_ptr<const TcpTransportConfig> tcpTransportConfig;
  // Frontend NIC name prefix for TCP lane striping. Must match
  // kDefaultFrontendDevicePrefix in TcpTransport.h, which owns the discovery;
  // this header cannot include that one (see tcpTransportConfig above), so the
  // value is repeated rather than shared. Devices are discovered by
  // name because link speed does not distinguish them -- every candidate here
  // is 200G -- and the backend fabric (beth*) is addressed identically; "eth"
  // is the frontend convention.
  std::string tcpDevicePrefix{"eth"};
  // An omitted request stays unset until actual GPU resource initialization.
  std::optional<std::string> gpuBackend;
};

} // namespace uniflow
