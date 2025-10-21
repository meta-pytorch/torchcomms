// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <fstream>

#include "comms/ctran/commstate/Topology.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran::commstate {

namespace {

constexpr std::string_view kDeviceName = "DEVICE_NAME";
constexpr std::string_view kNetworkTopo = "DEVICE_BACKEND_NETWORK_TOPOLOGY";
constexpr std::string_view kDeviceRackSerial = "DEVICE_RACK_SERIAL";
} // namespace

// DEVICE_BACKEND_NETWORK_TOPOLOGY should be present in all T20 hosts with
// backend network. If not found CTRAN initialization fails.
// Ignore this field for other platform types.
// Expected format for top of rack topology
// e.g
// DEVICE_BACKEND_NETWORK_TOPOLOGY=pci5/pci5.5D.z088//rtsw191.c088.f00.pci5
// Expected format for rail based topology
// e.g DEVICE_BACKEND_NETWORK_TOPOLOGY=/snb1.z081/snb1.z081.u015/
void parseTopologyValue(
    const std::string& value,
    const std::string& filepath,
    std::string& dc,
    std::string& zone,
    std::string& su,
    std::string& rtsw,
    bool& isBackendTopologyValid) {
  std::vector<std::string> topologyParts;
  folly::split('/', value, topologyParts);

  // Validate format - should have exactly 4 parts
  if (topologyParts.size() != 4) {
    CLOGF(
        ERR,
        "Invalid topology format: expected 4 parts separated by '/', got {} parts in '{}' from file: {}",
        topologyParts.size(),
        value,
        filepath);
    return;
  }

  dc = std::move(topologyParts[0]);
  zone = std::move(topologyParts[1]);
  su = std::move(topologyParts[2]);
  rtsw = std::move(topologyParts[3]);

  if ((rtsw.empty() && su.empty()) || (!rtsw.empty() && !su.empty()) ||
      zone.empty()) {
    return;
  }
  isBackendTopologyValid = true;
}

void parseRackSerial(const std::string& value, int& rackSerial) {
  try {
    rackSerial = folly::to<int>(value);
  } catch (const std::exception& e) {
    CLOGF(ERR, "Failed to parse rack serial '{}': {}", value, e.what());
  }
}

std::optional<ncclx::RankTopology> loadTopology(
    int rank,
    const std::string& filepath) {
  std::ifstream file(filepath);
  std::string line;

  int rackSerial = -1;

  // currently we don't use rtsw info yet, it's ok to have empty rtsw
  std::string rtsw;

  // Rail based topologies do not have rtsw info, we use scaling unit info
  // instead
  std::string dc, zone, su, host, backendNetworkTopology;
  bool isBackendTopologyValid = false;

  while (std::getline(file, line)) {
    size_t pos = line.find('=');
    if (pos == std::string::npos) {
      // skip if no "=" found
      continue;
    }

    std::vector<std::string> tokens;
    const auto key = line.substr(0, pos);
    const auto value = line.substr(pos + 1);
    if (key == kDeviceName) {
      // e.g DEVICE_NAME=rtptest021.nha1.facebook.com
      host = value;
    } else if (key == kNetworkTopo) {
      backendNetworkTopology = value;
      parseTopologyValue(
          value, filepath, dc, zone, su, rtsw, isBackendTopologyValid);
    } else if (key == kDeviceRackSerial) {
      // If device rack serial is not present, use default value -1
      if (!value.empty()) {
        parseRackSerial(value, rackSerial);
      }
    }
  }

  if (!NCCL_IGNORE_TOPO_LOAD_FAILURE) {
    if (host.empty()) {
      CLOGF(ERR, "Failed to load hostname (DEVICE_NAME) from {}", filepath);
      return std::nullopt;
    } else if (!backendNetworkTopology.empty() && !isBackendTopologyValid) {
      CLOGF(
          ERR,
          "CTRAN cannot be enabled due to missing topology information. "
          "If you think it is safe to proceed, set NCCL_IGNORE_TOPO_LOAD_FAILURE=1 "
          "to ignore this error");
      return std::nullopt;
    }
  }

  ncclx::RankTopology topo;
  topo.rank = rank;
  topo.rackSerial = rackSerial;
  std::strncpy(topo.dc, dc.c_str(), ncclx::kMaxNameLen);
  std::strncpy(topo.zone, zone.c_str(), ncclx::kMaxNameLen);
  std::strncpy(topo.host, host.c_str(), ncclx::kMaxNameLen);
  std::strncpy(topo.rtsw, rtsw.c_str(), ncclx::kMaxNameLen);
  std::strncpy(topo.su, su.c_str(), ncclx::kMaxNameLen);
  return topo;
}

} // namespace ctran::commstate
