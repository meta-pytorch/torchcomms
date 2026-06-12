// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <unistd.h>
#include <cstring>
#include <fstream>

#include "comms/ctran/commstate/Topology.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran::commstate {

namespace {

constexpr std::string_view kDeviceName = "DEVICE_NAME";
constexpr std::string_view kNetworkTopo = "DEVICE_BACKEND_NETWORK_TOPOLOGY";
constexpr std::string_view kDeviceRackSerial = "DEVICE_RACK_SERIAL";

template <size_t N>
void copyStringToFixedBuffer(char (&dst)[N], const std::string& src) {
  static_assert(N > 0);
  std::strncpy(dst, src.c_str(), N - 1);
  dst[N - 1] = '\0';
}
} // namespace

// DEVICE_BACKEND_NETWORK_TOPOLOGY should be present in all T20 hosts with
// backend network. If not found CTRAN initialization fails.
// Ignore this field for other platform types.
// Expected format for top of rack topology (NSF)
// e.g DEVICE_BACKEND_NETWORK_TOPOLOGY=pci5/pci5.5D.z088//rtsw191.c088.f00.pci5
// Expected format for rail based topology (DSF with scaling unit)
// e.g DEVICE_BACKEND_NETWORK_TOPOLOGY=/snb1.z081/snb1.z081.u015/
// Expected format for DSF without scaling unit (e.g. GB300 DSF at UCO1)
// e.g DEVICE_BACKEND_NETWORK_TOPOLOGY=uco1/uco1.z086//
void parseTopologyValue(
    const std::string& value,
    const std::string& filepath,
    std::string& dc,
    std::string& zone,
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

  if (zone.empty()) {
    return;
  }
  isBackendTopologyValid = true;
}

std::optional<TopologyResult> loadTopology(
    int rank,
    const std::string& filepath) {
  std::ifstream file(filepath);
  std::string line;

  std::string rackSerial;

  std::string dc, zone, host, backendNetworkTopology;
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
      parseTopologyValue(value, filepath, dc, zone, isBackendTopologyValid);
    } else if (key == kDeviceRackSerial) {
      if (value.size() >= ncclx::kMaxNameLen) {
        CLOGF(
            WARN,
            "DEVICE_RACK_SERIAL '{}' exceeds max length {}, ignoring to avoid truncation-based false matches",
            value,
            ncclx::kMaxNameLen);
      } else {
        rackSerial = value;
      }
    }
  }

  if (host.empty()) {
    char hostname[256] = {};
    if (gethostname(hostname, sizeof(hostname) - 1) == 0 &&
        hostname[0] != '\0') {
      host = hostname;
      CLOGF(
          WARN,
          "DEVICE_NAME not found in {}. Falling back to gethostname()={}",
          filepath,
          host);
    } else if (!NCCL_IGNORE_TOPO_LOAD_FAILURE) {
      CLOGF(
          ERR,
          "Failed to load hostname (DEVICE_NAME) from {} and gethostname() fallback failed",
          filepath);
      return std::nullopt;
    } else {
      CLOGF(
          WARN,
          "DEVICE_NAME not found in {} and gethostname() fallback failed. "
          "Continuing because NCCL_IGNORE_TOPO_LOAD_FAILURE=1",
          filepath);
    }
  }

  if (!NCCL_IGNORE_TOPO_LOAD_FAILURE) {
    if (!backendNetworkTopology.empty() && !isBackendTopologyValid) {
      CLOGF(
          ERR,
          "CTRAN cannot be enabled due to missing topology information. "
          "If you think it is safe to proceed, set NCCL_IGNORE_TOPO_LOAD_FAILURE=1 "
          "to ignore this error");
      return std::nullopt;
    }
    if (rackSerial.empty()) {
      CLOGF(
          WARN,
          "DEVICE_RACK_SERIAL not found in {}. "
          "isSameDeviceRack() will always return false, which may affect NVL fabric topology detection",
          filepath);
    }
  }

  ncclx::RankTopology topo{};
  topo.rank = rank;
  topo.pid = getpid();
  copyStringToFixedBuffer(topo.rackSerial, rackSerial);
  copyStringToFixedBuffer(topo.dc, dc);
  copyStringToFixedBuffer(topo.zone, zone);
  copyStringToFixedBuffer(topo.host, host);
  return TopologyResult{topo, std::move(backendNetworkTopology)};
}

} // namespace ctran::commstate
