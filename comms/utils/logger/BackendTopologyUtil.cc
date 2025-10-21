// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/BackendTopologyUtil.h"

#include <unordered_map>

#include <folly/FileUtil.h>
#include <folly/MapUtil.h>
#include <folly/String.h>
#include <folly/logging/xlog.h>

namespace {
std::unordered_map<std::string, std::string> getKeyValuePairsFromFile(
    const std::string& fileName) {
  std::unordered_map<std::string, std::string> keyValuePairs;
  std::string fileContents;
  if (!folly::readFile(fileName.c_str(), fileContents)) {
    XLOG(ERR) << "Failed to read file: " << fileName;
    return keyValuePairs;
  }

  std::vector<std::string> lines;
  folly::split('\n', fileContents, lines);
  for (const auto& line : lines) {
    if (folly::trimWhitespace(line).empty()) {
      continue;
    }
    std::vector<std::string> keyValuePair;
    folly::split('=', line, keyValuePair);
    if (keyValuePair.size() != 2) {
      XLOG(ERR) << "Invalid line in file: " << fileName << ", line: " << line;
      continue;
    }
    keyValuePairs[keyValuePair[0]] = keyValuePair[1];
  }
  return keyValuePairs;
}

void parseScaleUpFields(
    BackendTopologyUtil::Topology& topology,
    const std::vector<std::string>& backendTopo,
    const std::unordered_map<std::string, std::string>& keyValuePairs) {
  auto suRackPtr = folly::get_ptr(keyValuePairs, "DEVICE_RACK_SERIAL");
  if (suRackPtr == nullptr) {
    return;
  }
  topology.scaleUp.rack = *suRackPtr;

  auto suDomainPtr = folly::get_ptr(keyValuePairs, "SCALEUP_DOMAIN");
  if (suDomainPtr == nullptr) {
    return;
  }
  topology.scaleUp.domain = *suDomainPtr;

  // backendTopo is vector of tokens parsed from
  // DEVICE_BACKEND_NETWORK_TOPOLOGY. e.g.
  // DEVICE_BACKEND_NETWORK_TOPOLOGY=nao5/nao5.z081/nao5.z081.u003 will have
  // backendTopo = ["nao5", "nao5.z081", "nao5.z081.u003"]
  if (backendTopo.size() < 2) {
    return;
  }
  topology.scaleUp.unit = backendTopo[2];
}
} // namespace

// Schema is /etc/fbwhoami
/* static */
std::optional<BackendTopologyUtil::Topology>
BackendTopologyUtil::getBackendTopology(const std::string& fileName) {
  auto keyValuePairs = getKeyValuePairsFromFile(fileName);
  if (keyValuePairs.empty()) {
    XLOG(ERR) << "Failed to get backend topology from file: " << fileName;
    return std::nullopt;
  }

  BackendTopologyUtil::Topology topology;

  // SFZ
  auto sfzPtr = folly::get_ptr(keyValuePairs, "SHARED_FATE_ZONE");
  if (sfzPtr == nullptr) {
    return std::nullopt;
  }
  topology.sfz = *sfzPtr;

  // Region
  auto regionPtr = folly::get_ptr(keyValuePairs, "REGION_DATACENTER_PREFIX");
  if (regionPtr == nullptr) {
    return std::nullopt;
  }
  topology.region = *regionPtr;

  // DC
  auto dcPtr = folly::get_ptr(keyValuePairs, "DEVICE_DATACENTER");
  if (dcPtr == nullptr) {
    return std::nullopt;
  }
  topology.dc = *dcPtr;

  // AI Zone + RTSW
  auto backendTopologyPtr =
      folly::get_ptr(keyValuePairs, "DEVICE_BACKEND_NETWORK_TOPOLOGY");
  if (backendTopologyPtr == nullptr) {
    return std::nullopt;
  }
  // Example: pci2/pci2.2A.z085//rtsw049.c085.f00.pci2
  std::vector<std::string> topologyParts;
  folly::split('/', *backendTopologyPtr, topologyParts);
  if (topologyParts.size() != 4) {
    return std::nullopt;
  }
  topology.zone = topologyParts[1];
  topology.rtsw = topologyParts[3];

  // Parse scale-up information when applicable
  auto serverTypePtr =
      folly::get_ptr(keyValuePairs, "DEVICE_LOGICAL_SERVER_SUBTYPE");
  if (serverTypePtr != nullptr &&
      *serverTypePtr == "T20_CIC_GB200_186GB_HBM3E_NVLSO_DSF") {
    parseScaleUpFields(topology, topologyParts, keyValuePairs);
  }

  // Host name
  auto hostPtr = folly::get_ptr(keyValuePairs, "DEVICE_NAME");
  if (hostPtr == nullptr) {
    return std::nullopt;
  }
  topology.host = *hostPtr;

  // full scopes
  for (const auto& scope :
       {topology.sfz,
        topology.region,
        topology.dc,
        topology.zone,
        topology.scaleUp.unit,
        topology.scaleUp.domain,
        topology.scaleUp.rack,
        topology.rtsw,
        topology.host}) {
    if (!scope.empty()) {
      topology.fullScopes.push_back(scope);
    }
  }
  return topology;
}
